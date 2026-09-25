# Per-case, per-toolkit λ and ADMM-penalty calibration for the matched-accuracy comparison.
#
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/calibrate_lambda.jl --threads=N \
#       [--use-mkl] [--cases=...] [--data=...] [--frameworks=mrt,bart,...] [--methods=tv,lowrank,...]
#
# For every catalog case (synthetic by default; `--cases=` narrows by case id or method label), every
# regularized method the case admits (`--methods=` narrows by method symbol), and every toolkit that
# `supports` it, sweep λ over a wide log grid, run the solver to (near) convergence (`IT_CAL` outer
# iterations), and record NRMSE against the case's reference. BART's regularisation weight is on a
# different internal scale than the others (it rescales the data internally), and the four TV
# functionals differ, so a common λ is meaningless; instead the target NRMSE is what MRT reaches at
# its own best λ, and every other toolkit's λ is the grid point whose converged NRMSE is closest to
# that.
#
# A row that runs a fixed-penalty ADMM (`uses_admm`) sweeps ρ as well, on a grid of decades
# (`RHO_DECADES`) around the toolkit's uncalibrated default: `admm_rho(c)` for MRT, whose ρ is
# relative to `‖𝒜‖²`, and `CMP_RHO` for the others, whose ρ is absolute in their own operator
# scaling. One number cannot mean the same thing to all of them, so each gets the ρ at which it
# reaches its best NRMSE, and its λ is then picked on that ρ's curve. A best ρ at the edge of the
# grid is logged, since the optimum may lie beyond it.
#
# `--frameworks=` selects the toolkits to (re)calibrate, by case-insensitive substring of their
# label, `mrt` included; without it every toolkit runs. The curves of the toolkits left out are
# read back from the case's existing file, so the target, the picks and `race_target` are always
# computed over every toolkit calibrated so far. A toolkit calibrated before MRT gets its best λ
# until MRT's curve arrives, since there is no target to match yet.
#
# Results go to `results/lambda/<case id>.json`, one file per case, which the sections read back
# through `load_lambda` and `load_rho` (a real case falls back to its synthetic analogue's file:
# k-space is unit-RMS normalised everywhere, so λ transfers). Each file also records `race_target`:
# the worst toolkit's best NRMSE × 1.10, the target `run_accuracy_race.jl` races to, reachable by
# every toolkit. The file is merged under a lock, so several processes may calibrate different
# toolkits or methods of one case at once.
#
# Noise is what makes λ > 0 optimal at all. On a noiseless phantom every regularizer is pure bias
# (NRMSE decreases monotonically as λ → 0 and CG-SENSE beats all of them), so there is no operating
# point to calibrate; every catalog case carries `MRT_BENCH_SNR_DB` noise for that reason.
#
# One case per process parallelises trivially: `benchmark/slurm/calibrate.sh` submits an array job.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
include(joinpath(@__DIR__, "_methods.jl"))
using FileWatching.Pidfile: mkpidlock

const IT_CAL = parse(Int, get(ENV, "IT_CAL", "30"))   # above production's 20 — converged NRMSE(λ)
const NGRID = parse(Int, get(ENV, "NGRID", "8"))
const NGRID_HEAVY = parse(Int, get(ENV, "NGRID_HEAVY", "6"))
# Decades of the ρ grid around each toolkit's default, e.g. "-2,-1,0,1,2" = ρ₀ · 10^(-2:2).
const RHO_DECADES = parse.(Float64, split(get(ENV, "RHO_DECADES", "-2,-1,0,1,2"), ","))

"""
    METHOD_FILTER

Parsed from `--methods=m1,m2,...` (method symbols such as `tv`, `lowrank`), or `nothing` for every
regularized method of a case.
"""
const METHOD_FILTER = let i = findfirst(a -> startswith(a, "--methods="), ARGS)
    i === nothing ? nothing : Symbol.(lowercase.(split(ARGS[i][(length("--methods=") + 1):end], ",")))
end

"""
    calibrates(key) -> Bool

Whether toolkit `key` (`"MRT"`, `"BART"`, ...) is recalibrated in this run: every toolkit without
`--frameworks=`, otherwise those whose label a pattern is a substring of. Unlike the timing sections,
MRT can be left out here, and is then read back from the case's file like any other toolkit.
"""
function calibrates(tk::Symbol)
    tk === :mrt && return FRAMEWORK_FILTER === nothing || any(p -> occursin(p, "mrt"), FRAMEWORK_FILTER)
    return should_run_framework(framework_label(tk))
end

# Every sweep point is one untimed solve.
WARMUP[] = 0
RUNS[] = 1

"""
    grid_centre(method, toolkit, λc) -> Float64

The centre of a toolkit's λ grid. Every toolkit shares `λc` except where its *operating point*
differs enough that the shared decade is the wrong one — and one does.

MIRT's global low-rank row runs POGM at `proxgrad_budget(IT_CAL)` iterations (see that function),
where the other toolkits run ADMM. Early stopping is itself a regularizer, so a solver that runs
to convergence wants a larger λ than one stopped at 20 iterations: measured on the old 64² dynamic
phantom at the matched budget with a valid step size, the optimum was at λ ≈ 3–10, while the shared
grid stops at 0.316. Its calibrated λ used to come back pinned to that ceiling with a flat curve —
the sweep could not see the optimum. This shifts MIRT's grid up to cover it.
"""
function grid_centre(method, toolkit, λc)
    (method === :lowrank && toolkit == "MIRT") && return 3.0
    return λc
end

"""
    rho_grid(c, method, tk) -> Vector

The ADMM penalties swept for toolkit `tk` on `method`: `RHO_DECADES` around its default for a row
that [`uses_admm`](@ref), `[nothing]` (the row has no ρ) otherwise.
"""
function rho_grid(c::BenchCase, method::Symbol, tk::Symbol)
    uses_admm(tk, c, method) || return Any[nothing]
    ρ₀ = tk === :mrt ? admm_rho(c) : CMP_RHO
    return Any[ρ₀ * 10^d for d in RHO_DECADES]
end

function nrmse_at(c::BenchCase, method::Symbol, tk::Symbol, λ::Real, ρ)
    x = if tk === :mrt
        parent(mrt_reconstructor(c, method; λ, rho = something(ρ, admm_rho(c)), maxit = IT_CAL)())
    elseif tk === :bart
        last(bart_run(c, method; λ, maxit = IT_CAL, ρ = something(ρ, CMP_RHO)))
    else
        last(toolkit_run(tk, c, method; λ, ρ, maxit = IT_CAL, runs = 1))
    end
    return mag_nrmse(_score_image(c, method, x), c.reference)
end

"""
    sweep(c, method) -> Dict(toolkit => [(λ, ρ, nrmse), ...])

The curves of the toolkits this run [`calibrates`](@ref); `ρ` is `nothing` for a row without one.
"""
function sweep(c::BenchCase, method::Symbol)
    λc = default_lambda(c, method)
    ngrid = c.heavy ? NGRID_HEAVY : NGRID
    curves = Dict{String, Vector{Tuple{Float64, Any, Float64}}}()
    tks = [tk for tk in (:mrt, COMPETITORS...) if (tk === :mrt || supports(tk, c, method)) && calibrates(tk)]
    for tk in tks
        key = tk === :mrt ? "MRT" : toolkit_key(tk)
        centre = grid_centre(method, key, λc)
        pts = Tuple{Float64, Any, Float64}[]
        for ρ in rho_grid(c, method, tk), λ in 10 .^ range(log10(centre) - 2, log10(centre) + 1.5; length = ngrid)
            e = try
                nrmse_at(c, method, tk, λ, ρ)
            catch ex
                @warn "$key $(c.id) $method λ=$λ ρ=$ρ failed" exception = (ex, catch_backtrace())
                NaN
            end
            @info @sprintf("%-40s %-8s %-7s λ=%.4g  ρ=%-9s NRMSE=%.4f", c.id, method, key, λ, something(ρ, "-"), e)
            push!(pts, (λ, ρ, e))
        end
        curves[key] = pts
    end
    return curves
end

"""Best (lowest) finite NRMSE on a curve."""
best(curve) = minimum(e for (_, _, e) in curve if isfinite(e); init = Inf)

"""The ρ of a curve's best point: the penalty the toolkit is calibrated to (`nothing` if none)."""
function best_rho(curve)
    fin = [(ρ, e) for (_, ρ, e) in curve if isfinite(e)]
    isempty(fin) && return nothing
    return fin[argmin(last.(fin))][1]
end

"""
λ on the best ρ's slice of `curve` whose NRMSE is closest to `target`, or that slice's best λ when
there is no target yet.
"""
function pick_lambda(curve, target)
    ρ = best_rho(curve)
    fin = [(λ, e) for (λ, r, e) in curve if isfinite(e) && isequal(r, ρ)]
    isempty(fin) && return NaN
    score = target === nothing ? last.(fin) : [abs(e - target) for (_, e) in fin]
    return fin[argmin(score)][1]
end

# Sweeps as stored: `[λ, ρ, nrmse]` with `ρ = null` for a row without one; a file from before ρ
# calibration holds `[λ, nrmse]`.
_curve_from_json(pts) = [length(p) == 2 ? (Float64(p[1]), nothing, Float64(p[2])) : (Float64(p[1]), p[2], Float64(p[3])) for p in pts]
_curve_to_json(curve) = [[λ, ρ, e] for (λ, ρ, e) in curve]

function edge_warning(c, method, key, curve)
    ρs = unique(r for (_, r, _) in curve if r !== nothing)
    length(ρs) > 1 || return nothing
    ρ = best_rho(curve)
    (ρ == minimum(ρs) || ρ == maximum(ρs)) &&
        @warn "$key $(c.id) $method: best ρ = $ρ is at the edge of the grid; widen RHO_DECADES" ρs
    return nothing
end

"""
    write_calibration!(c, fresh)

Merge `fresh` (`method => curves` from this run) into the case's file under a lock, then recompute
every derived field of each method touched from the merged curves of all toolkits.
"""
function write_calibration!(c::BenchCase, fresh::Dict{String, Any})
    path = joinpath(LAMBDA_DIR, "$(c.id).json")
    return mkpidlock(path * ".lock"; stale_age = 4 * 3600) do
        file = isfile(path) ? JSON.parsefile(path) : Dict{String, Any}()
        for k in ("lambda", "rho", "target_nrmse", "race_target", "sweeps")
            haskey(file, k) || (file[k] = Dict{String, Any}())
        end
        for (method, curves) in fresh
            stored = Dict{String, Any}(tk => _curve_from_json(pts) for (tk, pts) in get(file["sweeps"], method, Dict()))
            merged = merge(stored, curves)
            # Target = the NRMSE MRT reaches at its own best λ and ρ (MRT is the reference
            # implementation); every other toolkit's λ is then chosen to match MRT's accuracy. If a
            # toolkit cannot reach that NRMSE anywhere on the grid, `pick_lambda` returns its closest
            # (best) point.
            target = haskey(merged, "MRT") ? best(merged["MRT"]) : nothing
            picks = Dict(tk => pick_lambda(curve, target) for (tk, curve) in merged)
            rhos = Dict(tk => best_rho(curve) for (tk, curve) in merged if best_rho(curve) !== nothing)
            worst = maximum(best(curve) for curve in values(merged))
            @info "calibrated" c.id method target picks rhos worst
            file["lambda"][method] = picks
            file["rho"][method] = rhos
            file["target_nrmse"][method] = target
            file["race_target"][method] = isfinite(worst) ? 1.1 * worst : nothing
            file["sweeps"][method] = Dict(tk => _curve_to_json(curve) for (tk, curve) in merged)
        end
        file["meta"] = Dict(
            "case_id" => c.id, "data_source" => c.source, "iterations" => IT_CAL,
            "rho_decades" => RHO_DECADES,
            "small" => small_mode(), "cine_frames" => cine_frames(),
            "backend" => USE_MKL ? "mkl" : "openblas", "threads" => NUM_THREADS,
            "git" => Dict(pairs(git_ref(normpath(joinpath(@__DIR__, "..", "..", ".."))))),
        )
        open(io -> JSON.print(io, file, 4), path, "w")
        @info "wrote" path
    end
end

mkpath(LAMBDA_DIR)
for c in section_cases(c -> !isempty(regularized_methods(c)))
    fresh = Dict{String, Any}()
    for method in regularized_methods(c)
        should_run(c.id, METHOD_LABEL[method]) || continue
        METHOD_FILTER === nothing || method in METHOD_FILTER || continue
        curves = sweep(c, method)
        isempty(curves) && continue
        for (key, curve) in curves
            edge_warning(c, method, key, curve)
        end
        fresh[String(method)] = curves
    end
    isempty(fresh) || write_calibration!(c, fresh)
end
