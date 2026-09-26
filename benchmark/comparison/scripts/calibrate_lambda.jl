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
# reaches its best NRMSE, and its λ is then picked on that ρ's curve. Both grids grow past an edge
# that holds a toolkit's best point (`sweep_toolkit`), since one toolkit's optimum can lie far from
# another's.
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
# How many grid steps `sweep_toolkit` may add past either edge of each axis to reach an interior optimum.
const MAX_GRID_EXTENSIONS = parse(Int, get(ENV, "MAX_GRID_EXTENSIONS", "4"))

"""
    LAMBDA_SHARD

`LAMBDA_SHARD=i/n` (`0 ≤ i < n`): measure only the λ points of the grid whose index is `i` modulo
`n`, with no grid extension, and add them to the stored curve as `--resume` does. `n` processes, one
per shard, then measure a toolkit's whole grid at once; one `--resume` run without a shard follows
and measures only the extensions an edge optimum still needs. `nothing` when unset.
"""
const LAMBDA_SHARD = let s = get(ENV, "LAMBDA_SHARD", "")
    isempty(s) ? nothing : Tuple(parse.(Int, split(s, "/")))
end

"""
    METHOD_FILTER

Parsed from `--methods=m1,m2,...` (method symbols such as `tv`, `lowrank`), or `nothing` for every
regularized method of a case.
"""
const METHOD_FILTER = let i = findfirst(a -> startswith(a, "--methods="), ARGS)
    i === nothing ? nothing : Symbol.(lowercase.(split(ARGS[i][(length("--methods=") + 1):end], ",")))
end

"""
    RESUME

`--resume`: reuse the points already stored in the case's file for the toolkits and methods being
calibrated, measuring only grid points it lacks (the extensions of a widened grid, say). Only a
file written at the same `IT_CAL` is reused. Off by default, since a stored point is stale once
the code or the case behind it changes.
"""
const RESUME = "--resume" in ARGS

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
    # A PDHG row runs as many iterations as its ADMM row applies the operator (`PDHG_ITERATIONS`).
    maxit = haskey(PDHG_METHODS, method) ? IT_CAL * CG_ITERATIONS : IT_CAL
    x = if tk === :mrt
        parent(mrt_reconstructor(c, method; λ, rho = something(ρ, admm_rho(c)), maxit)())
    elseif tk === :bart
        last(bart_run(c, method; λ, maxit, ρ = something(ρ, CMP_RHO)))
    else
        last(toolkit_run(tk, c, method; λ, ρ, maxit, runs = 1))
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
        λs = collect(10 .^ range(log10(centre) - 2, log10(centre) + 1.5; length = ngrid))
        LAMBDA_SHARD === nothing || (λs = λs[(LAMBDA_SHARD[1] + 1):LAMBDA_SHARD[2]:end])
        isempty(λs) && continue
        known = RESUME || LAMBDA_SHARD !== nothing ? stored_points(c, method, key) : Dict{Tuple{Float64, Any}, Float64}()
        checkpoint = pts -> write_calibration!(c, Dict{String, Any}(String(method) => Dict(key => pts)))
        curves[key] = sweep_toolkit(c, method, tk, key, λs, rho_grid(c, method, tk); known, checkpoint)
    end
    return curves
end

# The points of `key`'s stored curve for `method`, keyed by `(λ, ρ)`, when the case's file was
# written at this `IT_CAL`; empty otherwise.
function stored_points(c::BenchCase, method::Symbol, key)
    path = joinpath(LAMBDA_DIR, "$(c.id).json")
    out = Dict{Tuple{Float64, Any}, Float64}()
    isfile(path) || return out
    file = JSON.parsefile(path)
    get(get(file, "meta", Dict()), "iterations", nothing) == IT_CAL || return out
    pts = get(get(get(file, "sweeps", Dict()), String(method), Dict()), key, nothing)
    pts === nothing && return out
    for (λ, ρ, e) in _curve_from_json(pts)
        out[(λ, ρ === nothing ? nothing : Float64(ρ))] = e
    end
    return out
end

"""
    sweep_toolkit(c, method, tk, key, λs, ρs; known) -> [(λ, ρ, nrmse), ...]

One toolkit's curve over the grid `λs × ρs`, extended past whichever edge holds its best point.

The λ grid is shared across toolkits and centred on MRT's λ, but a toolkit that scales its
regularizer or its operator differently can have its optimum a decade or more away: BART's and
SigPy's radial TV optimum lay above the whole shared grid. So while the best point sits on the
largest or smallest λ (or ρ) swept, one more grid step is added in that direction for every ρ (or
every λ), up to `MAX_GRID_EXTENSIONS` times per axis; an optimum still on the edge after that is
logged. `known` holds points measured earlier (see [`RESUME`](@ref)), which are not measured again.

`checkpoint(points)` is called with every point measured so far, known ones included, after each new
measurement, so a run cut short (a job's time limit) leaves its points in the case's file for
`--resume` to pick up.
"""
function sweep_toolkit(
        c::BenchCase, method::Symbol, tk::Symbol, key, λs, ρs;
        known = Dict{Tuple{Float64, Any}, Float64}(), checkpoint = _ -> nothing,
    )
    λs, ρs = copy(λs), copy(ρs)
    nrmse = Dict{Tuple{Float64, Any}, Float64}(known)
    function measure(λ, ρ)
        haskey(nrmse, (λ, ρ)) && return nrmse[(λ, ρ)]
        e = try
            nrmse_at(c, method, tk, λ, ρ)
        catch ex
            @warn "$key $(c.id) $method λ=$λ ρ=$ρ failed" exception = (ex, catch_backtrace())
            NaN
        end
        @info @sprintf("%-40s %-8s %-7s λ=%.4g  ρ=%-9s NRMSE=%.4f", c.id, method, key, λ, something(ρ, "-"), e)
        flush(stderr)
        nrmse[(λ, ρ)] = e
        checkpoint(sort!([(λ, ρ, e) for ((λ, ρ), e) in nrmse]; by = p -> (something(p[2], 0.0), p[1])))
        return e
    end
    curve() = [(λ, ρ, measure(λ, ρ)) for ρ in ρs for λ in λs]
    LAMBDA_SHARD === nothing || return curve()
    λstep = λs[2] / λs[1]
    ρstep = 10.0
    extended = Dict(:λ => 0, :ρ => 0)
    while true
        pts = curve()
        fin = [p for p in pts if isfinite(p[3])]
        isempty(fin) && return pts
        bλ, bρ, _ = fin[argmin([p[3] for p in fin])]
        grew = false
        if extended[:λ] < MAX_GRID_EXTENSIONS && (bλ == last(λs) || bλ == first(λs))
            bλ == last(λs) ? push!(λs, last(λs) * λstep) : pushfirst!(λs, first(λs) / λstep)
            extended[:λ] += 1
            grew = true
        end
        if bρ !== nothing && length(ρs) > 1 && extended[:ρ] < MAX_GRID_EXTENSIONS &&
                (bρ == last(ρs) || bρ == first(ρs))
            bρ == last(ρs) ? push!(ρs, last(ρs) * ρstep) : pushfirst!(ρs, first(ρs) / ρstep)
            extended[:ρ] += 1
            grew = true
        end
        if !grew
            (bλ == last(λs) || bλ == first(λs)) &&
                @warn "$key $(c.id) $method: best λ = $bλ is still at the edge of the grid" λs
            bρ !== nothing && length(ρs) > 1 && (bρ == last(ρs) || bρ == first(ρs)) &&
                @warn "$key $(c.id) $method: best ρ = $bρ is still at the edge of the grid" ρs
            return pts
        end
    end
    return
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
        # Under `--resume` a toolkit's fresh points are added to its stored ones rather than
        # replacing them, so processes that sweep disjoint parts of one toolkit's grid (one ρ decade
        # each, say) all keep their points.
        union = (RESUME || LAMBDA_SHARD !== nothing) && get(get(file, "meta", Dict()), "iterations", nothing) == IT_CAL
        for (method, curves) in fresh
            stored = Dict{String, Any}(tk => _curve_from_json(pts) for (tk, pts) in get(file["sweeps"], method, Dict()))
            merged = merge(stored, curves)
            if union
                for (tk, curve) in curves
                    haskey(stored, tk) || continue
                    pts = Dict((λ, ρ) => e for (λ, ρ, e) in stored[tk])
                    for (λ, ρ, e) in curve
                        pts[(λ, ρ)] = e
                    end
                    merged[tk] = sort!([(λ, ρ, e) for ((λ, ρ), e) in pts]; by = p -> (something(p[2], 0.0), p[1]))
                end
            end
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
        fresh[String(method)] = curves
    end
    isempty(fresh) || write_calibration!(c, fresh)
end
