# Per-case, per-toolkit λ calibration for the matched-accuracy comparison.
#
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/calibrate_lambda.jl --threads=N [--use-mkl] [--cases=...] [--data=...]
#
# For every catalog case (synthetic by default; `--cases=` narrows by case id or method label), every
# regularized method the case admits, and every toolkit that `supports` it, sweep λ over a wide log
# grid, run the solver to (near) convergence (`IT_CAL` outer iterations), and record NRMSE against
# the case's reference. BART's regularisation weight is on a different internal scale than the
# others (it rescales the data internally), and the four TV functionals differ, so a common λ is
# meaningless; instead the target NRMSE is what MRT reaches at its own best λ, and every other
# toolkit's λ is the grid point whose converged NRMSE is closest to that.
#
# Results go to `results/lambda/<case id>.json`, one file per case, which the sections read back
# through `load_lambda` (a real case falls back to its synthetic analogue's file: k-space is unit-RMS
# normalised everywhere, so λ transfers). Each file also records `race_target`: the worst toolkit's
# best NRMSE × 1.10, the target `run_accuracy_race.jl` races to, reachable by every toolkit.
#
# Noise is what makes λ > 0 optimal at all. On a noiseless phantom every regularizer is pure bias
# (NRMSE decreases monotonically as λ → 0 and CG-SENSE beats all of them), so there is no operating
# point to calibrate; every catalog case carries `MRT_BENCH_SNR_DB` noise for that reason.
#
# One case per process parallelises trivially: `benchmark/slurm/calibrate.sh` submits an array job.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
include(joinpath(@__DIR__, "_methods.jl"))

const IT_CAL = parse(Int, get(ENV, "IT_CAL", "30"))   # above production's 20 — converged NRMSE(λ)
const NGRID = parse(Int, get(ENV, "NGRID", "8"))
const NGRID_HEAVY = parse(Int, get(ENV, "NGRID_HEAVY", "6"))

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

function nrmse_at(c::BenchCase, method::Symbol, tk::Symbol, λ::Real)
    x = if tk === :mrt
        parent(mrt_reconstructor(c, method; λ, maxit = IT_CAL)())
    elseif tk === :bart
        last(bart_run(c, method; λ, maxit = IT_CAL))
    else
        last(toolkit_run(tk, c, method; λ, maxit = IT_CAL, runs = 1))
    end
    return mag_nrmse(_score_image(c, method, x), c.reference)
end

"""
    sweep(c, method) -> Dict(toolkit => [(λ, nrmse), ...])
"""
function sweep(c::BenchCase, method::Symbol)
    λc = default_lambda(c, method)
    ngrid = c.heavy ? NGRID_HEAVY : NGRID
    curves = Dict{String, Vector{Tuple{Float64, Float64}}}()
    tks = [:mrt; [tk for tk in COMPETITORS if supports(tk, c, method) && should_run_framework(framework_label(tk))]]
    for tk in tks
        key = tk === :mrt ? "MRT" : toolkit_key(tk)
        centre = grid_centre(method, key, λc)
        pts = Tuple{Float64, Float64}[]
        for λ in 10 .^ range(log10(centre) - 2, log10(centre) + 1.5; length = ngrid)
            e = try
                nrmse_at(c, method, tk, λ)
            catch ex
                @warn "$key $(c.id) $method λ=$λ failed" exception = (ex, catch_backtrace())
                NaN
            end
            @info @sprintf("%-40s %-8s %-7s λ=%.4g  NRMSE=%.4f", c.id, method, key, λ, e)
            push!(pts, (λ, e))
        end
        curves[key] = pts
    end
    return curves
end

"""Best (lowest) finite NRMSE on a curve."""
best(curve) = minimum(e for (_, e) in curve if isfinite(e); init = Inf)

"""λ on `curve` whose NRMSE is closest to `target`."""
function pick_lambda(curve, target)
    fin = [(λ, e) for (λ, e) in curve if isfinite(e)]
    isempty(fin) && return NaN
    return fin[argmin(abs(e - target) for (_, e) in fin)][1]
end

mkpath(LAMBDA_DIR)
for c in section_cases(c -> !isempty(regularized_methods(c)))
    lambda = Dict{String, Any}()
    target_nrmse = Dict{String, Any}()
    race_target = Dict{String, Any}()
    sweeps = Dict{String, Any}()
    for method in regularized_methods(c)
        should_run(c.id, METHOD_LABEL[method]) || continue
        curves = sweep(c, method)
        # Target = the NRMSE MRT reaches at its own best λ (MRT is the reference implementation);
        # every other toolkit's λ is then chosen to match MRT's accuracy. If a toolkit cannot reach
        # that NRMSE anywhere on the grid, `pick_lambda` returns its closest (best) point.
        target = best(curves["MRT"])
        picks = Dict(tk => pick_lambda(curve, target) for (tk, curve) in curves)
        worst = maximum(best(curve) for curve in values(curves))
        @info "calibrated" c.id method target picks worst
        lambda[String(method)] = picks
        target_nrmse[String(method)] = target
        race_target[String(method)] = isfinite(worst) ? 1.1 * worst : nothing
        sweeps[String(method)] = Dict(tk => [[λ, e] for (λ, e) in curve] for (tk, curve) in curves)
    end
    isempty(lambda) && continue
    path = joinpath(LAMBDA_DIR, "$(c.id).json")
    # Merge into an existing file so a `--cases=` rerun of one method keeps the others.
    old = isfile(path) ? JSON.parsefile(path) : Dict{String, Any}()
    for (k, v) in (("lambda", lambda), ("target_nrmse", target_nrmse), ("race_target", race_target), ("sweeps", sweeps))
        old[k] = merge(get(old, k, Dict{String, Any}()), v)
    end
    old["meta"] = Dict(
        "case_id" => c.id, "data_source" => c.source, "iterations" => IT_CAL,
        "small" => small_mode(), "cine_frames" => cine_frames(),
        "backend" => USE_MKL ? "mkl" : "openblas", "threads" => NUM_THREADS,
        "git" => Dict(pairs(git_ref(normpath(joinpath(@__DIR__, "..", "..", ".."))))),
    )
    open(io -> JSON.print(io, old, 4), path, "w")
    @info "wrote" path
end
