# Method specs and the generic row driver of the comparison suite. Every section is a loop over
# catalog cases (benchmark/utils/cases.jl) calling `run_method_rows!`; this file and `_toolkits.jl`
# hold everything toolkit-specific, and no section prepares data of its own.
#
# Assumes `_setup.jl` and `_toolkits.jl` have been included.

"""
    METHOD_LABEL

The row label of each method (`method` column of a result).
"""
const METHOD_LABEL = Dict(
    :adjoint => "Adjoint", :gridding => "DCF Adjoint (Gridding)", :cgsense => "CG-SENSE",
    :tv => "Total Variation", :wavelet => "L1-Wavelet", :tgv => "TGV",
    :lowrank => "Global Low-Rank", :llr => "Locally Low-Rank", :ttv => "Temporal TV",
)

"""
    method_label(method, maxit) -> String

`"Total Variation (20 it)"` for an iterative method, the bare label for a direct one.
"""
method_label(m::Symbol, maxit) = m in (:adjoint, :gridding) ? METHOD_LABEL[m] : "$(METHOD_LABEL[m]) ($maxit it)"

default_maxit(m::Symbol) = m === :cgsense ? CG_ITERATIONS : OUTER_ITERATIONS

"""
    regularized_methods(c) -> Vector{Symbol}

The methods of case `c` that carry a λ: `applicable_methods` without the direct reconstruction
and CG-SENSE. These are what `calibrate_lambda.jl` sweeps.
"""
regularized_methods(c::BenchCase) = filter(m -> !(m in (:adjoint, :gridding, :cgsense)), applicable_methods(c))

"""
    race_methods(c) -> Vector{Symbol}

The methods the accuracy race runs on case `c`: every regularized method of a single-slice
Cartesian case and of the Cartesian cine, and only the first (TV, or global low rank for the cine)
elsewhere, where one row already answers the question and each ladder is expensive.
"""
function race_methods(c::BenchCase)
    ms = regularized_methods(c)
    (c.trajectory === :cartesian && c.family in (:single_slice, :cine)) && return ms
    return ms[1:1]
end

# ---------------------------------------------------------------- λ

"""
    LAMBDA_DIR

`results/lambda/`, one calibration file per case; under `MRT_BENCH_SMALL=1` the separate
`results/lambda_small/`, so a smoke calibration of the shrunken cases never overwrites the real one.
"""
const LAMBDA_DIR = normpath(joinpath(@__DIR__, "..", "results", small_mode() ? "lambda_small" : "lambda"))

"""
    load_lambda(c::BenchCase, method, toolkit, default) -> Float64

Per-toolkit λ for `method` on case `c`, from `LAMBDA_DIR/<case id>.json` (written by
`calibrate_lambda.jl`); failing that from the case's synthetic analogue's file (a real case uses the
λ of the synthetic case it mirrors -- valid because both k-spaces are unit-RMS normalised); failing
that from the pre-catalog `results/lambda_calibration.json`; failing that `default`.
"""
function load_lambda(c::BenchCase, method::Symbol, toolkit::AbstractString, default::Real)
    for id in unique((c.id, c.analogue))
        v = _lambda_from(joinpath(LAMBDA_DIR, "$id.json"), method, toolkit)
        v === nothing || return v
    end
    v = _lambda_from(normpath(joinpath(@__DIR__, "..", "results", "lambda_calibration.json")), method, toolkit)
    return v === nothing ? Float64(default) : v
end

function _lambda_from(path, method, toolkit)
    isfile(path) || return nothing
    tbl = try
        JSON.parsefile(path)["lambda"]
    catch
        return nothing
    end
    m = get(tbl, String(method), nothing)
    (m === nothing || !haskey(m, toolkit)) && return nothing
    v = m[toolkit]
    return (v isa Real && isfinite(v)) ? Float64(v) : nothing
end

"""
    load_race_target(c, method, default) -> Float64

The accuracy race's NRMSE target for `method` on `c`: `race_target` from the case's λ file (the
worst toolkit's best converged NRMSE × 1.10, so every toolkit can reach it), else `default`.
"""
function load_race_target(c::BenchCase, method::Symbol, default::Real)
    for id in unique((c.id, c.analogue))
        path = joinpath(LAMBDA_DIR, "$id.json")
        isfile(path) || continue
        t = try
            get(get(JSON.parsefile(path), "race_target", Dict()), String(method), nothing)
        catch
            nothing
        end
        t isa Real && isfinite(t) && return Float64(t)
    end
    return Float64(default)
end

# ---------------------------------------------------------------- cases of a section

"""
    section_cases(pred) -> Vector{BenchCase}

The catalog cases selected by `--data=` for which `pred(case)` holds. A case that cannot be
prepared (a real dataset that is unavailable) is logged and left out.
"""
function section_cases(pred)
    out = BenchCase[]
    for id in case_ids(; real = DATA in ("real", "all"), synthetic = DATA in ("synthetic", "all"))
        should_run_case(id) || continue
        c = try
            get_case(id)
        catch err
            @warn "case $id could not be prepared, skipping it" exception = (err, catch_backtrace())
            continue
        end
        pred(c) && push!(out, c)
    end
    return out
end

# ---------------------------------------------------------------- the row driver

# A direct reconstruction returns Σ conj(Sᶜ) xᶜ; it is scored after dividing by Σ|Sᶜ|², the
# unfolding of a fully sampled acquisition, for every toolkit alike.
function _score_image(c::BenchCase, method::Symbol, x)
    x = Array(x)
    if method in (:adjoint, :gridding) && c.smaps !== nothing
        cdim = c.family === :volume ? 4 : 3
        x = x ./ (dropdims(sum(abs2, c.smaps; dims = cdim); dims = cdim) .+ eps(Float32))   # broadcasts over frames
    end
    return x
end

"""
    run_method_rows!(section, c, method; maxit, toolkits, λ_default) -> MRT image or nothing

Time MRT's reconstruction of case `c` by `method` (with `mrt_reconstructor`, the same call the MRT
harness times), then every competitor in `toolkits` that `supports` it, and push one
`BenchResult` per toolkit. `nrmse_gt` is against the case's reference, `nrmse_mrt` against MRT's
own result. A competitor that throws is logged and dropped from the row.
"""
function run_method_rows!(
        section::AbstractString, c::BenchCase, method::Symbol;
        maxit::Int = default_maxit(method), toolkits = COMPETITORS,
        λ_default::Real = get(DEFAULT_LAMBDA, method, 0.0),
    )
    label = method_label(method, maxit)
    should_run(c.id, label) || should_run(section, label) || return nothing
    println("--> $(c.id): $label")
    runs = timed_runs(c)
    λ = load_lambda(c, method, "MRT", λ_default)
    tm, _, xm = time_run(mrt_reconstructor(c, method; λ, maxit); runs)
    xm = _score_image(c, method, parent(xm))
    push!(results, BenchResult(section, label, FW, NUM_THREADS, tm * 1000, mag_nrmse(xm, c.reference), 0.0, c.id, c.source))
    for tk in toolkits
        fw = framework_label(tk)
        (supports(tk, c, method) && should_run_framework(fw)) || continue
        try
            t_ms, x = toolkit_run(tk, c, method; λ = load_lambda(c, method, toolkit_key(tk), λ_default), maxit, runs)
            x = _score_image(c, method, x)
            push!(results, BenchResult(section, label, fw, NUM_THREADS, t_ms, mag_nrmse(x, c.reference), mag_nrmse(xm, x), c.id, c.source))
        catch err
            @warn "$fw $label on $(c.id) failed" exception = (err, catch_backtrace())
        end
    end
    flush_results!(lowercase(replace(section, r"[^A-Za-z0-9]+" => "_")))
    return xm
end
