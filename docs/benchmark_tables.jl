# Markdown tables of the committed cross-toolkit benchmark snapshots
# (`benchmark/comparison/results/benchmark_<backend>_<n>threads.json`), for the "Related packages"
# page and the README. Re-exporting the snapshots (`export_snapshot.jl`) updates both.
module BenchmarkTables

using JSON: JSON
using Printf: @sprintf

export comparison_markdown, scaling_markdown, hardware_markdown, readme_table

const RESULTS = joinpath(@__DIR__, "..", "benchmark", "comparison", "results")
const FRAMEWORKS = ("Ristretto", "BART", "SigPy", "MRIReco", "MIRT", "MRpro")

snapshot(backend, threads) = JSON.parsefile(joinpath(RESULTS, "benchmark_$(backend)_$(threads)threads.json"))
available(backend, threads) = isfile(joinpath(RESULTS, "benchmark_$(backend)_$(threads)threads.json"))

# "Ristretto (OpenBLAS)" and "BART (CUDA)" are columns "Ristretto" and "BART"; a variant label
# such as "Ristretto (OpenBLAS) (m=3, σ=1.25)" is not one of the columns.
function framework(label)
    name = replace(label, r" \((OpenBLAS|MKL|CUDA)\)$" => "")
    return name in FRAMEWORKS ? name : nothing
end

const CASE_WORDS = Dict(
    "shepp" => "Shepp–Logan", "logan" => "", "torso" => "torso", "cine" => "cine", "2d" => "2D",
    "3d" => "3D", "multislice" => "multi-slice", "cartesian" => "Cartesian", "radial" => "radial",
    "1ch" => "1 coil", "8ch" => "8 coils",
)
case_label(id) = join(filter(!isempty, [get(CASE_WORDS, w, w) for w in split(id, '_')]), " ")

fmt_ms(t) = t < 0 ? "—" : t < 10 ? @sprintf("%.2f", t) : t < 1000 ? @sprintf("%.0f", t) : @sprintf("%.1f s", t / 1000)

# One row per (case, method), one column per toolkit: time, and the NRMSE against the ground
# truth when `nrmse`. The fastest time in a row is bold.
function comparison_markdown(backend, threads; category, nrmse = true, method_filter = _ -> true)
    available(backend, threads) || return "*No snapshot for $backend at $threads threads.*\n"
    rows = filter(b -> b["category"] == category && method_filter(b["method"]), snapshot(backend, threads)["benchmarks"])
    keys_ = unique((b["case_id"], _method_key(b["method"])) for b in rows)
    cols = [f for f in FRAMEWORKS if any(b -> framework(b["framework"]) == f, rows)]
    io = IOBuffer()
    println(io, "| case | method | ", join(cols, " | "), " |")
    println(io, "|---|---|", repeat("---:|", length(cols)))
    for (case, method) in keys_
        cells = Dict{String, Any}()
        for b in rows
            (b["case_id"], _method_key(b["method"])) == (case, method) || continue
            f = framework(b["framework"])
            isnothing(f) || (cells[f] = b)
        end
        isempty(cells) && continue
        times = [b["time_ms"] for b in values(cells) if b["time_ms"] > 0]
        best = isempty(times) ? -1.0 : minimum(times)
        out = map(cols) do f
            haskey(cells, f) || return "—"
            b = cells[f]
            s = fmt_ms(b["time_ms"]) * (b["time_ms"] < 1000 && b["time_ms"] >= 0 ? " ms" : "")
            b["time_ms"] == best && (s = "**$s**")
            nrmse && b["nrmse_gt"] >= 0 ? s * @sprintf(" (%.3f)", b["nrmse_gt"]) : s
        end
        println(io, "| ", case_label(case), " | ", method, " | ", join(out, " | "), " |")
    end
    return String(take!(io))
end

# An accuracy-race method carries the iteration count it needed: "TV (NRMSE≤0.08, 8 it)" is the
# row "TV (NRMSE≤0.08)" for every toolkit.
_method_key(m) = replace(m, r", \d+ it\)$" => ")")

# Ristretto's time for each (case, method) of `category` at every thread count with a snapshot.
function scaling_markdown(backend; category = "Sparsity", method_filter = m -> startswith(m, "L1-Wavelet"), threads = (1, 2, 4, 8, 16))
    ths = [t for t in threads if available(backend, t)]
    isempty(ths) && return "*No $backend snapshots.*\n"
    data = Dict{Tuple{String, String}, Dict{Int, Float64}}()
    for t in ths, b in snapshot(backend, t)["benchmarks"]
        b["category"] == category && method_filter(b["method"]) && framework(b["framework"]) == "Ristretto" || continue
        get!(data, (b["case_id"], b["method"]), Dict{Int, Float64}())[t] = b["time_ms"]
    end
    io = IOBuffer()
    println(io, "| case | method | ", join(("$t thread" * (t == 1 ? "" : "s") for t in ths), " | "), " | speed-up |")
    println(io, "|---|---|", repeat("---:|", length(ths) + 1))
    for ((case, method), times) in sort!(collect(data); by = first)
        cells = [haskey(times, t) ? fmt_ms(times[t]) * (times[t] < 1000 ? " ms" : "") : "—" for t in ths]
        speedup = haskey(times, ths[1]) && haskey(times, ths[end]) ? @sprintf("%.1f×", times[ths[1]] / times[ths[end]]) : "—"
        println(io, "| ", case_label(case), " | ", method, " | ", join(cells, " | "), " | ", speedup, " |")
    end
    return String(take!(io))
end

# Where and when a snapshot was measured, from its `run` record (snapshot schema 3).
function hardware_markdown(backend, threads)
    available(backend, threads) || return ""
    s = snapshot(backend, threads)
    run = get(s, "run", nothing)
    isnothing(run) && return "*Snapshot schema $(get(s, "schema_version", 1)): no hardware record.*\n"
    part(k) = join(get(run, k, String[]), ", ")
    cpu = isempty(part("cpu_model")) ? "unrecorded CPU" : part("cpu_model")
    gpu = isempty(part("gpu")) ? "" : ", GPU $(part("gpu"))"
    return "$(backend), $threads thread(s): $cpu$gpu; Julia $(part("julia_version")); " *
        "BLAS $(part("blas_vendor")); measured $(run["dates"][1]) to $(run["dates"][2]).\n"
end

# The README's table: the accuracy race at one backend and thread count, times only.
readme_table(backend = "openblas", threads = 8) = comparison_markdown(backend, threads; category = "Accuracy race", nrmse = false)

end
