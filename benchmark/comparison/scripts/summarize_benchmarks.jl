#!/usr/bin/env julia
# Print the cross-toolbox comparison table from the four merged result files.
#
#   julia --project=benchmark/comparison benchmark/comparison/scripts/summarize_benchmarks.jl
#
# Rows are taken from the results themselves, in the order the benchmarks ran: a method
# renamed by its iteration count ("Total Variation (20 it)") must not silently vanish from
# the table, which is what a hardcoded row list did.

using JSON, Printf

res_dir = normpath(joinpath(@__DIR__, "..", "results"))

files = [
    ("OpenBLAS 1T", "benchmark_openblas_1threads.json"),
    ("MKL 1T", "benchmark_mkl_1threads.json"),
    ("OpenBLAS 8T", "benchmark_openblas_8threads.json"),
    ("MKL 8T", "benchmark_mkl_8threads.json"),
]

# "BART (OpenBLAS)" and "BART (MKL)" are the same toolbox measured in two columns.
strip_backend(fw) = replace(fw, r" \((?:OpenBLAS|MKL)\)" => "")

# In the accuracy race every toolbox runs the iteration count it needs to reach the same NRMSE,
# so the count belongs to the framework, not to the method: "TV (NRMSE<=0.005, 40 it)" / "BART"
# becomes one "TV (NRMSE<=0.005)" row with a "BART (40 it)" column entry, next to MRT's.
function split_iterations(method, fw)
    m = match(r"^(.*) \(NRMSE(.*), (\d+) it\)$", method)
    m === nothing && return (method, fw)
    return ("$(m.captures[1]) (NRMSE$(m.captures[2]))", "$fw ($(m.captures[3]) it)")
end

# A framework that was only compared for accuracy reports a negative sentinel instead of a time.
fmt_ms(t) = (t === nothing || isnan(t) || t < 0) ? "-" : @sprintf("%.2f ms", t)

data = Dict{Tuple{String, Tuple{String, String, String}}, NTuple{3, Float64}}()
rows = Tuple{String, String}[]          # (category, method), in the order first seen
frameworks = Dict{Tuple{String, String}, Vector{String}}()

num(x) = x === nothing ? NaN : Float64(x)

for (label, fname) in files
    fpath = joinpath(res_dir, fname)
    isfile(fpath) || continue
    j = JSON.parsefile(fpath)
    for b in j["benchmarks"]
        cat = b["category"]
        method, fw = split_iterations(b["method"], strip_backend(b["framework"]))
        data[(label, (cat, method, fw))] = (num(b["time_ms"]), num(b["nrmse_gt"]), num(b["nrmse_mrt"]))
        (cat, method) in rows || push!(rows, (cat, method))
        fws = get!(frameworks, (cat, method), String[])
        fw in fws || push!(fws, fw)
    end
end

isempty(rows) && (@info "no merged result files in $res_dir"; exit(0))

const WIDTH = 128
println("="^WIDTH)
println("                             MULTI-THREAD & MULTI-BLAS BENCHMARK SUMMARY (1 vs 8 THREADS)                         ")
println("="^WIDTH)
@printf("%-32s | %-18s | %11s | %10s | %11s | %10s | %10s\n", "Method", "Framework", "1T OpenBLAS", "1T MKL", "8T OpenBLAS", "8T MKL", "NRMSE (GT)")
println("-"^WIDTH)

current_cat = ""
for (cat, method) in rows
    if cat != current_cat
        @printf("%s\n", uppercase(cat))
        global current_cat = cat
    end
    for fw in frameworks[(cat, method)]
        cells = [get(data, (label, (cat, method, fw)), (NaN, NaN, NaN)) for (label, _) in files]
        all(isnan(c[1]) for c in cells) && all(isnan(c[2]) for c in cells) && continue
        gt = something(findfirst(c -> !isnan(c[2]), cells), 0)
        gt_err = gt == 0 ? NaN : cells[gt][2]
        @printf(
            "%-32s | %-18s | %11s | %10s | %11s | %10s | %10.2e\n",
            method, fw, fmt_ms(cells[1][1]), fmt_ms(cells[2][1]), fmt_ms(cells[3][1]), fmt_ms(cells[4][1]), gt_err
        )
    end
    println("-"^WIDTH)
end
