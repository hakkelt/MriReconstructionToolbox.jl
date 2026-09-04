using JSON, Printf

res_dir = "/home/c_mrrht/c_mrrecon/MriReconstructionToolbox/benchmark/comparison/results"

files = [
    ("OpenBLAS 1T", "benchmark_openblas_1threads.json"),
    ("MKL 1T",      "benchmark_mkl_1threads.json"),
    ("OpenBLAS 8T", "benchmark_openblas_8threads.json"),
    ("MKL 8T",      "benchmark_mkl_8threads.json"),
]

data = Dict()
methods = String[]

for (label, fname) in files
    fpath = joinpath(res_dir, fname)
    if isfile(fpath)
        j = JSON.parsefile(fpath)
        for b in j["benchmarks"]
            key = (b["category"], b["method"], b["framework"])
            data[(label, key)] = (b["time_ms"], b["nrmse_gt"], b["nrmse_mrt"])
            if !(b["method"] in methods)
                push!(methods, b["method"])
            end
        end
    end
end

println("==================================================================================================================")
println("                             MULTI-THREAD & MULTI-BLAS BENCHMARK SUMMARY (1 vs 8 THREADS)                         ")
println("==================================================================================================================")
@printf("%-26s | %-16s | %10s | %10s | %10s | %10s | %10s\n", "Method", "Framework", "1T OpenBLAS", "1T MKL", "8T OpenBLAS", "8T MKL", "NRMSE (GT)")
println("------------------------------------------------------------------------------------------------------------------")

for (cat, m) in [
    ("Base 1C", "1-Coil Adjoint"),
    ("Base MC", "Cartesian Adjoint"),
    ("Non-Cartesian", "DCF Adjoint (Gridding)"),
    ("Base MC", "CG-SENSE (10 it)"),
    ("Sparsity", "Total Variation (30 it)"),
    ("Sparsity", "L1-Wavelet (30 it)"),
    ("Sparsity", "TGV (30 it)"),
    ("Dynamic", "Global Low-Rank (20 it)"),
    ("Dynamic", "Locally Low-Rank (20 it)"),
    ("Dynamic", "Temporal TV (20 it)"),
    ("K-Space", "GRAPPA (RSS)"),
    ("K-Space", "GRAPPA (Sensitivity)")
]
    # Find all frameworks for this method
    fws = ["MRT", "BART", "SigPy", "MRIReco"]
    for fw in fws
        # Check if we have data
        t_1_ob = get(data, ("OpenBLAS 1T", (cat, m, fw == "MRT" ? "MRT (OpenBLAS)" : fw == "BART" ? "BART (OpenBLAS)" : fw)), (NaN, NaN, NaN))
        t_1_mkl = get(data, ("MKL 1T", (cat, m, fw == "MRT" ? "MRT (MKL)" : fw == "BART" ? "BART (MKL)" : fw)), (NaN, NaN, NaN))
        t_8_ob = get(data, ("OpenBLAS 8T", (cat, m, fw == "MRT" ? "MRT (OpenBLAS)" : fw == "BART" ? "BART (OpenBLAS)" : fw)), (NaN, NaN, NaN))
        t_8_mkl = get(data, ("MKL 8T", (cat, m, fw == "MRT" ? "MRT (MKL)" : fw == "BART" ? "BART (MKL)" : fw)), (NaN, NaN, NaN))

        if !isnan(t_1_ob[1]) || !isnan(t_1_mkl[1]) || !isnan(t_8_ob[1]) || !isnan(t_8_mkl[1])
            s_1_ob = isnan(t_1_ob[1]) ? "-" : @sprintf("%.2f ms", t_1_ob[1])
            s_1_mkl = isnan(t_1_mkl[1]) ? "-" : @sprintf("%.2f ms", t_1_mkl[1])
            s_8_ob = isnan(t_8_ob[1]) ? "-" : @sprintf("%.2f ms", t_8_ob[1])
            s_8_mkl = isnan(t_8_mkl[1]) ? "-" : @sprintf("%.2f ms", t_8_mkl[1])
            gt_err = !isnan(t_8_mkl[2]) ? t_8_mkl[2] : !isnan(t_8_ob[2]) ? t_8_ob[2] : t_1_mkl[2]
            @printf("%-26s | %-16s | %11s | %10s | %11s | %10s | %10.2e\n", m, fw, s_1_ob, s_1_mkl, s_8_ob, s_8_mkl, gt_err)
        end
    end
    println("------------------------------------------------------------------------------------------------------------------")
end
