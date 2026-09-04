# Section: CG-SENSE (10 it) on 2D multi-coil brain — MRT vs SigPy vs BART vs MRIReco.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_cgsense.jl --threads=N [--use-mkl]
#
# This is a *cost-per-iteration* row, not an accuracy row: the k-space here is fully sampled
# (`Acc: 1.00`), so the normal equations are trivially conditioned and all four toolkits hit
# NRMSE 0 (to float precision) by iteration 3 and agree exactly at iteration 1 (0.0157).
#
# `tol = 0.0` rather than a small tolerance: with `tol = 1e-14` MRT's CGNR exits as soon as the
# residual underflows, so a nominal 30-iteration MRT run measured *faster* than a 10-iteration one
# while SigPy / MRIReco / BART all ran their full count (they are given `CMP_TOL_INNER = 0`). The
# early exit is a real MRT capability but it is not the same work, so the section forces the full
# count on every toolkit. Measured slope, single thread, iterations 10 → 40: MRT 3.38 ms/it,
# MRIReco 3.32, BART 5.33, SigPy 15.3.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))

img_mc, kspace_mc, cmap = IMG_MC, KSPACE_MC, CMAP
smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))
acq_mc = CartesianAcquisitionInfo(NamedDimsArray(kspace_mc, (:kx, :ky, :coil)); is3D = false, sensitivity_maps = smaps_mc, shifted_image_dims = (:x, :y))

addrow(fw, t, x, xmrt) = push!(results, BenchResult("Base MC", "CG-SENSE (10 it)", fw, NUM_THREADS, t, mag_nrmse(x, img_mc), xmrt === nothing ? 0.0 : mag_nrmse(xmrt, x)))

println("--> CG-SENSE (10 it)")
method_cg = IterativeReconstruction(regularization = (), algorithm = MriReconstructionToolbox.CGNR(maxit = 10, tol = 0.0))
tm, _, xm = time_reconstruction(() -> reconstruct(acq_mc, method_cg; tol = 0.0, maxit = 10, verbose = false))
addrow(FW, tm * 1000, xm, nothing)

try
    ts, xs = sigpy_recon(:cgsense, kspace_mc, cmap; iterations = 10)
    addrow("SigPy", ts, xs, xm)
catch e
    @warn "SigPy CG-SENSE failed" exception = (e, catch_backtrace())
end
try
    tb, _, rb = time_bart("pics -S -w 1 -i 10", ComplexF32.(reshape(kspace_mc, N, N, 1, Nc)), ComplexF32.(reshape(cmap, N, N, 1, Nc)))
    addrow(BART_FW, tb * 1000, rb[:, :, 1], xm)
catch e
    @warn "BART CG-SENSE failed" exception = (e, catch_backtrace())
end
try
    tr, xr = mrireco(:cgsense, kspace_mc, cmap, (N, N); iterations = 10)
    addrow("MRIReco", tr, xr, xm)
catch e
    @warn "MRIReco CG-SENSE failed" exception = (e, catch_backtrace())
end

write_section("cgsense")
