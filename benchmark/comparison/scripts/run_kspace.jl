# Section: k-space GRAPPA (MRT only — kept as MRT reference rows).
#
# No cross-toolkit row is possible here, re-verified against each toolkit: BART 0.9.00's command
# list has no `grappa` (its k-space methods are `caldir` / `ecalib` / `sake` / `nlinv` / `pocsense`,
# none of which is autocalibrated GRAPPA kernel fitting), `sigpy.mri.app` ships SENSE / L1-wavelet /
# TV / JSENSE but no GRAPPA app, and MRIReco's reconstruction API exposes only `direct` /
# `multiCoil` solvers. There is nothing to compare against, not merely nothing convenient.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_kspace.jl --threads=N [--use-mkl]
include(joinpath(@__DIR__, "_setup.jl"))

img_mc, kspace_mc, cmap = IMG_MC, KSPACE_MC, CMAP
smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))
mask_g = falses(N, N)
mask_g[:, 1:2:N] .= true
mask_g[:, (N ÷ 2 - 12):(N ÷ 2 + 11)] .= true
acq_g = CartesianAcquisitionInfo(
    NamedDimsArray(kspace_mc[mask_g, :], (:kxy, :coil));
    is3D = false, image_size = (N, N), sensitivity_maps = smaps_mc,
    subsampling = mask_g, shifted_image_dims = (:x, :y),
)

for (meth, cc) in (("GRAPPA (RSS)", RootSumSquares()), ("GRAPPA (Sensitivity)", AdjointSensitivity()))
    println("--> $meth")
    m = GRAPPA(kernel_size = (4, 3), calib_size = (24, 24), coil_combination = cc)
    t, _, x = time_mrt("K-Space", meth, () -> reconstruct(acq_g, m; verbose = false))
    push!(results, BenchResult("K-Space", meth, FW, NUM_THREADS, t * 1000, mag_nrmse(x, img_mc), 0.0))
end

write_section("kspace")
