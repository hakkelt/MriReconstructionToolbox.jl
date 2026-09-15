#!/usr/bin/env julia
#
# mridata.org — 2D knee, fully sampled (`e3573a0f-34f7-4718-827a-027bf9dd4dea`)
#
# A plain fully sampled 2D Cartesian acquisition straight off the scanner: 288 profiles, a
# 768-sample readout, 15 channels, k-space `(:kx, :ky, :coil)`, reconstructed to 768 x 672.
# ~830 MB on disk.
#
# mridata.org serves vendor-exported ISMRMRD, which is the most faithful raw data in the
# catalog — nothing has been cropped, combined or re-gridded — and also the least uniform:
# see the other mridata examples for files with a separate calibration block, an unrecorded
# echo position, or no image content at all.
#
# Run: julia --project=examples examples/mridata/knee_2d_fully_sampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(MRIDATA, "e3573a0f-34f7-4718-827a-027bf9dd4dea")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils", x)
save_image("mridata_knee2d_direct", x)

Ψ = noise_covariance(raw)
acq = Ψ === nothing ? acq : prewhiten(acq, Ψ)

acq, _ = compress_coils(acq, 8; method = SVDCompression())  # returns (acquisition, mixing matrix)
acq = estimate_sensitivities(acq)

xs = reconstruct(acq)
report("SENSE", xs)
save_image("mridata_knee2d_sense", xs)

xcs = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 30))
report("CS, wavelet sparsity", xcs)
save_image("mridata_knee2d_cs", xcs)
