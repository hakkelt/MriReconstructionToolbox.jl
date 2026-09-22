#!/usr/bin/env julia
#
# OCMR — fully sampled 0.55 T cardiac cine (`fs_0152_0_55T`)
#
# The fully sampled half of the OCMR catalog: 1540 profiles, 384 samples, 15 channels, 14
# cardiac frames, acquired on a 0.55 T scanner. Layout is the same as for the undersampled
# OCMR data, `(:kx, :ky, :coil, :time)`, but because every phase-encoding line is present the
# direct reconstruction is already the reference image — which makes this the file to use
# when checking what an accelerated reconstruction costs in image quality. Retrospectively
# undersample it with `create_sampling_pattern` and compare against the image below.
#
# Note the low field strength: the images are noisy, and coil compression plus prewhitening
# matter more here than on the 1.5 T data.
#
# Run: julia --project=examples examples/ocmr/heart_cine_fully_sampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(OCMR_SOURCE, "fs_0152_0_55T")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/frames", x)
save_image("ocmr_cine_fully_sampled_direct", x)

# One cardiac frame is enough for the parallel-imaging part of the example.
acq1 = AcquisitionInfo(first_slab(raw))
acq1, _ = compress_coils(acq1, 6; method = SVDCompression())  # returns (acquisition, mixing matrix)
acq1 = estimate_sensitivities(acq1)

xs = reconstruct(acq1)
report("SENSE, one frame", xs)
save_image("ocmr_cine_fully_sampled_sense", xs)

xcs = reconstruct(acq1, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 30))
report("CS, wavelet sparsity", xcs)
save_image("ocmr_cine_fully_sampled_cs", xcs)
