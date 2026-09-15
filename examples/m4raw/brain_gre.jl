#!/usr/bin/env julia
#
# M4Raw — 0.3 T brain GRE, 4 channels (`gre/2022062704_GRE02`)
#
# The spoiled gradient-echo contrast of the M4Raw catalog, and the only one that lives in a
# separate `gre/` split: 4608 profiles, 256 samples, 4 channels, 18 slices, k-space
# `(:kx, :ky, :coil, :z)`.
#
# GRE at 0.3 T is the noisiest contrast in the set, so it is the useful one for judging
# denoising and regularization strength: raise the wavelet weight below and watch the
# background noise go before the anatomy does.
#
# Run: julia --project=examples examples/m4raw/brain_gre.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(M4RAW, "gre/2022062704_GRE02")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices", x)
save_image("m4raw_gre_direct", x; z = 9)

# One slice for the iterative part, so that the example stays quick.
acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, one slice", xs)
save_image("m4raw_gre_sense", xs)

xcs = reconstruct(acq1, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 30))
report("CS, wavelet sparsity", xcs)
save_image("m4raw_gre_cs", xcs)
