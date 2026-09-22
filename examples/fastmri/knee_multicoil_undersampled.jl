#!/usr/bin/env julia
#
# fastMRI — knee, multi-coil, undersampled test split (`multicoil_test/file1000747`)
#
# Proton-density-weighted knee FSE, 15 channels, 640-sample readout, 85 of 368
# phase-encoding lines acquired. This file is loaded one slice at a time
# (`load_example(...; slice = 1)`): at ~820 MB the whole volume does not need to be in memory
# to show what the pipeline does with it, and `load_raw`'s `slice`/`contrast`/`repetition`
# keywords are the intended way to stay bounded on the large fastMRI files.
#
# Because only one slice is loaded, the k-space has no batch dimension at all:
# `(:kx, :ky, :coil)`. Drop the keyword to get `(:kx, :ky, :coil, :z)` and the whole stack.
#
# fastMRI needs registration: the signed download URLs from the access e-mail are registered
# once with `MRITestData.set_fastmri_urls!`, and they expire after 90 days.
#
# Run: julia --project=examples examples/fastmri/knee_multicoil_undersampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(FASTMRI, "multicoil_test/file1000747"; slice = 1)
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils", x)
save_image("fastmri_knee_multicoil_test_direct", x)

acq, _ = compress_coils(acq, 8; method = SVDCompression())  # returns (acquisition, mixing matrix)
acq = estimate_sensitivities(acq)

xs = reconstruct(acq)
report("SENSE", xs)
save_image("fastmri_knee_multicoil_test_sense", xs)

xcs = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 50))
report("CS, wavelet sparsity", xcs)
save_image("fastmri_knee_multicoil_test_cs", xcs)
