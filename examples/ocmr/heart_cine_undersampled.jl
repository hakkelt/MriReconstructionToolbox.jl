#!/usr/bin/env julia
#
# OCMR — prospectively undersampled 1.5 T cardiac cine (`us_0167_pt_1_5T`)
#
# Real-time free-breathing cine: 572 profiles, 352 readout samples, 24 channels, 30 cardiac
# frames. The frame counter lands in `phase`, which the extension maps to a `:time` batch
# dimension, so the k-space comes out as `(:kx, :ky, :coil, :time)` and `reconstruct`
# splits the direct reconstruction into one task per (coil, frame).
#
# This file still carries its noise-adjustment profiles (32 of them), which is what makes it
# the example for `estimate_noise_covariance` + `prewhiten`. Because the acquisition is
# genuinely undersampled, the temporal-Fourier sparsity of `L1TemporalFourier` is the
# regularizer that suits it: cardiac motion is periodic, so the x-f representation is sparse.
#
# One limitation to know about: `estimate_sensitivities` does return one map set per slab of a
# batched acquisition, but `AcquisitionInfo` only accepts a per-slab map array when the batch
# dimension is `:z` (the multi-slice layout `(:x, :y, :coil, :z)`). A cine therefore gets a
# single map set for the whole series, which is the right answer here anyway — see below.
#
# Run: julia --project=examples examples/ocmr/heart_cine_undersampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase  # loads the extension that turns a RawAcquisitionData into an AcquisitionInfo

setup()

raw = load_example(OCMR_SOURCE, "us_0167_pt_1_5T")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

# Direct reconstruction of every coil and every frame: no sensitivity maps, no iterations.
x = reconstruct(acq)
report("direct, all coils/frames", x)
save_image("ocmr_cine_undersampled_direct", x)

# Prewhitening needs the noise profiles the exporter kept alongside the image profiles.
Ψ = noise_covariance(raw)
acq = Ψ === nothing ? acq : prewhiten(acq, Ψ)
Ψ === nothing || println("  prewhitened with a $(size(Ψ)) noise covariance")

# 24 channels are more than this matrix needs; compress before the iterative solve.
acq, _ = compress_coils(acq, 8; method = SVDCompression())  # returns (acquisition, mixing matrix)

# Maps come from one frame and are used for the whole series. That is what a cine wants — the
# coils do not move, and one map set keeps every frame on the same intensity scale, which
# matters as soon as the frames are compared to each other. Note that the frame is cut out of
# the *compressed* acquisition: virtual coils from a separate `compress_coils` call would be a
# different basis, and maps from one basis do not describe k-space in another.
frame = AcquisitionInfo(acq; kspace_data = NamedDimsArray{(:kx, :ky, :coil)}(unname(acq.kspace_data)[:, :, :, 1]))
maps = estimate_sensitivities(frame).sensitivity_maps
println("  maps: $(dimnames(maps)) $(size(maps)), shared by all $(size(acq.kspace_data, :time)) frames")

acq = AcquisitionInfo(acq; sensitivity_maps = maps)
xs = reconstruct(acq)
report("SENSE (all frames)", xs)
save_image("ocmr_cine_undersampled_sense", xs)

xcs = reconstruct(acq, IterativeReconstruction(L1TemporalFourier(1.0f-3); maxit = 30))
report("CS, x-f sparsity", xcs)
save_image("ocmr_cine_undersampled_cs", xcs)
