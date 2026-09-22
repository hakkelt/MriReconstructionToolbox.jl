#!/usr/bin/env julia
#
# fastMRI — knee, single-coil, fully sampled train split (`singlecoil_train/file1002078`)
#
# The fully sampled half of the emulated single-coil track: `(:kx, :ky, :coil, :z)` with
# `:coil` of length 1, 640 x 290, 28 slices.
#
# This is the simplest real dataset in the whole catalog — one channel, Cartesian, fully
# sampled, no batch dimension beyond the slice stack — which makes it the right first file to
# try a new reconstruction method on. If it does not work here, the problem is the method,
# not the data.
#
# fastMRI needs registration: the signed download URLs from the access e-mail are registered
# once with `MRITestData.set_fastmri_urls!`, and they expire after 90 days.
#
# Run: julia --project=examples examples/fastmri/knee_singlecoil_fully_sampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(FASTMRI, "singlecoil_train/file1002078")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all slices", x)
save_image("fastmri_knee_singlecoil_train_direct", x; z = 14)

acq1 = AcquisitionInfo(first_slab(raw))

xd = reconstruct(acq1)
report("direct, one slice", xd)
save_image("fastmri_knee_singlecoil_train_slice", xd)

xcs = reconstruct(estimate_sensitivities(acq1), IterativeReconstruction(L1Wavelet2D(1.0f-4); maxit = 30))
report("CS on fully sampled data", xcs)
save_image("fastmri_knee_singlecoil_train_cs", xcs)
