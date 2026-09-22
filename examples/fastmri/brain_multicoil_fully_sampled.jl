#!/usr/bin/env julia
#
# fastMRI — brain, multi-coil, fully sampled train split
# (`multicoil_train/file_brain_AXT1POST_201_6002738`)
#
# The train splits hold fully sampled k-space, which is what makes them the reference half of
# fastMRI: `(:kx, :ky, :coil, :z)` with all 290 phase-encoding lines and 12 slices.
#
# Use this file, not a test-split file, when measuring reconstruction error. Undersample it
# yourself with `create_sampling_pattern`, reconstruct, and compare against the fully sampled
# image below; the test splits have no reference to compare to, which is the point of a
# challenge but useless for development.
#
# fastMRI needs registration: the signed download URLs from the access e-mail are registered
# once with `MRITestData.set_fastmri_urls!`, and they expire after 90 days.
#
# Run: julia --project=examples examples/fastmri/brain_multicoil_fully_sampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(FASTMRI, "multicoil_train/file_brain_AXT1POST_201_6002738")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices", x)
save_image("fastmri_brain_train_direct", x; z = 6)

acq1 = AcquisitionInfo(first_slab(raw))

xrss = reconstruct(acq1, DirectReconstruction(; coil_combination = RootSumSquares()))
report("RSS reference, one slice", xrss)
save_image("fastmri_brain_train_rss", xrss)

xs = reconstruct(estimate_sensitivities(acq1))
report("SENSE, one slice", xs)
save_image("fastmri_brain_train_sense", xs)
