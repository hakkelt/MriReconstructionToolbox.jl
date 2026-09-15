#!/usr/bin/env julia
#
# fastMRI — knee, multi-coil, fully sampled train split (`multicoil_train/file1001465`)
#
# The classic fastMRI knee reference: proton-density FSE, 15 channels, 640 x 290 fully
# sampled, loaded here one slice at a time out of a ~700 MB file.
#
# This is the file to calibrate a pipeline against, because both halves of the comparison are
# available: reconstruct it fully sampled for the reference, then subsample it at the
# acceleration you care about and measure. The 15 channels also make it the most instructive
# place to look at coil compression — compare the SENSE image below at 15, 8 and 4 virtual
# coils and see where the gain in speed starts costing SNR.
#
# fastMRI needs registration: the signed download URLs from the access e-mail are registered
# once with `MRITestData.set_fastmri_urls!`, and they expire after 90 days.
#
# Run: julia --project=examples examples/fastmri/knee_multicoil_fully_sampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(FASTMRI, "multicoil_train/file1001465"; slice = 1)
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

xrss = reconstruct(acq, DirectReconstruction(; coil_combination = RootSumSquares()))
report("RSS reference", xrss)
save_image("fastmri_knee_multicoil_train_rss", xrss)

for ncoils in (15, 8)
    a = ncoils == size(acq.kspace_data, :coil) ? acq : first(compress_coils(acq, ncoils; method = SVDCompression()))
    xs = reconstruct(estimate_sensitivities(a))
    report("SENSE, $(ncoils) virtual coils", xs)
    save_image("fastmri_knee_multicoil_train_sense_$(ncoils)coils", xs)
end
