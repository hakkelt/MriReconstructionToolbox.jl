#!/usr/bin/env julia
#
# fastMRI — prostate diffusion EPI (`fastMRI_prostate_DIFF_IDS_056_066/file_prostate_AXDIFF_056`)
#
# The only echo-planar acquisition in the catalog, and the one whose raw data looks least like
# the rest: a 200 x 300 declared encoding read out in a single shot per excitation, 75
# profiles per slice, 14 channels. Loaded one slice at a time from a ~5.4 GB file, the k-space
# is `(:kx, :ky, :coil)`.
#
# EPI raw data needs corrections that a spin-warp acquisition does not, and none of them are
# in this example: alternating readout polarity (every second line is reversed and has to be
# flipped before the FFT), Nyquist ghost correction from the phase-correction profiles, and
# geometric distortion along the phase-encoding direction. What the example does show is that
# `sample_time_us` is the field all of that depends on — the dwell time converts sample index
# to physical time, and the ghost/distortion corrections are formulated in physical time.
#
# So: treat the image below as "the k-space, Fourier transformed", not as a diffusion image.
# It is the starting point for an EPI pipeline, not the end of one.
#
# fastMRI needs registration: the signed download URLs from the access e-mail are registered
# once with `MRITestData.set_fastmri_urls!`, and they expire after 90 days.
#
# Run: julia --project=examples examples/fastmri/prostate_diffusion.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(FASTMRI, "fastMRI_prostate_DIFF_IDS_056_066/file_prostate_AXDIFF_056"; slice = 1)
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils", x)
save_image("fastmri_prostate_diff_direct", x)

acq = estimate_sensitivities(acq)

xs = reconstruct(acq)
report("SENSE", xs)
save_image("fastmri_prostate_diff_sense", xs)
