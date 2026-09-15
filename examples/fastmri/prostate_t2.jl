#!/usr/bin/env julia
#
# fastMRI — prostate T2 TSE (`fastMRI_prostate_T2_IDS_021_040/file_prostate_AXT2_023`)
#
# Axial T2-weighted turbo spin echo of the prostate, 10 channels, 640-sample readout, 450 of
# 642 phase-encoding lines. The file is ~2.1 GB, so it is loaded one slice at a time and the
# k-space has no batch dimension: `(:kx, :ky, :coil)`.
#
# The prostate release is the one where the field of view is large and the structure of
# interest is small and central. That is worth remembering when judging a reconstruction on
# it: the global error is dominated by fat, bowel and body wall, while what matters is the
# peripheral zone in the middle. Crop before computing any metric.
#
# fastMRI needs registration: the signed download URLs from the access e-mail are registered
# once with `MRITestData.set_fastmri_urls!`, and they expire after 90 days.
#
# Run: julia --project=examples examples/fastmri/prostate_t2.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(FASTMRI, "fastMRI_prostate_T2_IDS_021_040/file_prostate_AXT2_023"; slice = 1)
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils", x)
save_image("fastmri_prostate_t2_direct", x)

acq = estimate_sensitivities(acq)

xs = reconstruct(acq)
report("SENSE", xs)
save_image("fastmri_prostate_t2_sense", xs)

xcs = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 40))
report("CS, wavelet sparsity", xcs)
save_image("fastmri_prostate_t2_cs", xcs)
