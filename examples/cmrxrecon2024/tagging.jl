#!/usr/bin/env julia
#
# CMRxRecon2024 — SPAMM-tagged cine (`Tagging/TestSet/P032/tagging`)
#
# Tagged cine for myocardial strain: a SPAMM preparation stripes the magnetization before
# each cine, so the images carry a high-frequency grid that deforms with the tissue. 10
# virtual channels, 3 slices, 12 cardiac phases: `(:kx, :ky, :coil, :z, :contrast)`.
#
# The tag grid is exactly what regularization tends to destroy. Total variation and wavelet
# sparsity both treat the stripes as texture to be smoothed, and the strain analysis that
# follows depends on tracking them, so this is the subset to test a regularizer against
# before trusting it: reconstruct, then look at whether the stripes survive with their
# contrast intact rather than at how clean the background is.
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon2024/tagging.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON2024, "Tagging/TestSet/P032/tagging")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices/phases", x)
save_image("cmrx2024_tagging_direct", x)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, one slice/phase", xs)
save_image("cmrx2024_tagging_sense", xs)

# Compare the two images: the wavelet penalty is what a strain analysis has to survive.
xcs = reconstruct(acq1, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 30))
report("CS, wavelet sparsity", xcs)
save_image("cmrx2024_tagging_cs", xcs)
