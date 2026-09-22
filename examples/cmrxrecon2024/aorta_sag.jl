#!/usr/bin/env julia
#
# CMRxRecon2024 — sagittal aorta cine (`Aorta/TrainingSet/P087/aorta_sag`)
#
# Fully sampled sagittal aortic cine, 10 virtual channels, 2 slices, 12 cardiac phases:
# k-space `(:kx, :ky, :coil, :z, :contrast)`. Two batch dimensions on top of the coil axis,
# so `reconstruct` splits the direct reconstruction into 10 x 2 x 12 tasks and
# `estimate_sensitivities` produces one map set per (slice, phase).
#
# The aorta subset is the one where through-plane flow dominates: the vessel lumen changes
# brightness from phase to phase, which is what breaks naive temporal regularization and is
# the reason the challenge scores this subset separately.
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon2024/aorta_sag.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON2024, "Aorta/TrainingSet/P087/aorta_sag")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices/phases", x)
save_image("cmrx2024_aorta_sag_direct", x)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, one slice/phase", xs)
save_image("cmrx2024_aorta_sag_sense", xs)
