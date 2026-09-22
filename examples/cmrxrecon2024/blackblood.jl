#!/usr/bin/env julia
#
# CMRxRecon2024 — black-blood TSE (`BlackBlood/TestSet/P001/blackblood`)
#
# Turbo spin echo with blood suppression: a static, non-gated acquisition, so unlike the rest
# of the CMRxRecon subsets it has no frame dimension at all. 10 virtual channels, 5 slices,
# k-space `(:kx, :ky, :coil, :z)` — the plain multi-slice 2D case, and at ~30 MB the smallest
# CMRxRecon2024 file in the catalog.
#
# Start here when trying the challenge data for the first time: one batch dimension, no
# temporal structure to get wrong, and the suppressed blood pool makes it obvious when
# regularization has smoothed the myocardial border.
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon2024/blackblood.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON2024, "BlackBlood/TestSet/P001/blackblood")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices", x)
save_image("cmrx2024_blackblood_direct", x)

# Maps for all five slices at once; SENSE then runs on the whole stack.
acq = estimate_sensitivities(acq)
println("  maps: $(size(acq.sensitivity_maps))")

xs = reconstruct(acq)
report("SENSE, all slices", xs)
save_image("cmrx2024_blackblood_sense", xs; z = 3)
