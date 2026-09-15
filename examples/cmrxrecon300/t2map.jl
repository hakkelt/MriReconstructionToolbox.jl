#!/usr/bin/env julia
#
# CMRxRecon-300 — T2 mapping, test set (`TestSet/P111/t2map`)
#
# Undersampled T2-prepared FLASH from the 300-volunteer test set: 12 channels, 5 slices, 3
# preparation times, k-space `(:kx, :ky, :coil, :z, :contrast)` with 61 of 116 phase-encoding
# lines sampled. At ~20 MB it is the smallest CMRxRecon-300 file, so it is the cheapest way to
# check that the Synapse credentials and the per-set index are in place.
#
# Like every CMRxRecon-300 file it carries its auto-calibration lines as flagged profiles
# mixed in with the imaging lines (540 of the 1185 profiles here). This one survives being
# assembled without separating them, because its imaging profiles are written last and
# therefore win the central phase-encoding lines they share. Its sibling
# `TestSet/P111/t1map` does not — see `t1map_calibration_lines.jl` for what that looks like
# and why the flag, not the profile order, is what to rely on.
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon300/t2map.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON300, "TestSet/P111/t2map")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices/preps", x)
save_image("cmrx300_t2map_direct", x)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, one slice/prep", xs)
save_image("cmrx300_t2map_sense", xs)

xcs = reconstruct(acq1, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 30))
report("CS, wavelet sparsity", xcs)
save_image("cmrx300_t2map_cs", xcs)
