#!/usr/bin/env julia
#
# CMRxRecon2024 — 2D phase-contrast flow (`Flow2d/TestSet/P039/flow2d`)
#
# Two-dimensional phase-contrast flow imaging: 10 virtual channels, k-space
# `(:kx, :ky, :coil, :z, :contrast)` with 2 along `:z` and 12 along `:contrast`. The
# converter maps the `.mat` axes onto ISMRMRD counters positionally, so which counter carries
# the velocity encoding and which the cardiac phase is a property of the file, not of the
# reconstruction — check it against the phase images before reading velocities out.
#
# It matters because velocity comes from the *phase difference* between the flow-compensated
# and the flow-encoded acquisition. Both members of that pair have to be reconstructed with
# the same sensitivity maps and the same regularization, otherwise the difference picks up a
# reconstruction bias on top of the flow. Estimating maps once and reusing them, as below, is
# the safe habit for any phase-sensitive acquisition.
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon2024/flow2d.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON2024, "Flow2d/TestSet/P039/flow2d")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/encodings", x)
save_image("cmrx2024_flow2d_direct", x)

# Maps from the first flow encoding, reused for both.
acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))
maps = acq1.sensitivity_maps
println("  maps: $(size(maps)), reused for both flow encodings")

xs = reconstruct(acq1)
report("SENSE, first encoding", xs)
save_image("cmrx2024_flow2d_sense", xs)
