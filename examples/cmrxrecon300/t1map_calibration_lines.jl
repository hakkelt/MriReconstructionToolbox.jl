#!/usr/bin/env julia
#
# CMRxRecon-300 — MOLLI T1 mapping, and the calibration lines that have to be separated first
# (`TestSet/P111/t1map`)
#
# This one is in the examples because of how it fails.
#
# CMRxRecon-300 ships its auto-calibration signal as *profiles in the same file*, flagged
# `ACQ_IS_PARALLEL_CALIBRATION`, on the same phase-encoding grid as the imaging lines — 540 of
# the 1185 profiles here. The ACS readout is a low-resolution 128-sample block written into the
# declared 512-sample readout without being centred (its samples occupy bins 55–182), while its
# header still says `center_sample = 256`.
#
# Assemble both kinds into one k-space and the ACS lines land on top of the central imaging
# lines they share, whichever comes last winning. Where the ACS wins, the centre of the encoded
# matrix ends up empty, so `estimate_sensitivities` reads a calibration window that is all
# zeros and returns all-zero maps (with a warning); SENSE on those maps is an empty image and
# an iterative solve on them diverges to NaN. Whether it happens at all depends on the order
# the profiles sit in the file — the sibling `TestSet/P111/t2map` has the same ACS lines and
# survives, because there the imaging profiles are written last.
#
# The fix is to treat the two for what they are: separate acquisitions. Split on the flag,
# reconstruct the imaging lines, and use the ACS block for calibration (recentred, since its
# header is not to be trusted on that point).
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon300/t1map_calibration_lines.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON300, "TestSet/P111/t1map")
describe(raw)

is_calibration(p) = MRIBase.flag_is_set(p, "ACQ_IS_PARALLEL_CALIBRATION")
ncal = count(is_calibration, raw.profiles)
println("  $(ncal) of $(length(raw.profiles)) profiles are flagged as calibration")

# Where the calibration readout actually holds samples, against what its header claims.
calibration = [p for p in raw.profiles if is_calibration(p)]
support = findall(!iszero, sum(abs2, reduce(hcat, (vec(sum(abs2, p.data; dims = 2)) for p in calibration)); dims = 2)[:])
println(
    "  calibration samples occupy readout bins $(first(support)):$(last(support)) of " *
        "$(size(calibration[1].data, 1)), header center_sample = $(Int(calibration[1].head.center_sample))"
)

# Everything together: the maps come back empty, and MRT says so.
mixed = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))
println("  maps from imaging + calibration profiles: all zero = $(iszero(mixed.sensitivity_maps))")
report("SENSE on those maps", reconstruct(mixed))

# Split on the flag and the imaging lines behave like any other undersampled acquisition.
imaging = typeof(raw)(raw.params, [p for p in raw.profiles if !is_calibration(p)])

acq = AcquisitionInfo(imaging)
describe(acq)

x = reconstruct(acq)
report("direct, imaging lines only", x)
save_image("cmrx300_t1map_direct", x)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(imaging)))
println("  maps from imaging profiles only: all zero = $(iszero(acq1.sensitivity_maps))")

xs = reconstruct(acq1)
report("SENSE, imaging lines only", xs)
save_image("cmrx300_t1map_sense", xs)
