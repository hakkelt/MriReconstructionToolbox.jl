#!/usr/bin/env julia
#
# mridata.org — 3D brain with a separate calibration block
# (`25952770-5d0e-4cc1-917d-a77538f44a08`)
#
# `AcquisitionInfo(raw)` refuses this file outright:
#
#     ArgumentError: profiles have differing readout lengths; assemble manually
#
# and it is right to. The 7744 profiles are two different acquisitions in one file: 6720
# imaging profiles with a 384-sample readout over a 192 x 192 phase-encoding grid, and 1024
# profiles with a 128-sample readout over a 32 x 32 grid — a separate low-resolution
# calibration scan, flagged `ACQ_IS_PARALLEL_CALIBRATION`. Nothing can be assembled into one
# rectangular array from that, and guessing which profiles to drop is not the reader's call
# to make.
#
# Splitting the two by their flag is: the imaging profiles alone assemble into a
# `(:kx, :ky, :kz, :coil)` k-space of 384 x 192 x 192 x 18, and the calibration block is a
# second, small acquisition in its own right — which is exactly what it is for, since maps
# estimated from a low-resolution fully sampled scan are better than maps estimated from the
# centre of an undersampled one.
#
# Coil compression before the 3D solve is not optional at this size: 384 x 192 x 192 x 18
# ComplexF32 is about 1 GB of k-space.
#
# Run: julia --project=examples examples/mridata/brain_3d_with_calibration_block.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(MRIDATA, "25952770-5d0e-4cc1-917d-a77538f44a08")
describe(raw)
println("  readout lengths present: $(sort(unique(size(p.data, 1) for p in raw.profiles)))")

is_calibration(p) = MRIBase.flag_is_set(p, "ACQ_IS_PARALLEL_CALIBRATION")
imaging = typeof(raw)(raw.params, [p for p in raw.profiles if !is_calibration(p)])
calibration = typeof(raw)(raw.params, [p for p in raw.profiles if is_calibration(p)])
println("  $(length(imaging.profiles)) imaging profiles, $(length(calibration.profiles)) calibration profiles")

acq = AcquisitionInfo(imaging)
describe(acq)

# 18 channels down to 4 before anything touches the whole volume.
acq, _ = compress_coils(acq, 4; method = SVDCompression())  # returns (acquisition, mixing matrix)

x = reconstruct(acq)
report("direct 3D", x)
save_image("mridata_brain3d_direct", x; z = size(x, :z) ÷ 2)

acq = estimate_sensitivities(acq)
println("  maps: $(size(acq.sensitivity_maps))")

xs = reconstruct(acq)
report("SENSE 3D", xs)
save_image("mridata_brain3d_sense", xs; z = size(xs, :z) ÷ 2)
