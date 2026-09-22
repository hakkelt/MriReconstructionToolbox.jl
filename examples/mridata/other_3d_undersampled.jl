#!/usr/bin/env julia
#
# mridata.org — prospectively accelerated 3D MPRAGE, 58 channels
# (`4a39e0bb-5c36-45a6-9c45-7415e054fb1b`)
#
# 1 mm isotropic MPRAGE, 512 x 240 x 208, acquired with the scanner's own CS acceleration on
# a 64-channel head/neck coil (58 active channels). At 1.9 GB it is the largest mridata.org
# file in the catalog and the one that needs the most care about memory: a dense
# 512 x 240 x 208 x 58 ComplexF32 k-space is about 12 GB, and the images that come out of it
# are the same size again, so coil compression has to come before anything else.
#
# The example therefore compresses to 4 virtual coils immediately after assembling the
# acquisition, and reconstructs 3D directly. Expect it to want tens of gigabytes of memory;
# on a workstation, run the compression and the solve as separate steps and keep only the
# compressed k-space.
#
# Like mridata.org's 3D brain data it also holds a separate calibration scan, with a shorter
# readout, so the profiles have to be split on `ACQ_IS_PARALLEL_CALIBRATION` before anything
# can be assembled — see `brain_3d_with_calibration_block.jl` for that story in full.
#
# This is the file to reach for when testing that a pipeline scales — nothing else in the
# catalog combines a large 3D matrix, heavy undersampling and a high channel count.
#
# Run: julia --project=examples examples/mridata/other_3d_undersampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(MRIDATA, "4a39e0bb-5c36-45a6-9c45-7415e054fb1b")
describe(raw)
println("  readout lengths present: $(sort(unique(size(p.data, 1) for p in raw.profiles)))")

is_calibration(p) = MRIBase.flag_is_set(p, "ACQ_IS_PARALLEL_CALIBRATION")
imaging = typeof(raw)(raw.params, [p for p in raw.profiles if !is_calibration(p)])
println("  $(length(imaging.profiles)) imaging profiles, $(length(raw.profiles) - length(imaging.profiles)) calibration profiles")

acq = AcquisitionInfo(imaging)
describe(acq)

acq, _ = compress_coils(acq, 4; method = SVDCompression())  # returns (acquisition, mixing matrix)
println("  compressed to $(size(acq.kspace_data, :coil)) virtual coils")

x = reconstruct(acq)
report("direct 3D", x)
save_image("mridata_mprage_direct", x; z = size(x, :z) ÷ 2)

acq = estimate_sensitivities(acq)
xs = reconstruct(acq)
report("SENSE 3D", xs)
save_image("mridata_mprage_sense", xs; z = size(xs, :z) ÷ 2)
