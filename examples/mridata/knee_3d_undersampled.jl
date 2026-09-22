#!/usr/bin/env julia
#
# mridata.org — 3D knee, undersampled, no recorded echo position
# (`24777e9a-ba0e-49fb-a674-2df22179b126`)
#
# A 3D Cartesian knee acquisition: 7779 profiles, 320-sample readout, 8 channels, k-space
# `(:kx, :ky, :kz, :coil)` at 320 x 288 x 192.
#
# What makes it worth its own example is the header: `center_sample = 0` in every profile.
# Several exporters, this GE one included, spell "echo position not recorded" that way rather
# than leaving the field out, and taking it literally puts the readout outside the encoded
# matrix. MRT detects that case, assumes a symmetric readout, and says so in a warning —
# which is the right assumption for a full-echo readout and the wrong one for a
# partial-Fourier acquisition, hence the warning rather than silence. Expect it on load.
#
# The reconstruction is genuinely 3D: one FFT over three axes, sensitivity maps with a kz
# extent, and `L1Wavelet3D` rather than `L1Wavelet2D` if a prior is wanted. Coil compression
# first keeps the memory bounded.
#
# Run: julia --project=examples examples/mridata/knee_3d_undersampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(MRIDATA, "24777e9a-ba0e-49fb-a674-2df22179b126")
describe(raw)
println("  recorded echo positions: $(unique(Int(p.head.center_sample) for p in raw.profiles))")

acq = AcquisitionInfo(raw)  # warns about the assumed symmetric readout
describe(acq)

x = reconstruct(acq)
report("direct 3D, all coils", x)
save_image("mridata_knee3d_direct", x; z = 96)  # one axial partition

acq, _ = compress_coils(acq, 4; method = SVDCompression())  # returns (acquisition, mixing matrix)
acq = estimate_sensitivities(acq)
println("  maps: $(size(acq.sensitivity_maps))")

xs = reconstruct(acq)
report("SENSE 3D", xs)
save_image("mridata_knee3d_sense", xs; z = 96)
