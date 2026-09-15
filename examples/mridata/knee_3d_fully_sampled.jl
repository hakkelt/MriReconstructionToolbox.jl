#!/usr/bin/env julia
#
# mridata.org — 3D knee, fully sampled, one partition at a time
# (`52c2fd53-d233-4444-8bfd-7c454240d314`)
#
# A 1.5 GB fully sampled 3D knee volume. Loaded with `slice = 1` it yields a single
# partition: 320 profiles, 320 samples, 8 channels, and an encoded matrix the header reports
# as 320 x 320 x 1.
#
# That single-partition slab is the instructive part. The header still says the acquisition
# is 3D, so MRT builds a `(:kx, :ky, :kz, :coil)` k-space with a `:kz` extent of one, and
# everything that needs a *volume* then has nowhere to work: the direct reconstruction is
# fine, but the calibration window sensitivity estimation cuts out of the centre of the
# encoded matrix has a kz extent it cannot satisfy, so the maps come back all zero (with a
# warning), SENSE gives an empty image and an iterative solve on those maps diverges to NaN.
#
# There are two ways out, and this example shows both:
#
#  * Treat the slab as what it is, a 2D acquisition: drop the degenerate `:kz` axis and build
#    a 2D `CartesianAcquisitionInfo` around it. Maps and SENSE then behave normally.
#  * Load the whole volume (no `slice` keyword, ~4 GB of memory) and reconstruct in 3D, where
#    the calibration window fits.
#
# The header also leaves `center_sample` at zero, so expect the symmetric-readout warning
# described in `knee_3d_undersampled.jl`.
#
# Run: julia --project=examples examples/mridata/knee_3d_fully_sampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(MRIDATA, "52c2fd53-d233-4444-8bfd-7c454240d314"; slice = 1)
describe(raw)

acq3d = AcquisitionInfo(raw)
describe(acq3d)

x = reconstruct(acq3d)
report("direct, one partition", x)
save_image("mridata_knee3d_fs_direct", x)

# A 3D k-space with a single partition: the maps come back zero and MRT warns.
maps3d = estimate_sensitivities(acq3d).sensitivity_maps
println("  maps from the degenerate 3D slab: all zero = $(iszero(maps3d))")

# The fix — the slab is 2D, so say so.
kspace = NamedDimsArray{(:kx, :ky, :coil)}(dropdims(unname(acq3d.kspace_data); dims = 3))
acq2d = CartesianAcquisitionInfo(kspace; image_size = acq3d.image_size[1:2])
describe(acq2d)

acq2d = estimate_sensitivities(acq2d)
println("  maps from the 2D acquisition: all zero = $(iszero(acq2d.sensitivity_maps))")

xs = reconstruct(acq2d)
report("SENSE, as 2D", xs)
save_image("mridata_knee3d_fs_sense", xs)
