#!/usr/bin/env julia
#
# mridata.org — multi-slice, multi-echo 2D FSE of the ankle
# (`0e0b0437-866c-4f0c-ac52-1e7592c159c6`)
#
# Catalogued as a 3D acquisition (`acquisition_dim = 3`) because that is what the header
# declares, this is really a 15-slice, 2-echo 2D fast spin echo (Stanford 2D FSE project, foot
# and ankle coil, 288 x 202 matrix, 8 channels). The k-space comes out as
# `(:kx, :ky, :coil, :z, :contrast)` — two batch dimensions, and the two `:contrast` entries
# are the two echo times, not two frames of anything.
#
# The lesson generalises past this file: `acquisition_dim` in the catalog is metadata, while
# the dimension names on `acq.kspace_data` are what the reconstruction actually sees. Read the
# latter before deciding whether a dataset is 3D.
#
# Run: julia --project=examples examples/mridata/other_3d_fully_sampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(MRIDATA, "0e0b0437-866c-4f0c-ac52-1e7592c159c6")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices/echoes", x)
save_image("mridata_ankle_fse_echo1", x; z = 8, contrast = 1)
save_image("mridata_ankle_fse_echo2", x; z = 8, contrast = 2)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, one slice/echo", xs)
save_image("mridata_ankle_fse_sense", xs)
