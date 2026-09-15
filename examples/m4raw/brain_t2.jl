#!/usr/bin/env julia
#
# M4Raw — 0.3 T brain T2-weighted TSE, 4 channels (`multicoil_train/2022062402_T203`)
#
# T2-weighted turbo spin echo: 4608 profiles, 256 samples, 4 channels, 18 slices, k-space
# `(:kx, :ky, :coil, :z)`.
#
# With only 4 receive channels there is little to gain from coil compression, but the
# acquisition is fully sampled, which makes it a good place to see what retrospective
# undersampling costs: build a sampling pattern with `create_sampling_pattern`, apply it to
# `acq.kspace_data`, and compare an accelerated reconstruction against the image below.
#
# Run: julia --project=examples examples/m4raw/brain_t2.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(M4RAW, "multicoil_train/2022062402_T203")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices", x)
save_image("m4raw_t2_direct", x; z = 9)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, one slice", xs)
save_image("m4raw_t2_sense", xs)

xcs = reconstruct(acq1, IterativeReconstruction(TotalVariation2D(1.0f-3); maxit = 30))
report("CS, total variation", xcs)
save_image("m4raw_t2_cs", xcs)
