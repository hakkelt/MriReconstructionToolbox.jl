#!/usr/bin/env julia
#
# M4Raw — 0.3 T brain T1-weighted TSE, 4 channels (`multicoil_train/2022062402_T103`)
#
# T1-weighted turbo spin echo, third repeat of the session: 4608 profiles, 256 samples, 4
# channels, 18 slices, k-space `(:kx, :ky, :coil, :z)`.
#
# The same subject and session also holds `T101` and `T102`. Loading two repeats and
# averaging the magnitude images is the low-field reference recipe the dataset was built for;
# reconstructing one repeat, as here, is what a method has to match.
#
# Run: julia --project=examples examples/m4raw/brain_t1.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(M4RAW, "multicoil_train/2022062402_T103")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices", x)
save_image("m4raw_t1_direct", x; z = 9)

acq1 = AcquisitionInfo(first_slab(raw))

# Root-sum-of-squares needs no maps and is the reference this stack is usually compared to.
xrss = reconstruct(acq1, DirectReconstruction(; coil_combination = RootSumSquares()))
report("RSS, one slice", xrss)
save_image("m4raw_t1_rss", xrss)

xs = reconstruct(estimate_sensitivities(acq1))
report("SENSE, one slice", xs)
save_image("m4raw_t1_sense", xs)
