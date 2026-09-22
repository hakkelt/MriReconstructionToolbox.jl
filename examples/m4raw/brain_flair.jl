#!/usr/bin/env julia
#
# M4Raw — 0.3 T brain FLAIR, 4 channels (`multicoil_train/2022062402_FLAIR01`)
#
# Low-field (0.3 T) fully sampled Cartesian brain k-space: 4608 profiles, 256 samples, 4
# channels, 18 slices. The slice counter becomes a `:z` batch dimension, so the k-space is
# `(:kx, :ky, :coil, :z)` and both the direct reconstruction and the sensitivity estimation
# run slice by slice — the maps come back as `(256, 256, 4, 18)`, one set per slice, which is
# what a multi-slice 2D acquisition requires (the coils see a different object in each slab).
#
# The exporter writes `sample_time_us = 0`. On a Cartesian acquisition nothing in the
# reconstruction needs it, so this is harmless here, but see the USC Speech example for a
# case where it matters.
#
# M4Raw ships several repeats of every contrast (`FLAIR01`, `FLAIR02`, ...) acquired
# back-to-back, which is the point of the dataset: averaging repeats gives a low-noise
# reference for training and evaluating denoisers at 0.3 T.
#
# Run: julia --project=examples examples/m4raw/brain_flair.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(M4RAW, "multicoil_train/2022062402_FLAIR01")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices", x)
save_image("m4raw_flair_direct", x; z = 9)

# Sensitivity maps for all 18 slices at once, then SENSE on the whole stack.
acq = estimate_sensitivities(acq)
println("  maps: $(size(acq.sensitivity_maps))")

xs = reconstruct(acq)
report("SENSE, all slices", xs)
save_image("m4raw_flair_sense", xs; z = 9)
