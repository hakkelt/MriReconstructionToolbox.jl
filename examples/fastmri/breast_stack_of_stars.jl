#!/usr/bin/env julia
#
# fastMRI — breast stack-of-stars radial VIBE
# (`fastMRI_breast_IDS_001_010/fastMRI_breast_006_2`)
#
# Golden-angle stack-of-stars: radial spokes in-plane, Cartesian along kz, 640 samples per
# spoke, 288 spokes, 16 channels, 83 partitions, ~4.5 GB. The second non-Cartesian dataset in
# the catalog after USC Speech, and the only 3D one.
#
# Loading one partition (`slice = 1`) turns the problem into a 2D radial reconstruction:
# `(:sample, :readout, :coil)` k-space with a `(:coord, :sample, :readout)` trajectory, which
# the NUFFT gridding handles directly. The proper stack-of-stars treatment is to FFT along kz
# first and then reconstruct each partition — same thing, one partition at a time, which is
# what this example shows one slab of.
#
# Golden-angle ordering means any contiguous set of spokes covers k-space roughly uniformly,
# so the number of spokes per frame is a free parameter: fewer spokes give more temporal
# resolution and more streaking. `stack_of_stars_trajectory` builds the same geometry for
# simulation.
#
# Sensitivity estimation is Cartesian-only — a radial acquisition has no calibration window —
# so grid first and estimate maps from the gridded image if parallel imaging is wanted.
#
# fastMRI needs registration: the signed download URLs from the access e-mail are registered
# once with `MRITestData.set_fastmri_urls!`, and they expire after 90 days.
#
# Run: julia --project=examples examples/fastmri/breast_stack_of_stars.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(FASTMRI, "fastMRI_breast_IDS_001_010/fastMRI_breast_006_2"; slice = 1)
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("gridded, all coils", x)
save_image("fastmri_breast_sos_direct", x)

# Total variation suits the streaking a radial undersampling produces better than wavelets.
xcs = reconstruct(acq, IterativeReconstruction(TotalVariation2D(1.0f-2); maxit = 20))
report("CS, total variation", xcs)
save_image("fastmri_breast_sos_cs", xcs)
