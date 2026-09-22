#!/usr/bin/env julia
#
# CMRxRecon-300 — long-axis bSSFP cine, training set (`TrainingSet/P003/cine_lax`)
#
# The 300-volunteer release differs from CMRxRecon2024 in two ways that matter here: the
# k-space is *prospectively* undersampled (`fully_sampled = false`) and the coils are the
# scanner's own 30 channels rather than 10 compressed virtual coils.
#
# Loaded k-space is `(:kx, :ky, :coil, :z, :contrast)` with 20 slices and 4 phases, and only
# 48 of the 132 phase-encoding lines are sampled — the `subsampling` of the acquisition
# records which ones, and the direct reconstruction therefore shows the aliasing the sampling
# pattern produces.
#
# CMRxRecon-300 also carries its auto-calibration lines as flagged profiles in the same file.
# This file's slabs happen to be free of them, so the k-space assembles cleanly; where they are
# present they have to be separated first, for the reasons `t1map_calibration_lines.jl` spells
# out.
#
# With 30 channels, coil compression pays for itself before any iterative solve.
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon300/cine_lax.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON300, "TrainingSet/P003/cine_lax")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)
println("  image size $(acq.image_size) from $(size(acq.kspace_data, :ky)) sampled ky lines")

x = reconstruct(acq)
report("direct, all coils/slices/phases", x)
save_image("cmrx300_cine_lax_direct", x)

acq1 = AcquisitionInfo(first_slab(raw))
acq1, _ = compress_coils(acq1, 8; method = SVDCompression())  # returns (acquisition, mixing matrix)
acq1 = estimate_sensitivities(acq1)

xs = reconstruct(acq1)
report("SENSE, one slice/phase", xs)
save_image("cmrx300_cine_lax_sense", xs)

xcs = reconstruct(acq1, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 30))
report("CS, wavelet sparsity", xcs)
save_image("cmrx300_cine_lax_cs", xcs)
