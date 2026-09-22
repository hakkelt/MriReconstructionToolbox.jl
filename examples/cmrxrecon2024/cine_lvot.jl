#!/usr/bin/env julia
#
# CMRxRecon2024 — bSSFP cine, LVOT view (`Cine/TrainingSet/P045/cine_lvot`)
#
# Balanced steady-state free precession cine from the 2024 challenge training set: fully
# sampled, 10 virtual channels (the challenge ships coil-compressed data, hence
# `coil_data = :derived`), one slice, 12 cardiac phases.
#
# The `.mat` source is converted to ISMRMRD on first load, and the converter puts the cardiac
# phases in the `contrast` counter, so the k-space is `(:kx, :ky, :coil, :contrast)` and every
# phase is reconstructed independently. That is the right default — nothing in the file says
# the frames belong to one time series — but it also means a temporal regularizer has nothing
# to act on. To exploit temporal redundancy, relabel the dimension:
#
#     ksp = NamedDimsArray{(:kx, :ky, :coil, :time)}(unname(acq.kspace_data))
#
# and rebuild the `CartesianAcquisitionInfo` around it; `L1TemporalFourier` and
# `TemporalTotalVariation` then see a 12-frame series instead of 12 unrelated images.
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon2024/cine_lvot.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON2024, "Cine/TrainingSet/P045/cine_lvot")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/phases", x)
save_image("cmrx2024_cine_lvot_direct", x)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, one phase", xs)
save_image("cmrx2024_cine_lvot_sense", xs)

xcs = reconstruct(acq1, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 30))
report("CS, wavelet sparsity", xcs)
save_image("cmrx2024_cine_lvot_cs", xcs)
