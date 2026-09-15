#!/usr/bin/env julia
#
# CMRxRecon2024 — T2-prepared FLASH T2 mapping (`Mapping/TrainingSet/P196/T2map`)
#
# Three T2-preparation times, 5 slices, 10 virtual channels: k-space
# `(:kx, :ky, :coil, :z, :contrast)`, and the smallest file in the mapping subset (~38 MB),
# which makes it the quickest CMRxRecon example to run.
#
# With only three points along the decay curve, every one of them counts: a reconstruction
# artefact in the shortest-T2prep image propagates straight into the fitted T2 map. That is
# the argument for reconstructing all preparations with identical settings — same maps, same
# regularizer, same number of iterations — rather than tuning each image to look best on its
# own.
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon2024/t2map.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON2024, "Mapping/TrainingSet/P196/T2map")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices/preps", x)
for prep in 1:size(x, ndims(x))
    save_image("cmrx2024_t2map_direct_prep$(prep)", x; contrast = prep)
end

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, first prep", xs)
save_image("cmrx2024_t2map_sense", xs)

xcs = reconstruct(acq1, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 30))
report("CS, wavelet sparsity", xcs)
save_image("cmrx2024_t2map_cs", xcs)
