#!/usr/bin/env julia
#
# CMRxRecon2024 — MOLLI T1 mapping (`Mapping/TrainingSet/P157/T1map`)
#
# Modified Look-Locker inversion recovery with a FLASH readout: 10 virtual channels, k-space
# `(:kx, :ky, :coil, :z, :contrast)` with 5 slices and 9 inversion times.
#
# The 9 `:contrast` entries are not a time series but a magnetization-recovery curve: signal
# intensity, including its sign, changes drastically between them, and the earliest inversion
# times are nearly null images. Two consequences for reconstruction:
#
#  * Each inversion time must keep its own intensity scale. Because the batch dimensions are
#    reconstructed as independent tasks, that happens by itself here; if the stack is ever
#    solved jointly, check what the scaling (`ReconstructionConfig(; scaling = ...)`) does to
#    the near-null images.
#  * Sparsity across `:contrast` is not the sparsity a cine has. If a joint reconstruction is
#    wanted, `LowRank` (the recovery curve lives in a few components) is the right prior, not
#    a temporal Fourier transform.
#
# Needs a Synapse token: `MRITestData.set_synapse_token!(...)` or `SYNAPSE_AUTH_TOKEN`.
#
# Run: julia --project=examples examples/cmrxrecon2024/t1map.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(CMRXRECON2024, "Mapping/TrainingSet/P157/T1map")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all coils/slices/TIs", x)
save_image("cmrx2024_t1map_direct_ti1", x)
save_image("cmrx2024_t1map_direct_ti9", x; contrast = 9)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, first TI", xs)
save_image("cmrx2024_t1map_sense", xs)
