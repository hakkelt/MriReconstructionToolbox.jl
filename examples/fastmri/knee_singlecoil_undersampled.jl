#!/usr/bin/env julia
#
# fastMRI — knee, single-coil, undersampled test split (`singlecoil_test/file1001483`)
#
# The single-coil track of fastMRI is not a single-channel acquisition: it is an
# emulated one, built by combining the multi-coil data into one complex image and
# transforming that back to k-space (hence `coil_data = :derived`). The k-space is
# `(:kx, :ky, :coil, :z)` with `:coil` of length 1, 110 of 368 phase-encoding lines, 40
# slices.
#
# With one channel there is no parallel imaging: the coil axis carries no information to
# unfold with, so `estimate_sensitivities` returns a trivial map and the only thing that can
# recover the missing lines is the prior. That makes this the cleanest place to compare
# regularizers — nothing else contributes to the result.
#
# fastMRI needs registration: the signed download URLs from the access e-mail are registered
# once with `MRITestData.set_fastmri_urls!`, and they expire after 90 days.
#
# Run: julia --project=examples examples/fastmri/knee_singlecoil_undersampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(FASTMRI, "singlecoil_test/file1001483")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)

x = reconstruct(acq)
report("direct, all slices", x)
save_image("fastmri_knee_singlecoil_test_direct", x; z = 20)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

# One channel, so these differ only in the prior.
for (name, reg) in (("wavelet", L1Wavelet2D(1.0f-3)), ("total variation", TotalVariation2D(1.0f-3)))
    xcs = reconstruct(acq1, IterativeReconstruction(reg; maxit = 50))
    report("CS, $(name)", xcs)
    save_image("fastmri_knee_singlecoil_test_cs_$(replace(name, ' ' => '_'))", xcs)
end
