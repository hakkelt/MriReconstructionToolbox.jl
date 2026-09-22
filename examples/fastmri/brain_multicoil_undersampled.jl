#!/usr/bin/env julia
#
# fastMRI — brain, multi-coil, undersampled test split
# (`multicoil_test/file_brain_AXFLAIR_203_6000903`)
#
# The fastMRI test splits are the released *undersampled* k-space: 322 profiles, a declared
# 512 x 213 encoding, 14 slices, and only 23 of the 213 phase-encoding lines actually
# acquired. The k-space therefore comes out as `(:kx, :ky, :coil, :z)` with `:ky` of length
# 23, while `acq.image_size` stays at the full matrix and `acq.subsampling` records which
# lines those 23 are.
#
# That is the interesting part of this example: the encoding operator only ever touches the
# sampled lines, so an accelerated reconstruction costs no more than the samples that exist,
# and the missing lines are filled by the prior rather than by zero padding.
#
# fastMRI needs registration: the signed download URLs from the access e-mail are registered
# once with `MRITestData.set_fastmri_urls!`, and they expire after 90 days.
#
# Run: julia --project=examples examples/fastmri/brain_multicoil_undersampled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(FASTMRI, "multicoil_test/file_brain_AXFLAIR_203_6000903")
describe(raw)

acq = AcquisitionInfo(raw)
describe(acq)
println("  reconstructing a $(acq.image_size) image from $(size(acq.kspace_data, :ky)) ky lines")

x = reconstruct(acq)
report("direct, all coils/slices", x)
save_image("fastmri_brain_test_direct", x)

acq1 = estimate_sensitivities(AcquisitionInfo(first_slab(raw)))

xs = reconstruct(acq1)
report("SENSE, one slice", xs)
save_image("fastmri_brain_test_sense", xs)

# The undersampling is what compressed sensing is for: with 23 of 213 lines, the wavelet
# prior does the work the missing lines cannot.
xcs = reconstruct(acq1, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 50))
report("CS, wavelet sparsity", xcs)
save_image("fastmri_brain_test_cs", xcs)
