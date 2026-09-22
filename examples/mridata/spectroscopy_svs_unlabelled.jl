#!/usr/bin/env julia
#
# mridata.org — single-voxel breast spectroscopy, left side
# (`a9c5a204-56ec-449b-ba7f-aa750d835337`)
#
# The smallest file in the whole catalog, 0.1 MB, and not an image either: protocol
# `svs_se_breast_int_ref LEFT`, a spin-echo single-voxel spectroscopy internal reference. One
# profile, 2144 samples, 4 channels, no phase encoding at all, a 500 us dwell time (2 kHz
# bandwidth), and a declared encoded matrix of 2 x 2.
#
# Its catalog entry carries `fully_sampled = nothing` — the source has no notion of sampling
# for an acquisition with no phase encoding — which is a useful signal in itself when
# filtering the catalog for imaging data: require `fully_sampled !== nothing` and an
# `acquisition_dim` matching the encoding limits.
#
# `AcquisitionInfo` refuses it for the same reason as `spectroscopy_semilaser.jl`. That file
# has the longer discussion; this one shows what a single-shot FID looks like.
#
# Run: julia --project=examples examples/mridata/spectroscopy_svs_unlabelled.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase
using FFTW

setup()

raw = load_example(MRIDATA, "a9c5a204-56ec-449b-ba7f-aa750d835337")
describe(raw)
println("  profiles: $(length(raw.profiles)), protocol: $(get(raw.params, "protocolName", "?"))")

try
    AcquisitionInfo(raw)
catch err
    println("  AcquisitionInfo refuses it: $(sprint(showerror, err))")
end

fid = raw.profiles[1].data
spectrum = vec(sum(abs2, fftshift(fft(fid, 1), 1); dims = 2))
bandwidth_hz = 1 / (raw.profiles[1].head.sample_time_us * 1.0e-6)
decay = sum(abs2, fid; dims = 2) |> vec
println("  FID: $(length(decay)) samples, bandwidth $(round(bandwidth_hz; digits = 1)) Hz")
println("  signal decays to $(round(100 * decay[end] / decay[1]; digits = 2))% of its first sample")
println("  spectral peak at bin $(argmax(spectrum)) of $(length(spectrum))")
