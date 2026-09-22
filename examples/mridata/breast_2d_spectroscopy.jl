#!/usr/bin/env julia
#
# mridata.org — single-voxel breast spectroscopy, catalogued as breast 2D
# (`e3a5bed8-eb13-416e-bfba-0dee85f0ef5e`)
#
# The catalog's only `anatomy = :breast, acquisition_dim = 2` mridata.org entry, and it is
# spectroscopy: `svs_se_breast_int_ref`, one profile of 2144 samples on 4 channels, 500 us
# dwell, no phase encoding. Same protocol family as
# `spectroscopy_svs_unlabelled.jl` — the two differ only in the anatomy label the uploader
# gave them, which is why both appear as separate "types" of the source.
#
# Worth keeping in mind when a script iterates over catalog groups: a (source, anatomy,
# dimension) group is a metadata grouping, not a promise that the files in it are images, and
# the only reliable check is the encoding limits in the file itself.
#
# Run: julia --project=examples examples/mridata/breast_2d_spectroscopy.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase
using FFTW
using Statistics

setup()

raw = load_example(MRIDATA, "e3a5bed8-eb13-416e-bfba-0dee85f0ef5e")
describe(raw)

steps1 = length(unique(p.head.idx.kspace_encode_step_1 for p in raw.profiles))
steps2 = length(unique(p.head.idx.kspace_encode_step_2 for p in raw.profiles))
println("  encoding steps: $(steps1) x $(steps2) — no image to reconstruct")

try
    AcquisitionInfo(raw)
catch err
    println("  AcquisitionInfo refuses it: $(sprint(showerror, err))")
end

spectrum = vec(sum(abs2, fftshift(fft(raw.profiles[1].data, 1), 1); dims = 2))
println(
    "  spectral peak at bin $(argmax(spectrum)) of $(length(spectrum)), " *
        "$(round(10 * log10(maximum(spectrum) / median(spectrum)); digits = 1)) dB above the median bin"
)
