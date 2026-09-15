#!/usr/bin/env julia
#
# mridata.org — semi-LASER spectroscopy, not an image (`a133799b-e621-4f97-aedb-1882df644372`)
#
# The catalog's `anatomy = :other` entries are not all images, and this is one that is not.
# The protocol is `mslaser_wref1_onlyRFoff` from mridata.org's "MRS TEST" project: a
# semi-LASER single-voxel spectroscopy water reference, RF off. Ten profiles of 4128 samples
# on 34 channels, one phase-encoding step, a declared encoded matrix of 2 x 2, and a 1.8 us
# dwell time.
#
# `AcquisitionInfo(raw)` refuses it:
#
#     ArgumentError: readout centering places samples outside the encoded matrix
#                    (-2062:2065 vs 1:2); assemble manually
#
# which is the correct answer — there is no image in this file to reconstruct. The signal is a
# free induction decay per channel, and what it is for is a spectrum: Fourier transform along
# the readout, with the frequency axis set by the dwell time and the ppm axis by the proton
# resonance frequency in the header.
#
# The point of the example is the diagnosis. One encoding step plus a 2 x 2 encoded matrix
# plus a dwell time far from the 1-30 us of an imaging readout is the signature of a
# non-imaging acquisition; check for it before blaming the reconstruction.
#
# Run: julia --project=examples examples/mridata/spectroscopy_semilaser.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase
using FFTW

setup()

raw = load_example(MRIDATA, "a133799b-e621-4f97-aedb-1882df644372")
describe(raw)

encode_steps = length(unique(p.head.idx.kspace_encode_step_1 for p in raw.profiles))
println("  phase-encoding steps: $(encode_steps), encoded matrix: $(Int.(raw.params["encodedSize"]))")
println("  protocol: $(get(raw.params, "protocolName", "?"))")

# What MRT says about it, for the record.
try
    AcquisitionInfo(raw)
catch err
    println("  AcquisitionInfo refuses it: $(sprint(showerror, err))")
end

# The reconstruction such data does have: an FID per channel, transformed to a spectrum.
fid = raw.profiles[1].data                       # (samples, channels)
# Channel phases are arbitrary, so add the channels in power, not in amplitude.
spectrum = vec(sum(abs2, fftshift(fft(fid, 1), 1); dims = 2))
dwell_s = raw.profiles[1].head.sample_time_us * 1.0e-6
bandwidth_hz = 1 / dwell_s
larmor_hz = get(raw.params, "H1resonanceFrequency_Hz", 0.0)
peak = argmax(spectrum)
offset_hz = (peak - 1 - size(fid, 1) ÷ 2) * bandwidth_hz / size(fid, 1)
println("  $(size(fid, 1)) samples, $(size(fid, 2)) channels, bandwidth $(round(bandwidth_hz / 1000; digits = 1)) kHz")
println(
    "  strongest line $(round(offset_hz; digits = 1)) Hz from the carrier" *
        (larmor_hz > 0 ? ", i.e. $(round(1.0e6 * offset_hz / larmor_hz; digits = 2)) ppm" : "")
)
