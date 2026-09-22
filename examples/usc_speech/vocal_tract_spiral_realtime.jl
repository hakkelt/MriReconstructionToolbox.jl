#!/usr/bin/env julia
#
# USC Speech — 8-channel spiral real-time vocal-tract imaging (`sub029/2drt/04_bvt_r2`)
#
# The only non-Cartesian source in the catalog, and the one that exercises the NUFFT path:
# a 13-interleaf spiral-out readout, 630 samples per arm, 8 channels, an 84x84 encoded
# matrix, acquired continuously at ~83 frames per second while the subject speaks.
#
# Three things are characteristic of this data:
#
#  * The third trajectory row is the vendor's density compensation, not a kz coordinate.
#    The extension recognises a trajectory row beyond the encoding dimensions and reads it as
#    the `density_compensation` of the acquisition, which is what makes the gridding weight
#    the oversampled spiral centre correctly.
#  * `sample_time_us` is zero in every profile; `ExampleUtils.repair_dwell!` recovers it from
#    the trajectory description (4 us here). The direct gridding does not use it, anything
#    working in physical time does.
#  * Every profile carries its own `repetition` counter, so nothing in the header groups the
#    interleaves into frames. Reconstructed as loaded, all arms grid into one
#    motion-averaged image. Renumbering `repetition` per frame (below) gives the extension a
#    counter it can turn into a `:time` batch dimension, and the reconstruction becomes a
#    frame-by-frame real-time series.
#
# Sensitivity estimation is Cartesian-only — there is no calibration window in a spiral
# acquisition — so grid the samples first (the direct reconstruction here) and estimate maps
# from that image if parallel imaging is wanted.
#
# Run: julia --project=examples examples/usc_speech/vocal_tract_spiral_realtime.jl

isdefined(Main, :ExampleUtils) || include(joinpath(@__DIR__, "..", "ExampleUtils.jl"))
using .ExampleUtils
using MriReconstructionToolbox
using MRITestData
using MRIBase

setup()

raw = load_example(USC_SPEECH, "sub029/2drt/04_bvt_r2")
describe(raw)

ninterleaves = length(unique(p.head.idx.kspace_encode_step_1 for p in raw.profiles))
println("  $(ninterleaves) interleaves per frame, $(length(raw.profiles) ÷ ninterleaves) frames in the file")

# Keep a handful of frames and number them in `repetition` so that the frame structure is in
# the counters rather than implicit in profile order.
nframes = 8
frames = raw.profiles[1:(nframes * ninterleaves)]
for (i, p) in enumerate(frames)
    p.head.idx.repetition = (i - 1) ÷ ninterleaves
end
dynamic = typeof(raw)(raw.params, frames)

acq = AcquisitionInfo(dynamic)
describe(acq)

x = reconstruct(acq)
report("gridded, $(nframes) frames", x)
save_image("usc_spiral_frame1", x)
save_image("usc_spiral_frame$(nframes)", x; time = nframes)

# Total variation in the image plane suits the low-SNR spiral frames better than wavelets.
xcs = reconstruct(acq, IterativeReconstruction(TotalVariation2D(1.0f-2); maxit = 20))
report("CS, total variation", xcs)
save_image("usc_spiral_cs_frame1", xcs)
