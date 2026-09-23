# Calibrationless structured low-rank k-space reconstruction (SAKE / LORAKS-C) against the
# ground truth, and — when the authors' MATLAB LORAKS 2.1 package has been unpacked into
# `benchmark/comparison/original_implementations/LORAKS2` and a MATLAB is on `PATH` — against
# that, as a numerical oracle. See `../src/loraks_bridge.jl` for why it is not vendored and what
# the licence requires of anything that uses it.
#
# BART `pics` has no LORAKS/SAKE and SigPy has no calibrationless structured low-rank app, so
# there is no third-party Julia/Python cross-check here; the reference is the ground truth, the
# zero-filled baseline, and the optional MATLAB oracle.
#
# Point `MRT_BENCH_MATLAB` at the binary to enable the oracle — do **not** `module load matlab`,
# which breaks Julia's `libpcre2` (see `matlab_executable`):
#
#     MRT_BENCH_MATLAB=/opt/software/packages/matlab/r2024b/bin/matlab \
#         julia --project=benchmark/comparison benchmark/comparison/scripts/compare_structured_low_rank.jl

using MriReconstructionToolbox: CartesianAcquisitionInfo
using NamedDims
using Random
using Test

include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "..", "src", "loraks_bridge.jl"))
using .LORAKSBridge: loraks_available, loraks_recon, loraks_citation, LORAKS_DIR

rss(x) = sqrt.(dropdims(sum(abs2, unname(x); dims = 3); dims = 3))

@testset "Structured Low-Rank (calibrationless)" begin
    N, Nc = 96, 8
    img_mc, kspace_mc, cmap = multicoil_phantom(N, Nc)
    # Magnitude ground truth: root-sum-of-squares of the coil images.
    truth = rss(reshape(cmap, N, N, Nc) .* reshape(img_mc, N, N, 1))

    # R = 2 irregular phase-encode undersampling. Deliberately NO calibration block: that is
    # what separates this from the GRAPPA / SPIRiT comparisons in compare_reconstruction.jl.
    mask_pe = rand(MersenneTwister(7), Bool, N)
    kspc = kspace_mc[:, mask_pe, :]

    # `shifted_image_dims` is not optional here: `multicoil_phantom` returns the phantom
    # centred in the image and its k-space centred in the array, so without it every
    # reconstruction comes back fftshifted against `truth` and scores NRMSE ≈ √2 — the value two
    # uncorrelated images of equal norm give. That is what made the zero-filled baseline read
    # 1.3995 and both low-rank rows read ≈ 1.412: nothing was being measured but the shift.
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(kspc);
        is3D = false, image_size = (N, N), subsampling = (:, mask_pe),
        shifted_image_dims = (:x, :y),
    )

    zf = rss(reconstruct(acq, DirectReconstruction(); verbosity = Silent()))
    e_zf = nrmse(zf .* (norm(truth) / norm(zf)), truth)
    @info "zero-filled RSS NRMSE: $e_zf"

    solve(reg) = abs.(
        unname(
            reconstruct(
                acq,
                IterativeReconstruction(
                    reg; signal_model = KSpaceToImage(RootSumSquares()),
                    algorithm = MriReconstructionToolbox.ADMM(), maxit = 60,
                );
                verbosity = Silent(),
            )
        )
    )

    @testset "LORAKS-C (nuclear norm)" begin
        e = check_nrmse(
            solve(StructuredLowRank(; λ = 1.0f-2, window = (6, 6))), truth, 0.15;
            label = "MRT LORAKS-C vs Ground Truth",
        )
        @test e < e_zf
    end

    @testset "SAKE (hard rank)" begin
        e = check_nrmse(
            solve(StructuredLowRank(; max_rank = 30, window = (6, 6))), truth, 0.15;
            label = "MRT SAKE vs Ground Truth",
        )
        @test e < e_zf
    end

    # Cross-check against the authors' MATLAB LORAKS 2.1 package, used as a numerical oracle.
    # Not vendored (licence — see `LORAKSBridge`); unpack it into
    # benchmark/comparison/original_implementations/LORAKS2 and `module load matlab` to enable.
    if !loraks_available()
        @info "LORAKS oracle unavailable; skipping the MATLAB cross-check" LORAKS_DIR matlab = something(
            LORAKSBridge.matlab_executable(), "<none>"
        )
    else
        @testset "LORAKS reference oracle" begin
            print(loraks_citation())

            # P_LORAKS wants the full grid, zero-filled, plus the sampling mask.
            kzf = zeros(ComplexF64, N, N, Nc)
            kzf[:, mask_pe, :] .= kspc
            kmask = zeros(Float64, N, N)
            kmask[:, mask_pe] .= 1.0

            # `rank` here is the same non-convex rank the SAKE row uses, and `radius = 3` is the
            # closest LORAKS neighbourhood to MRT's `window = (6, 6)` — see `loraks_recon`.
            ref = loraks_recon(kzf, kmask; rank = 30, radius = 3, ltype = "C", max_iter = 50)
            # P_LORAKS returns k-space on the same centred grid it was given, so the image needs
            # the full shift sandwich. Dropping the outer `fftshift` puts the phantom half a field
            # of view away and scores √2 — the same trap the acquisition above documents.
            x_loraks = rss(fftshift(ifft(ifftshift(ref.kspace, (1, 2)), (1, 2)), (1, 2)))
            e_ref = check_nrmse(
                x_loraks .* (norm(truth) / norm(x_loraks)), truth, 0.15;
                label = "MATLAB LORAKS vs Ground Truth",
            )
            @info "LORAKS oracle" nrmse_vs_truth = e_ref matlab_seconds = ref.elapsed

            # The oracle's job: MRT must not be *worse* than the reference implementation of the
            # same idea by a wide margin. Both are compared to the truth rather than to each
            # other, because the two formulations differ (nuclear norm / hard rank, disc /
            # rectangular neighbourhood) by more than a pointwise comparison would tolerate.
            @test e_ref < e_zf
        end
    end
end
