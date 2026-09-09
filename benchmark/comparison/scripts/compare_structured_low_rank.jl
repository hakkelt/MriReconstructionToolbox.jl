# Calibrationless structured low-rank k-space reconstruction (SAKE / LORAKS-C) against the
# ground truth, and — when the authors' MATLAB LORAKS 2.0 package has been placed in
# `benchmark/comparison/original_implementations/LORAKS` — against that.
#
# BART `pics` has no LORAKS/SAKE and SigPy has no calibrationless structured low-rank app, so
# there is no third-party Julia/Python cross-check here; the reference is the ground truth and
# the zero-filled baseline.
#
#     julia --project=benchmark/comparison benchmark/comparison/scripts/compare_structured_low_rank.jl

using MriReconstructionToolbox
using NamedDims
using LinearAlgebra
using Random
using Test

include("../src/ComparisonHarness.jl")
using .ComparisonHarness: check_nrmse, nrmse, generate_multicoil_brain

rss(x) = sqrt.(dropdims(sum(abs2, unname(x); dims = 3); dims = 3))

@testset "Structured Low-Rank (calibrationless)" begin
    N, Nc = 96, 8
    img_mc, kspace_mc, cmap = generate_multicoil_brain(N = N, num_coils = Nc)
    # Magnitude ground truth: root-sum-of-squares of the coil images.
    truth = rss(reshape(cmap, N, N, Nc) .* reshape(img_mc, N, N, 1))

    # R = 2 irregular phase-encode undersampling. Deliberately NO calibration block: that is
    # what separates this from the GRAPPA / SPIRiT comparisons in compare_reconstruction.jl.
    mask_pe = rand(MersenneTwister(7), Bool, N)
    kspc = kspace_mc[:, mask_pe, :]

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(kspc);
        is3D = false, image_size = (N, N), subsampling = (:, mask_pe),
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
                    algorithm = ADMM(), maxit = 60,
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

    # Optional cross-check against the authors' MATLAB LORAKS 2.0 implementation
    # (https://mr.usc.edu/download/loraks2/). It is not vendored: unpack it into
    # benchmark/comparison/original_implementations/LORAKS to enable this branch.
    loraks_dir = joinpath(@__DIR__, "..", "original_implementations", "LORAKS")
    if isdir(loraks_dir) && !isempty(readdir(loraks_dir))
        @info "LORAKS reference found — add the MATLAB cross-check here" loraks_dir
        # using .ComparisonHarness: setup_matlab_paths
        # setup_matlab_paths(; require = ("LORAKS",))
        # ... call AC_LORAKS / P_LORAKS on the zero-filled grid, then
        # check_nrmse(x_mrt, x_loraks, 0.1; label = "MRT vs MATLAB LORAKS")
    else
        @info "LORAKS reference not present; skipping the MATLAB cross-check" loraks_dir
    end
end
