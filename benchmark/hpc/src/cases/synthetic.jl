# Synthetic-phantom benchmark groups. Each `build_*` returns a `Vector{NamedTuple}` of
# `(category, method, reference, run)`; `run()` performs one reconstruction, `reference` is the
# ground-truth image for the NRMSE check. Phantoms / masks / iteration counts match the MRT rows
# of `benchmark/comparison/scripts/` so the two suites are directly comparable.

using Random: MersenneTwister

"""
    build_base(; N = 128, Nc = 8) -> cases

`"Base MC"`: unregularised CG-SENSE on a fully-sampled multi-coil brain phantom.
"""
function build_base(; N = 128, Nc = 8)
    img_mc, kspace_mc, cmap = generate_multicoil_brain(N = N, num_coils = Nc)
    acq_mc = CartesianAcquisitionInfo(
        NamedDimsArray(kspace_mc, (:kx, :ky, :coil));
        is3D = false, sensitivity_maps = NamedDimsArray(cmap, (:x, :y, :coil)),
        shifted_image_dims = (:x, :y),
    )
    return NamedTuple[
        (
            category = "Base MC", method = "CG-SENSE (10 it)", reference = img_mc,
            run = () -> reconstruct(
                acq_mc,
                IterativeReconstruction(
                    regularization = (), algorithm = MriReconstructionToolbox.CGNR(maxit = 10, tol = 1.0e-14)
                );
                tol = 1.0e-14, maxit = 10, verbose = false,
            ),
        ),
    ]
end

"""
    build_sparsity(; N = 128, Nc = 8) -> cases

`"Sparsity"`: 2×-undersampled TV / L1-wavelet / TGV on the multi-coil brain phantom.
"""
function build_sparsity(; N = 128, Nc = 8)
    img_mc, kspace_mc, cmap = generate_multicoil_brain(N = N, num_coils = Nc)
    smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))
    mask_reg = rand(MersenneTwister(42), Bool, N, N)
    mask_reg[(N ÷ 2 - 8):(N ÷ 2 + 8), :] .= true
    acq_reg = CartesianAcquisitionInfo(
        NamedDimsArray(kspace_mc[mask_reg, :], (:kxy, :coil));
        is3D = false, image_size = (N, N), sensitivity_maps = smaps_mc,
        shifted_image_dims = (:x, :y), subsampling = mask_reg,
    )
    cases = NamedTuple[]
    for (name, reg, it) in (
            ("Total Variation (30 it)", TotalVariation2D(0.01), 30),
            ("L1-Wavelet (30 it)", L1Wavelet2D(0.005), 30),
            ("TGV (30 it)", TotalGeneralizedVariation2D(0.01; ratio = 2.0), 30),
        )
        push!(
            cases,
            (
                category = "Sparsity", method = name, reference = img_mc,
                run = () -> reconstruct(
                    acq_reg, IterativeReconstruction(regularization = reg); maxit = it, tol = 1.0e-5, verbose = false,
                ),
            ),
        )
    end
    return cases
end

"""
    build_dynamic(; Nd = 64, Ncd = 4, Td = 8) -> cases

`"Dynamic"`: global / locally low-rank + temporal-TV on an undersampled dynamic brain phantom.
"""
function build_dynamic(; Nd = 64, Ncd = 4, Td = 8)
    img_dyn, kspace_dyn, cmap_dyn = generate_dynamic_multicoil_brain(N = Nd, num_coils = Ncd, num_frames = Td)
    mask_pe = rand(MersenneTwister(42), Bool, Nd)
    mask_pe[(Nd ÷ 2 - 4):(Nd ÷ 2 + 4)] .= true
    acq_dyn = CartesianAcquisitionInfo(
        NamedDimsArray(
            permutedims(kspace_dyn[:, mask_pe, :, :], (1, 2, 4, 3)), (:kx, :ky, :coil, :time)
        );
        is3D = false, image_size = (Nd, Nd),
        sensitivity_maps = NamedDimsArray(cmap_dyn, (:x, :y, :coil)),
        subsampling = (:, mask_pe), shifted_image_dims = (:x, :y),
    )
    cases = NamedTuple[]
    for (name, reg) in (
            ("Global Low-Rank (20 it)", LowRank(0.01; time_dim = :time)),
            ("Locally Low-Rank (20 it)", LocallyLowRank(0.01; block_size = (8, 8), time_dim = :time)),
            ("Temporal TV (20 it)", TemporalTotalVariation(0.01; time_dim = :time)),
        )
        push!(
            cases,
            (
                category = "Dynamic", method = name, reference = img_dyn,
                run = () -> reconstruct(
                    acq_dyn, IterativeReconstruction(regularization = reg); maxit = 20, tol = 1.0e-4, verbose = false,
                ),
            ),
        )
    end
    return cases
end

"""
    build_kspace(; N = 128, Nc = 8) -> cases

`"K-Space"`: GRAPPA with RSS and adjoint-sensitivity coil combination.
"""
function build_kspace(; N = 128, Nc = 8)
    img_mc, kspace_mc, cmap = generate_multicoil_brain(N = N, num_coils = Nc)
    smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))
    mask_g = falses(N, N)
    mask_g[:, 1:2:N] .= true
    mask_g[:, (N ÷ 2 - 12):(N ÷ 2 + 11)] .= true
    acq_g = CartesianAcquisitionInfo(
        NamedDimsArray(kspace_mc[mask_g, :], (:kxy, :coil));
        is3D = false, image_size = (N, N), sensitivity_maps = smaps_mc,
        subsampling = mask_g, shifted_image_dims = (:x, :y),
    )
    cases = NamedTuple[]
    for (name, cc) in (("GRAPPA (RSS)", RootSumSquares()), ("GRAPPA (Sensitivity)", AdjointSensitivity()))
        push!(
            cases,
            (
                category = "K-Space", method = name, reference = img_mc,
                run = () -> reconstruct(
                    acq_g, GRAPPA(kernel_size = (4, 3), calib_size = (24, 24), coil_combination = cc); verbose = false,
                ),
            ),
        )
    end
    return cases
end
