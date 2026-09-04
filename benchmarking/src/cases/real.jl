# Real-scanner-data benchmark groups, backed by `RealData.jl` (MRITestData.jl). Each downloads
# (cached) on first use; a failure is caught and the group's cases are skipped, not fatal.

using MriReconstructionToolbox: DecomposedImage

# CG-SENSE (full k-space) + 2×-undersampled TV / L1-wavelet for one loaded `case`
# (`(; kspace, smaps, reference, image_size, label)` — 2D or dynamic; see `_us_mask`).
function _real_block(category, case)
    nkx, nky = case.image_size[1], case.image_size[2]
    smaps = case.smaps
    ref = case.reference
    @info "real-data case" category label = case.label size = case.image_size coils = size(smaps, ndims(smaps))

    acq_full = CartesianAcquisitionInfo(
        case.kspace; is3D = false, sensitivity_maps = smaps, shifted_image_dims = (:x, :y)
    )
    cases = NamedTuple[
        (
            category, method = "CG-SENSE (10 it)", reference = ref,
            run = () -> reconstruct(
                acq_full,
                IterativeReconstruction(
                    regularization = (), algorithm = MriReconstructionToolbox.CGNR(maxit = 10, tol = 1.0e-14)
                );
                tol = 1.0e-14, maxit = 10, verbose = false,
            ),
        ),
    ]

    mask = falses(nkx, nky)
    mask[:, 1:2:nky] .= true
    mask[:, (nky ÷ 2 - 8):(nky ÷ 2 + 8)] .= true
    kdata_us = NamedDimsArray(unname(case.kspace)[mask, :], (:kxy, :coil))
    acq_us = CartesianAcquisitionInfo(
        kdata_us; is3D = false, image_size = (nkx, nky), sensitivity_maps = smaps,
        shifted_image_dims = (:x, :y), subsampling = mask,
    )
    for (name, reg) in (
            ("Total Variation (30 it)", TotalVariation2D(0.01)),
            ("L1-Wavelet (30 it)", L1Wavelet2D(0.005)),
        )
        push!(
            cases,
            (
                category, method = name, reference = ref,
                run = () -> reconstruct(
                    acq_us, IterativeReconstruction(regularization = reg); maxit = 30, tol = 1.0e-5, verbose = false,
                ),
            ),
        )
    end
    return cases
end

function _try_real(f, what)
    try
        return f()
    catch e
        @warn "real-data group '$what' failed; skipping" exception = (e, catch_backtrace())
        return NamedTuple[]
    end
end

"""
    build_real() -> cases

`"Real Data"` (pinned M4RAW multi-coil member) and `"Real Data 1ch"` (SENSE-combined to a
single virtual channel). CG-SENSE + 2×-undersampled TV / L1-wavelet each.
"""
function build_real()
    cases = NamedTuple[]
    for (category, combine) in (("Real Data", false), ("Real Data 1ch", true))
        append!(
            cases, _try_real(category) do
                _real_block(category, load_real_case(; combine_coils = combine))
            end
        )
    end
    return cases
end

"""
    build_real3d() -> cases

`"Real 3D"`: the large multi-slice Stanford 3D FSE knee (`RealData.PINNED_3D`), reconstructed
slab-by-slab — many independent slices, each above the per-slab threading threshold, so this is
the group where multi-threading is expected to pay. CG-SENSE + 2×-undersampled TV.
"""
function build_real3d()
    return _try_real("Real 3D") do
        case = load_real_case_3d()
        smaps = case.smaps
        ref = case.reference
        @info "real-3d case" label = case.label size = case.image_size

        acq_full = CartesianAcquisitionInfo(
            case.kspace; is3D = false, sensitivity_maps = smaps, shifted_image_dims = (:x, :y)
        )
        nkx, nky = case.image_size[1], case.image_size[2]
        mask = falses(nky)
        mask[1:2:nky] .= true
        mask[(nky ÷ 2 - 12):(nky ÷ 2 + 12)] .= true
        ksp_us = unname(case.kspace)[:, mask, :, :]            # (kx, n_sampled, coil, z)
        acq_us = CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil, :z)}(ksp_us);
            is3D = false, image_size = (nkx, nky), sensitivity_maps = smaps,
            shifted_image_dims = (:x, :y), subsampling = (:, mask),
        )
        NamedTuple[
            (
                category = "Real 3D", method = "CG-SENSE (10 it)", reference = ref,
                run = () -> reconstruct(
                    acq_full,
                    IterativeReconstruction(regularization = (), algorithm = MriReconstructionToolbox.CGNR(maxit = 10, tol = 1.0e-14));
                    tol = 1.0e-14, maxit = 10, verbose = false,
                ),
            ),
            (
                category = "Real 3D", method = "Total Variation (30 it)", reference = ref,
                run = () -> reconstruct(
                    acq_us, IterativeReconstruction(regularization = TotalVariation2D(0.01)); maxit = 30, tol = 1.0e-5, verbose = false,
                ),
            ),
        ]
    end
end

"""
    build_realdyn() -> cases

`"Real Dynamic"`: an OCMR fully-sampled cardiac cine (`RealData.PINNED_DYNAMIC`), retrospectively
2×-undersampled in phase encode, reconstructed with global low-rank / locally low-rank /
temporal TV — the dynamic counterpart of the synthetic `"Dynamic"` group.
"""
function build_realdyn()
    return _try_real("Real Dynamic") do
        case = load_real_dynamic()
        smaps = case.smaps
        ref = case.reference
        @info "real-dynamic case" label = case.label size = case.image_size
        acq = CartesianAcquisitionInfo(
            case.kspace; is3D = false, image_size = (case.image_size[1], case.image_size[2]),
            sensitivity_maps = smaps, subsampling = case.subsampling, shifted_image_dims = (:x, :y),
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
                    category = "Real Dynamic", method = name, reference = ref,
                    run = () -> reconstruct(acq, IterativeReconstruction(regularization = reg); maxit = 20, tol = 1.0e-4, verbose = false),
                ),
            )
        end
        cases
    end
end
