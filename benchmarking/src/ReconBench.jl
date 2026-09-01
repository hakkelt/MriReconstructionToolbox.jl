module ReconBench

# MRT-only reconstruction timings. Same phantoms, masks and iteration counts as the MRT rows
# of comparison/scripts/run_benchmarks.jl, so the JSON written here can be used directly as the
# baseline column there.

using LinearAlgebra
using Statistics: median
using Random: MersenneTwister
using FFTW: fft, ifftshift
using NamedDims
using MriReconstructionToolbox

include(joinpath(@__DIR__, "Phantoms.jl"))
using .Phantoms
include(joinpath(@__DIR__, "RealData.jl"))
using .RealData

export build_cases, run_cases

nrmse(x, xref) = norm(vec(x) .- vec(xref)) / norm(vec(xref))
aligned_nrmse(est, ref) = nrmse(est .* (norm(abs.(ref)) / norm(abs.(est))), ref)

"""
    build_cases(; N=128, Nc=8, Nd=64, Ncd=4, Td=8)

Return a `Vector{NamedTuple}` of `(category, method, run, reference)` where `run()` performs one
reconstruction and `reference` is the ground-truth image for the NRMSE check.

Set `ENV["MRT_BENCH_REAL_DATA"] = "1"` to append a "Real Data" block driven by real scanner
k-space from `MRITestData.jl` (see `RealData.jl`); it downloads a small dataset on first use.
"""
function build_cases(; N = 128, Nc = 8, Nd = 64, Ncd = 4, Td = 8, real_data = get(ENV, "MRT_BENCH_REAL_DATA", "0") == "1")
    cases = NamedTuple[]

    img_mc, kspace_mc, cmap = generate_multicoil_brain(N = N, num_coils = Nc)
    kdata_mc = NamedDimsArray(kspace_mc, (:kx, :ky, :coil))
    smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))
    acq_mc = CartesianAcquisitionInfo(
        kdata_mc; is3D = false, sensitivity_maps = smaps_mc, shifted_image_dims = (:x, :y)
    )

    push!(
        cases,
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
    )

    # 2x-undersampled sparsity set
    mask_reg = rand(MersenneTwister(42), Bool, N, N)
    mask_reg[(N ÷ 2 - 8):(N ÷ 2 + 8), :] .= true
    kdata_reg = NamedDimsArray(kspace_mc[mask_reg, :], (:kxy, :coil))
    acq_reg = CartesianAcquisitionInfo(
        kdata_reg; is3D = false, image_size = (N, N), sensitivity_maps = smaps_mc,
        shifted_image_dims = (:x, :y), subsampling = mask_reg,
    )

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

    # Dynamic / low-rank set
    img_dyn, kspace_dyn, cmap_dyn = generate_dynamic_multicoil_brain(N = Nd, num_coils = Ncd, num_frames = Td)
    mask_pe = rand(MersenneTwister(42), Bool, Nd)
    mask_pe[(Nd ÷ 2 - 4):(Nd ÷ 2 + 4)] .= true
    kdata_dyn = NamedDimsArray(
        permutedims(kspace_dyn[:, mask_pe, :, :], (1, 2, 4, 3)), (:kx, :ky, :coil, :time)
    )
    acq_dyn = CartesianAcquisitionInfo(
        kdata_dyn; is3D = false, image_size = (Nd, Nd),
        sensitivity_maps = NamedDimsArray(cmap_dyn, (:x, :y, :coil)),
        subsampling = (:, mask_pe), shifted_image_dims = (:x, :y),
    )

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

    # GRAPPA
    mask_g = falses(N, N)
    mask_g[:, 1:2:N] .= true
    mask_g[:, (N ÷ 2 - 12):(N ÷ 2 + 11)] .= true
    kdata_g = NamedDimsArray(kspace_mc[mask_g, :], (:kxy, :coil))
    acq_g = CartesianAcquisitionInfo(
        kdata_g; is3D = false, image_size = (N, N), sensitivity_maps = smaps_mc,
        subsampling = mask_g, shifted_image_dims = (:x, :y),
    )
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

    real_data && append_real_cases!(cases)

    return cases
end

"""
    append_real_cases!(cases) -> cases

Append CG-SENSE / TV / L1-wavelet cases backed by real scanner k-space (`RealData.load_real_case`)
to `cases`. The full k-space gives the CG-SENSE row and the RSS reference; a 2× phase-encode mask
(central 17 lines kept) gives the undersampled sparsity rows. Iteration counts match the synthetic
`Base MC` / `Sparsity` rows so the two are directly comparable. A download failure is caught and
logged rather than aborting the suite.
"""
function append_real_cases!(cases)
    local case
    try
        case = load_real_case()
    catch e
        @warn "MRT_BENCH_REAL_DATA set but load_real_case failed; skipping real-data rows" exception = (e, catch_backtrace())
        return cases
    end
    nkx, nky = case.image_size
    smaps = case.smaps
    ref = case.reference
    @info "real-data case" label = case.label size = case.image_size coils = size(smaps, 3)

    acq_full = CartesianAcquisitionInfo(
        case.kspace; is3D = false, sensitivity_maps = smaps, shifted_image_dims = (:x, :y)
    )
    push!(
        cases,
        (
            category = "Real Data", method = "CG-SENSE (10 it)", reference = ref,
            run = () -> reconstruct(
                acq_full,
                IterativeReconstruction(
                    regularization = (), algorithm = MriReconstructionToolbox.CGNR(maxit = 10, tol = 1.0e-14)
                );
                tol = 1.0e-14, maxit = 10, verbose = false,
            ),
        ),
    )

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
                category = "Real Data", method = name, reference = ref,
                run = () -> reconstruct(
                    acq_us, IterativeReconstruction(regularization = reg); maxit = 30, tol = 1.0e-5, verbose = false,
                ),
            ),
        )
    end
    return cases
end

"""
    run_cases(cases; num_runs=3) -> Vector{NamedTuple}

Warm up each case once, then time `num_runs` more; report `time_ms` (minimum) and the aligned
NRMSE against the case reference.
"""
function run_cases(cases; num_runs = 3)
    out = NamedTuple[]
    for c in cases
        res = c.run()
        ts = Float64[]
        for _ in 1:num_runs
            t0 = time_ns()
            res = c.run()
            push!(ts, (time_ns() - t0) / 1.0e9)
        end
        img = res isa DecomposedImage ? total(res) : res
        e = aligned_nrmse(img, c.reference)
        push!(
            out,
            (category = c.category, method = c.method, time_ms = minimum(ts) * 1000, nrmse_gt = e),
        )
        println(rpad(c.method, 28), " ", round(minimum(ts) * 1000; digits = 2), " ms   nrmse=", round(e; digits = 5))
    end
    return out
end

end
