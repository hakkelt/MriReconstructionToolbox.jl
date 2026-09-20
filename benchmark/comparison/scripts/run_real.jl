# Section: real scanner data — MRT vs SigPy vs BART vs MRIReco.
#   2D M4RAW multi-coil ("Real Data"), its SENSE-combined single channel ("Real Data 1ch"),
#   the large multi-slice 3D FSE knee ("Real 3D", one central slice for the cross-toolkit row),
#   and the OCMR cine ("Real Dynamic", CG-SENSE on one frame).
# CG-SENSE (10 it), TV and L1-wavelet, all on the same 2× phase-encode mask (see the comment on
# the mask in `real_case_rows!` for why CG-SENSE cannot use the acquired k-space here). The
# single-channel case has no CG-SENSE row — see where it is built.
#   MRT_BENCH_REAL_DATA is not required here — this section always runs the real-data rows.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_real.jl --threads=N [--use-mkl]
#
# ## Read `nrmse_gt` here as *similarity to the zero-filled RSS image*, not as accuracy
#
# There is no ground truth for real data. `RealData.jl` builds `reference` as the root-sum-of-squares
# coil combination of the *same* k-space the recon gets — so it carries the same partial-Fourier
# truncation, and for the undersampled rows the same 2× aliasing. A reconstruction that successfully
# removes those artifacts therefore moves *away* from the reference. Measured on the M4RAW 2D case:
# the zero-filled adjoint of the 2×-undersampled data scores 0.383, while every regularized recon
# scores worse — MRT TV 0.605 at the synthetic λ, 0.460 at its best λ (0.3, i.e. nearly a constant
# image), and MRIReco / BART / SigPy all land in 0.55–0.58. λ cannot be calibrated against this
# reference either: the curve has no interior optimum that means anything.
#
# So the real-data rows are a **matched-effort wall-time comparison** plus a cross-toolkit agreement
# check (`nrmse_mrt`). λ comes from the synthetic calibration, which is legitimate for the timing
# question (the cost per iteration does not depend on λ) and is what keeps every toolkit on its own
# convention. Do not quote the real-data `nrmse_gt` column as an accuracy result.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
using NamedDims: unname

# One 2D case = (ksp3 (nx,ny,coil), smaps3, ref (nx,ny)); build the CG-SENSE + undersampled
# TV / wavelet rows for MRT / SigPy / BART / MRIReco.
function real_case_rows!(category, ksp3_raw, smaps3, ref; cgsense = true)
    ksp3 = norm_ksp(CMP_CTYPE.(ksp3_raw))               # unit-RMS so calibrated λ transfers here
    nx, ny, nc = size(ksp3)
    @info "real case" category size = (nx, ny) coils = nc
    add(meth, fw, t, x, xmrt) = push!(results, BenchResult(category, meth, fw, NUM_THREADS, t, mag_nrmse(x, ref), xmrt === nothing ? 0.0 : mag_nrmse(xmrt, x)))

    # What the scanner actually sampled. BART / SigPy / MRIReco infer this from the zeros in the
    # data — BART prints `Acc: 1.31` on the M4RAW case, SigPy's `SenseRecon` derives `weights` from
    # `abs(y).sum(axis=0) > 0` — and MRT is told it explicitly through `subsampling`.
    sampled = dropdims(sum(abs, ksp3, dims = 3), dims = 3) .> 0

    kymask = falses(ny)
    kymask[1:2:ny] .= true
    kymask[max(1, ny ÷ 2 - 8):min(ny, ny ÷ 2 + 8)] .= true
    # The 2× phase-encode mask is intersected with what the scanner actually sampled, for the same
    # reason: the other toolkits see only the zero-filled array, so their effective pattern is the
    # intersection whether we like it or not.
    mask2 = falses(nx, ny); mask2[:, kymask] .= true; mask2 .&= sampled
    ksp_z = copy(ksp3); ksp_z[.!mask2, :] .= 0
    acqu = CartesianAcquisitionInfo(
        NamedDimsArray(ksp3[mask2, :], (:kxy, :coil));
        is3D = false, image_size = (nx, ny), sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(CMP_CTYPE.(smaps3)),
        shifted_image_dims = (:x, :y), subsampling = mask2
    )

    # **CG-SENSE runs on the same 2× mask as the sparsity rows, not on the acquired k-space**, for
    # a reason that took an iteration sweep to see: these sensitivity maps are normalized
    # (Σ|Sᶜ|² ≡ 1 to machine precision), so on a *fully* sampled frame 𝒜ᴴ𝒜 = I and the adjoint
    # image is already the exact least-squares solution. Three of the four cases here are fully
    # sampled, and on them the row measured nothing: BART's `pics` hits its own residual tolerance
    # and stops, while MRT / SigPy / MRIReco are pinned to `CMP_TOL_INNER = 0` and spend all ten.
    # Measured on the knee slice, BART's raw wall time is flat in the iteration count —
    # 174.6 ms at 5, 177.4 at 10, 176.0 at 20, 175.6 at 40 — a fitted 0.58 ms per iteration against
    # MRT's 4.82 ms on the *same* problem, i.e. an apparent 8.4x faster iteration that is an early
    # exit. On the undersampled M4RAW case, where 𝒜ᴴ𝒜 ≠ I, BART scales linearly and its
    # per-iteration cost is 4.93 ms against MRT's 4.88 ms — the same work rate, as it should be.
    # BART's inner-CG tolerance is hardcoded and not CLI-settable, so the only way to hold every
    # toolkit to the same ten iterations is to give them a problem that needs ten.
    #
    # (The NRMSE *rising* with iterations is CG semi-convergence on noisy real data, not a defect:
    # the least-squares solution is worse than an early iterate. It is the same for every toolkit.)
    #
    # MRT must be told the pattern either way. Handed the dense array with no `subsampling` it
    # solved the fully-sampled problem instead, moved the iterate by 5e-16 in ten iterations, and
    # reported a *different* NRMSE (0.2635) from everyone else (0.3095) while still paying for
    # them. With the mask passed it reproduces SigPy's trajectory digit for digit.
    #
    # `reltol = 0.0`, as in `run_cgsense.jl`: the other three toolkits are given `CMP_TOL_INNER = 0`
    # and run their full 10 iterations, so MRT must not be allowed to exit early here either.
    if cgsense
        mcg = IterativeReconstruction(regularization = (), algorithm = MriReconstructionToolbox.CGNR(maxit = 10, tol = 0.0); maxit = 10, reltol = 0.0)
        tm, _, xm = time_reconstruction(() -> reconstruct(acqu, mcg; verbosity = Silent()))
        add("CG-SENSE (10 it)", FW, tm * 1000, xm, nothing)
        for (fw, f) in (
                ("SigPy", () -> sigpy_recon(:cgsense, ksp_z, smaps3; iterations = 10)),
                (
                    "BART ($(USE_MKL ? "MKL" : "OpenBLAS"))", () -> begin
                        tb, _, rb = time_bart(
                            "pics -S -w 1 -i 10",
                            reshape(ComplexF32.(ksp_z), nx, ny, 1, nc), reshape(ComplexF32.(smaps3), nx, ny, 1, nc)
                        )
                        (tb * 1000, rb[:, :, 1])
                    end,
                ),
                ("MRIReco", () -> mrireco(:cgsense, ksp_z, smaps3, (nx, ny); iterations = 10)),
            )
            try
                t, x = f()
                add("CG-SENSE (10 it)", fw, t, x, xm)
            catch e
                @warn "$fw CG-SENSE ($category) failed" exception = (e, catch_backtrace())
            end
        end
    end

    # Real data has no ground truth, so λ is taken from the synthetic calibration — valid because
    # both k-spaces are unit-RMS normalised (see `norm_ksp`). Each toolkit uses its own λ.
    IT = CMP_OUTER
    for (key, meth, mrtbuild, mrtkind, spm, mrm, bartfn) in (
            (
                :tv, "Total Variation ($IT it)", λ -> TotalVariation2D(λ), :admm, :tv, :tv,
                λ -> "pics -S -w 1 -F -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -R T:3:0:$λ",
            ),
            (
                :wavelet, "L1-Wavelet ($IT it)", mrt_wavelet, :fista, :wavelet, :wavelet,
                λ -> "pics -S -w 1 -e -i $IT -R W:3:0:$λ",
            ),
        )
        λdef = key === :tv ? 0.01 : 0.005
        tm, _, xm = time_reconstruction(() -> mrt_run(acqu, mrtbuild(load_lambda(key, "MRT", λdef)); maxit = IT, kind = mrtkind))
        add(meth, FW, tm * 1000, xm, nothing)
        for (fw, f) in (
                ("SigPy", () -> sigpy_recon(spm, ksp_z, smaps3; λ = load_lambda(key, "SigPy", λdef), iterations = IT)),
                (
                    "BART ($(USE_MKL ? "MKL" : "OpenBLAS"))", () -> begin
                        tb, _, rb = time_bart(
                            bartfn(load_lambda(key, "BART", λdef)),
                            reshape(ComplexF32.(ksp_z), nx, ny, 1, nc), reshape(ComplexF32.(smaps3), nx, ny, 1, nc)
                        )
                        (tb * 1000, rb[:, :, 1])
                    end,
                ),
                ("MRIReco", () -> mrireco(mrm, ksp_z, smaps3, (nx, ny); λ = load_lambda(key, "MRIReco", λdef), iterations = IT)),
            )
            try
                t, x = f()
                add(meth, fw, t, x, xm)
            catch e
                @warn "$fw $meth ($category) failed" exception = (e, catch_backtrace())
            end
        end
    end
    return
end

# 2D M4RAW, full + single-channel. The single-channel case carries the TV / wavelet rows only: at
# 2x a lone coil has no information to unfold the aliasing with, so 𝒜ᴴ𝒜 is singular and CG-SENSE's
# least-squares solution is an affine set rather than a point — each toolkit lands on a different
# member of it depending on its warm start, and comparing them measures the warm starts.
for (category, combine, cg) in (("Real Data", false, true), ("Real Data 1ch", true, false))
    try
        rc = load_real_case(; combine_coils = combine)
        real_case_rows!(category, unname(rc.kspace), unname(rc.smaps), rc.reference; cgsense = cg)
    catch e
        @warn "$category section failed" exception = (e, catch_backtrace())
    end
end

# 3D knee — one central slice for the cross-toolkit CG-SENSE / sparsity row.
try
    rc = load_real_case_3d(; nslices = 1)
    ks = unname(rc.kspace)[:, :, :, 1]
    ss = unname(rc.smaps)[:, :, :, 1]
    real_case_rows!("Real 3D (1 slice)", ks, ss, rc.reference[:, :, 1])
catch e
    @warn "Real 3D section failed" exception = (e, catch_backtrace())
end

# OCMR cine — one frame, CG-SENSE / sparsity.
try
    rc = load_real_dynamic()
    ks = unname(rc.kspace)[:, :, :, 1]                        # already 2×-undersampled (compacted)
    # rebuild a dense frame for the toolkit rows
    ny_full = size(rc.reference, 2)
    ks_full = zeros(CMP_CTYPE, size(ks, 1), ny_full, size(ks, 3))
    ks_full[:, rc.subsampling[2], :] .= ks
    real_case_rows!("Real Dynamic (1 frame)", ks_full, unname(rc.smaps), rc.reference[:, :, 1])
catch e
    @warn "Real Dynamic section failed" exception = (e, catch_backtrace())
end

write_section("real")
