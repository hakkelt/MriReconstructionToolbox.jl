# Section: 2×-undersampled sparsity — TV / L1-wavelet / TGV — MRT vs BART vs SigPy vs MRIReco.
#   julia --project=comparison -t N comparison/scripts/run_sparsity.jl --threads=N [--use-mkl]
#
# Matched-effort comparison: CMP_OUTER (20) outer iterations, CMP_CG_ITERS (10) inner CG, fixed
# ADMM ρ, no early stop for the in-process toolkits. k-space is unit-RMS normalised and each
# toolkit uses its own calibrated λ (`load_lambda`) so the operating point, not the nominal λ, is
# matched. See `run_accuracy_race.jl` for the fairer time-to-target-NRMSE view; this section is the
# fixed-effort snapshot.
#
# ## BART flags — three non-obvious ones, all verified against the 0.9.00 source
#
# * **`-i N` is not the outer iteration count for ADMM.** `src/iter/admm.c:453` breaks on
#   `nr_invokes > maxiter`, and `nr_invokes` accumulates ≈ (outer iterations + cumulative inner CG
#   iterations) — so `-i N` is effectively a budget of N applications of the normal operator.
#   Measured on this problem: `-i 20` → 3 outer / 21 CG, `-i 80` → 29 outer / 52 CG, `-i 150` → 99
#   outer / 52 CG. To give BART an effort comparable to `CMP_OUTER` outer iterations of
#   `CMP_CG_ITERS` inner CG each, `-i` must be roughly their product, hence `BART_BUDGET`.
# * **`-w 1` disables BART's internal k-space rescaling.** `pics` otherwise divides k-space so the
#   ~90th-percentile magnitude of a low-res RSS image is ≈1 (`src/sense/optcom.c:71-87`) and leaves
#   λ untouched, putting its λ on a private scale that no calibration can transfer.
# * **`-F` changes the algorithm, not the iteration count.** The eps_pri/eps_dual break it disables
#   is already dead in `pics` (`src/grecon/italgo.c:142-143` hardcodes `ABSTOL = RELTOL = 0`); what
#   `-F` actually does is switch off over-relaxation, α 1.6 → 1.0 (`admm.c:341-354`). We keep it
#   because plain ADMM is what MRT, MRIReco and SigPy run.
# * **`-e` is mandatory for the FISTA (wavelet) path.** Without it `pics` assumes λ_max = 1 and uses
#   a hardcoded step of 0.95 (`src/pics.c:793-794, 801`). With `-w 1` the true ‖𝒜‖ here is 1.52, so
#   λ_max(𝒜ᴴ𝒜) ≈ 2.3, 0.95 > 2/L, and the recon stalls at NRMSE 0.59 for *every* λ. `-e` runs a
#   30-iteration power method (`src/iter/misc.c:51-65`) — a real one-off cost, counted honestly.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
using Random: MersenneTwister

const IT = CMP_OUTER   # `BART_BUDGET` comes from `_toolkits.jl` — see its docstring for why.

img_mc, cmap = IMG_MC, CMAP
kspace_mc = add_noise(norm_ksp(KSPACE_MC); snr_db = CMP_SNR_DB)   # match calibrate_lambda.jl
smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))

mask_reg = rand(MersenneTwister(42), Bool, N, N)
mask_reg[(N ÷ 2 - 8):(N ÷ 2 + 8), :] .= true
acq_us = CartesianAcquisitionInfo(
    NamedDimsArray(kspace_mc[mask_reg, :], (:kxy, :coil));
    is3D = false, image_size = (N, N), sensitivity_maps = smaps_mc,
    shifted_image_dims = (:x, :y), subsampling = mask_reg,
)
ksp_z = copy(kspace_mc); ksp_z[.!mask_reg, :] .= 0          # zero-filled, for BART / SigPy / MRIReco
kbart = reshape(ksp_z, N, N, 1, Nc)
sbart = reshape(cmap, N, N, 1, Nc)

addrow(cat, meth, fw, t, x, xmrt) =
    push!(results, BenchResult(cat, meth, fw, NUM_THREADS, t, mag_nrmse(x, img_mc), xmrt === nothing ? 0.0 : mag_nrmse(xmrt, x)))

# (key, label, MRT-reg builder, MRT alg kind, sigpy-method, mrireco-method, BART cmd builder(λ), default λ)
specs = (
    (:tv, "Total Variation ($IT it)", λ -> TotalVariation2D(λ), :admm, :tv, :tv,
        λ -> "pics -S -w 1 -F -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -R T:3:0:$λ", 0.01),
    # Wavelet runs FISTA in every toolkit, so there is no inner CG and `-i` *is* the iteration
    # count here (`iter_fista_defaults.tol = 0`, never overridden — BART runs all of them).
    (:wavelet, "L1-Wavelet ($IT it)", mrt_wavelet, :fista, :wavelet, :wavelet,
        λ -> "pics -S -w 1 -e -i $IT -R W:3:0:$λ", 0.005),
    (:tgv, "TGV ($IT it)", λ -> TotalGeneralizedVariation2D(λ; ratio = 2.0), :admm, nothing, nothing,
        λ -> "pics -S -w 1 -F -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -R G:3:0:$λ", 0.01),
)

for (key, meth, mrtreg, mrtkind, spm, mrm, bartcmd, λdef) in specs
    println("--> $meth")
    xm = let
        tm, _, x = time_reconstruction(() -> mrt_run(acq_us, mrtreg(load_lambda(key, "MRT", λdef)); maxit = IT, kind = mrtkind))
        addrow("Sparsity", meth, FW, tm * 1000, x, nothing)
        x
    end

    try
        tb, _, rb = time_bart(bartcmd(load_lambda(key, "BART", λdef)), ComplexF32.(kbart), ComplexF32.(sbart))
        addrow("Sparsity", meth, BART_FW, tb * 1000, rb[:, :, 1], xm)
    catch e
        @warn "BART $meth failed" exception = (e, catch_backtrace())
    end
    if spm !== nothing
        try
            ts, xs = sigpy_recon(spm, ksp_z, cmap; λ = load_lambda(key, "SigPy", λdef), iterations = IT)
            addrow("Sparsity", meth, "SigPy", ts, xs, xm)
        catch e
            @warn "SigPy $meth failed" exception = (e, catch_backtrace())
        end
    end
    if mrm !== nothing
        try
            tr, xr = mrireco(mrm, ksp_z, cmap, (N, N); λ = load_lambda(key, "MRIReco", λdef), iterations = IT)
            addrow("Sparsity", meth, "MRIReco", tr, xr, xm)
        catch e
            @warn "MRIReco $meth failed" exception = (e, catch_backtrace())
        end
    end
end

write_section("sparsity")
