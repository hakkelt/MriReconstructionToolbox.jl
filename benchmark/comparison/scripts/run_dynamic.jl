# Section: dynamic / low-rank on the synthetic 2D+t brain — MRT vs BART vs MRIReco
# (global low-rank ↔ `-R L -b <N>`, locally low-rank ↔ `-R L -b 8`, temporal TV ↔ `-R T:32`).
# SigPy has no stock low-rank MRI app. MRIReco joins the two low-rank rows via `mrireco_dynamic`
# (frames as contrasts, `reco = "multiCoilMultiEcho"`); it cannot express temporal TV, because that
# path wraps `regTrafo` per contrast and so cannot couple frames — see `mrireco_dynamic`'s
# docstring for the exact line.
#
# Matched effort: 20 outer iterations everywhere (BART temporal-TV was `-i 200 -C 10`), fixed-ρ
# ADMM for MRT, per-toolkit calibrated λ, BART wisdom auto-enabled only above 5 s.
#
# ## Three BART `-R L` flags this section needs, all established by measurement
#
# * **`-b` defaults to 8, so a bare `-R L:3:3:λ` is *locally* low rank with 8×8 blocks.** The
#   earlier "Global Low-Rank" row was therefore the same reconstruction as the LLR row, bit for bit
#   (both NRMSE 0.0803516351), and neither matched MRT's `LowRank`, which is a nuclear norm on the
#   *whole* Casorati matrix. `-b $Nd` makes the image one block, i.e. the global case. Measured at
#   λ=0.01, `-i 20`: `-b 4` 0.0634, `-b 8` (= default) 0.0804, `-b 16` 0.0898, `-b 64` 0.1009.
# * **`-R L` selects FISTA, not ADMM** (`Total Time` banner prints `FISTA`), because the low-rank
#   prox needs no linear transform — the same reason `-R W` does. That is a different algorithm
#   from MRT's ADMM and much slower here (MRT FISTA 0.1026 vs MRT ADMM 0.0811 at 20 iterations, at
#   which point MRT and BART FISTA agree to 4 digits: 0.10262 vs 0.10265). `-m` forces ADMM so both
#   sides run the same algorithm, and `-u` / `-C` become meaningful.
# * **`-n` disables random wavelet/block cycle spinning**, which BART applies to LLR by default.
#   MRT's `LocallyLowRank` defaults to `shift = :none` (a fixed tiling, exact prox), so `-n` is what
#   makes the two the same objective. (`-N`, fully overlapping blocks, is a third variant neither
#   MRT nor this section uses; it costs ~10× more.)
#
# ## Temporal TV is the one method whose accuracy depends on the *inner* solve
#
# Every other regularizer here is insensitive to `cg_maxit` (50 outer iterations, λ at its optimum:
# TV2D 0.00674 at both 10 and 80 inner CG, LowRank 0.07934 vs 0.07931, LLR 0.04184 vs 0.04172).
# Temporal TV is not: 0.0797 (cg 5), 0.0769 (10), 0.0862 (20), 0.1357 (40), 0.2440 (80). Solving the
# x-update *more* exactly makes it worse, monotonically, and at cg = 80 the result stops depending on
# λ at all (identical NRMSE 0.26725 at λ = 1 and λ = 3).
#
# That is not a broken prox — it is fixed-ρ ADMM not converging. `D_t` has a large null space (any
# image constant in time), so at ρ = CMP_RHO = 0.05 the splitting barely couples `z` to `D_t x` and
# 50 iterations are nowhere near the fixed point; an inexact CG damps the x-update and accidentally
# behaves like a larger ρ. Raising ρ instead of crippling the solve fixes it: at cg = 80, ρ = 5.0
# gives 0.0961 against ρ = 0.05's 0.2440. Setting MRT's inner tolerance to BART's own `1e-3` gives
# 0.0935.
#
# **BART hides this**, which is why its temporal-TV row looks stable: its inner CG tolerance is
# hardcoded at `1e-3 · ‖rhs‖` (`src/iter/iter.c:106`) and `-C` only caps the count, so `-C 10` and
# `-C 80` return the same image to 5 digits (0.07587 / 0.07586). At the section's `CMP_CG_ITERS = 10`
# MRT and BART agree closely (λ = 0.03: 0.0769 vs 0.0759; λ = 0.3: 0.1099 vs 0.1037), so that is the
# operating point λ is calibrated at and the one this section measures. See TODO.md for the MRT-side
# follow-up (ρ selection for `TemporalTotalVariation`).
#
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_dynamic.jl --threads=N [--use-mkl]
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
using Random: MersenneTwister

const IT = CMP_OUTER
Nd, Ncd, Td = 64, 4, 8
img_dyn, kspace_dyn0, cmap_dyn = generate_dynamic_multicoil_brain(N = Nd, num_coils = Ncd, num_frames = Td)
# Unit-RMS **and noisy**, consistent with the other sections. Without noise no regularizer helps at
# all on this phantom: NRMSE falls monotonically as λ → 0 and plain CG-SENSE (0.0757) beats every
# regularized run, so the accuracy column carries no information. At CMP_SNR_DB the optimum is
# interior (LLR 0.0408 at λ = 0.02, tTV 0.0748 at 0.005, LowRank 0.0803 at 0.02, CG-SENSE 0.0990).
kspace_dyn = add_noise(norm_ksp(kspace_dyn0); snr_db = CMP_SNR_DB)
mask_pe = rand(MersenneTwister(42), Bool, Nd)
mask_pe[(Nd ÷ 2 - 4):(Nd ÷ 2 + 4)] .= true

acq_dyn = CartesianAcquisitionInfo(
    NamedDimsArray(permutedims(kspace_dyn[:, mask_pe, :, :], (1, 2, 4, 3)), (:kx, :ky, :coil, :time));
    is3D = false, image_size = (Nd, Nd), sensitivity_maps = NamedDimsArray(cmap_dyn, (:x, :y, :coil)),
    subsampling = (:, mask_pe), shifted_image_dims = (:x, :y),
)
# BART: (x, y, 1, coil, 1, time), zero-filled
kbart = zeros(ComplexF32, Nd, Nd, 1, Ncd, 1, Td)
for t in 1:Td
    kbart[:, mask_pe, 1, :, 1, t] .= ComplexF32.(kspace_dyn[:, mask_pe, t, :])
end
# Zero-filled `(nx, ny, time, coil)` stack for MRIReco.
ksp_z = zeros(ComplexF64, Nd, Nd, Td, Ncd)
for t in 1:Td
    ksp_z[:, mask_pe, t, :] .= kspace_dyn[:, mask_pe, t, :]
end
sbart = reshape(ComplexF32.(cmap_dyn), Nd, Nd, 1, Ncd)

magd(x, ref) = (a = abs.(x); r = abs.(ref); nrmse(a .* (norm(r) / norm(a)), r))
addrow(meth, fw, t, x, xmrt) = push!(results, BenchResult("Dynamic", meth, fw, NUM_THREADS, t, magd(x, img_dyn), xmrt === nothing ? 0.0 : magd(xmrt, x)))

# λ comes from `calibrate_lambda.jl`'s dynamic sweeps (`lowrank` / `llr` / `ttv`), one per toolkit.
specs = (
    (:lowrank, "Global Low-Rank ($IT it)", λ -> LowRank(λ; time_dim = :time),
        λ -> "pics -S -w 1 -m -F -n -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -b $Nd -R L:3:3:$λ",
        :lowrank, 0.01),
    (:llr, "Locally Low-Rank ($IT it)", λ -> LocallyLowRank(λ; block_size = (8, 8), time_dim = :time),
        λ -> "pics -S -w 1 -m -F -n -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -b 8 -R L:3:3:$λ",
        :llr, 0.01),
    (:ttv, "Temporal TV ($IT it)", λ -> TemporalTotalVariation(λ; time_dim = :time),
        λ -> "pics -S -w 1 -F -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -R T:32:0:$λ",
        nothing, 0.01),
)

for (key, meth, mrtreg, bartcmd, mrm, λdef) in specs
    println("--> $meth")
    tm, _, xm = time_reconstruction(() -> mrt_run(acq_dyn, mrtreg(load_lambda(key, "MRT", λdef)); maxit = IT))
    addrow(meth, FW, tm * 1000, xm, nothing)

    try
        tb, _, rb = time_bart(bartcmd(load_lambda(key, "BART", λdef)), kbart, sbart)
        addrow(meth, BART_FW, tb * 1000, dropdims(rb, dims = (3, 4, 5)), xm)
    catch e
        @warn "BART $meth failed" exception = (e, catch_backtrace())
    end
    if mrm !== nothing
        try
            tr, xr = mrireco_dynamic(mrm, ksp_z, cmap_dyn, (Nd, Nd);
                λ = load_lambda(key, "MRIReco", λdef), iterations = IT)
            addrow(meth, "MRIReco", tr, xr, xm)
        catch e
            @warn "MRIReco $meth failed" exception = (e, catch_backtrace())
        end
    end
end

write_section("dynamic")
