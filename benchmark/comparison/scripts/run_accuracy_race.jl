# Section: time-to-target-accuracy — the fair cross-toolkit comparison.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_accuracy_race.jl --threads=N [--use-mkl]
#
# ## Why not a fixed iteration count
#
# Every other section pins iterations and compares wall time. That is only meaningful if one
# iteration means the same work everywhere, and it does not:
#
#   * BART's ADMM `-i N` is a budget of ≈N normal-operator applications, not N outer iterations
#     (`src/iter/admm.c:453`), and its inner CG stops at `1e-3 · ‖rhs‖` with `cg_eps` hardcoded at
#     `src/iter/iter.c:106` — **not** settable from the CLI. With a warm start it often takes 0
#     inner iterations, so at ρ = 0.05 `-i 20` buys 18 outer iterations whose x-update never runs
#     and an NRMSE of 0.096, where MRT's 20 genuine iterations reach 0.003. Neither number is
#     wrong; they are answers to different questions.
#   * The four TV functionals differ (MRT isotropic + mirrored boundary, SigPy anisotropic +
#     circular, MRIReco `GradientOp` anisotropic + truncated, BART joint), so no single λ is
#     comparable and each toolkit needs its own.
#   * BART and MRIReco additionally pay one-off Lipschitz estimates on their FISTA paths (`-e` is
#     30 `𝒜ᴴ𝒜`; `power_iterations` is 2–30) that a per-iteration accounting hides.
#
# So: sweep the iteration count per toolkit at its own calibrated λ, and report the wall time at
# the first count that reaches a common NRMSE target. That is the number a user actually cares
# about — "how long until the picture is this good" — and it is invariant to all of the above.
#
# BART's process spawn and cfl I/O are subtracted (`time_bart`), so every figure is solver time.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
using Random: MersenneTwister

# NRMSE targets, chosen so every toolkit can actually reach them (their *converged* values on this
# problem are ≈0.0028 for TV and ≈0.0061 for wavelet, so these leave headroom without being free).
const TGT_TV = parse(Float64, get(ENV, "TGT_TV", "0.005"))
const TGT_WAV = parse(Float64, get(ENV, "TGT_WAV", "0.010"))
# Same rule for the remaining methods: just above the *worst* toolkit's converged NRMSE on the
# calibration grid, so the target is reachable by all of them. Converged values measured by
# `calibrate_lambda.jl` (50 outer iterations, per-toolkit λ):
#   TGV      MRT 0.00416, BART 0.00285
#   lowrank  MRT 0.07935, MRIReco 0.07933, BART 0.08558
#   LLR      MRT 0.04184, MRIReco 0.04259, BART 0.04841
#   tTV      MRT 0.07499, BART 0.08253
const TGT_TGV = parse(Float64, get(ENV, "TGT_TGV", "0.005"))
const TGT_LR = parse(Float64, get(ENV, "TGT_LR", "0.090"))
const TGT_LLR = parse(Float64, get(ENV, "TGT_LLR", "0.055"))
const TGT_TTV = parse(Float64, get(ENV, "TGT_TTV", "0.090"))
# Iteration ladders. BART's ADMM ladder is in `-i` budget units, hence much larger numbers.
const LADDER = [3, 5, 8, 12, 20, 30, 50]
const LADDER_BART_ADMM = [10, 20, 40, 80, 150, 300]

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
ksp_z = copy(kspace_mc); ksp_z[.!mask_reg, :] .= 0
kbart = reshape(ksp_z, N, N, 1, Nc)
sbart = reshape(cmap, N, N, 1, Nc)

"""
    race(label, ladder, target, run) -> (iterations, time_ms, nrmse) or nothing

Walk `ladder`, stopping at the first entry whose reconstruction reaches `target`. `run(it)` must
return `(time_seconds, image)`. Returns `nothing` when the target is never reached, which is itself
a result worth reporting — it means the toolkit cannot get there at this λ.
"""
function race(label, ladder, target, run)
    for it in ladder
        t, x = try
            run(it)
        catch e
            @warn "$label failed" it exception = (e, catch_backtrace())
            continue
        end
        err = mag_nrmse(x, img_mc)
        @info @sprintf("%-22s it=%4d  %9.1f ms  NRMSE=%.5f%s", label, it, t * 1000, err,
            err <= target ? "  <= target" : "")
        err <= target && return (it, t * 1000, err)
    end
    @warn "$label never reached target" target
    return nothing
end

addrow(meth, target, fw, r) = r === nothing ? nothing :
    push!(results, BenchResult("Accuracy race", "$meth (NRMSE≤$target, $(r[1]) it)", fw, NUM_THREADS, r[2], r[3], 0.0))

for (key, meth, target, mrtreg, mrtkind, spm, mrm, bartcmd, bartladder, λdef) in (
        (
            :tv, "Total Variation", TGT_TV, λ -> TotalVariation2D(λ), :admm, :tv, :tv,
            (λ, i) -> "pics -S -w 1 -F -i $i -u $CMP_RHO -C $CMP_CG_ITERS -R T:3:0:$λ",
            LADDER_BART_ADMM, 0.01,
        ),
        (
            :wavelet, "L1-Wavelet", TGT_WAV, mrt_wavelet, :fista, :wavelet, :wavelet,
            (λ, i) -> "pics -S -w 1 -e -i $i -R W:3:0:$λ",
            LADDER, 0.005,
        ),
        (
            :tgv, "TGV", TGT_TGV, λ -> TotalGeneralizedVariation2D(λ; ratio = 2.0), :admm, nothing, nothing,
            (λ, i) -> "pics -S -w 1 -F -i $i -u $CMP_RHO -C $CMP_CG_ITERS -R G:3:0:$λ",
            LADDER_BART_ADMM, 0.003,
        ),
    )
    println("--> $meth  (target NRMSE ≤ $target)")

    addrow(meth, target, FW, race("MRT $meth", LADDER, target, it -> begin
        t, _, x = time_reconstruction(() -> mrt_run(acq_us, mrtreg(load_lambda(key, "MRT", λdef)); maxit = it, kind = mrtkind))
        (t, x)
    end))

    addrow(meth, target, BART_FW, race("BART $meth", bartladder, target, it -> begin
        t, _, r = time_bart(bartcmd(load_lambda(key, "BART", λdef), it), ComplexF32.(kbart), ComplexF32.(sbart))
        (t, r[:, :, 1])
    end))

    spm === nothing || addrow(meth, target, "SigPy", race("SigPy $meth", LADDER, target, it -> begin
        ms, x = sigpy_recon(spm, ksp_z, cmap; λ = load_lambda(key, "SigPy", λdef), iterations = it)
        (ms / 1000, x)
    end))

    mrm === nothing || addrow(meth, target, "MRIReco", race("MRIReco $meth", LADDER, target, it -> begin
        ms, x = mrireco(mrm, ksp_z, cmap, (N, N); λ = load_lambda(key, "MRIReco", λdef), iterations = it)
        (ms / 1000, x)
    end))
end

# --- dynamic (2D+t): global low-rank / locally low-rank / temporal TV ---------------------------
# Same phantom, mask and BART flags as `run_dynamic.jl` (see its header for why `-m -n -b` are all
# required); MRIReco reaches the two low-rank rows through `mrireco_dynamic` and cannot express
# temporal TV.
Nd, Ncd, Td = 64, 4, 8
img_dyn, kspace_dyn0, cmap_dyn = generate_dynamic_multicoil_brain(N = Nd, num_coils = Ncd, num_frames = Td)
kspace_dyn = add_noise(norm_ksp(kspace_dyn0); snr_db = CMP_SNR_DB)
mask_pe = rand(MersenneTwister(42), Bool, Nd)
mask_pe[(Nd ÷ 2 - 4):(Nd ÷ 2 + 4)] .= true
acq_dyn = CartesianAcquisitionInfo(
    NamedDimsArray(permutedims(kspace_dyn[:, mask_pe, :, :], (1, 2, 4, 3)), (:kx, :ky, :coil, :time));
    is3D = false, image_size = (Nd, Nd), sensitivity_maps = NamedDimsArray(cmap_dyn, (:x, :y, :coil)),
    subsampling = (:, mask_pe), shifted_image_dims = (:x, :y),
)
kbart_dyn = zeros(ComplexF32, Nd, Nd, 1, Ncd, 1, Td)
ksp_dyn_z = zeros(ComplexF64, Nd, Nd, Td, Ncd)
for t in 1:Td
    kbart_dyn[:, mask_pe, 1, :, 1, t] .= ComplexF32.(kspace_dyn[:, mask_pe, t, :])
    ksp_dyn_z[:, mask_pe, t, :] .= kspace_dyn[:, mask_pe, t, :]
end
sbart_dyn = reshape(ComplexF32.(cmap_dyn), Nd, Nd, 1, Ncd)
err_dyn(x) = mag_nrmse(x, img_dyn)

# `race` scores against the static phantom, so the dynamic rows get their own copy with `err_dyn`.
function race_dyn(label, ladder, target, run)
    for it in ladder
        t, x = try
            run(it)
        catch e
            @warn "$label failed" it exception = (e, catch_backtrace())
            continue
        end
        err = err_dyn(x)
        @info @sprintf("%-22s it=%4d  %9.1f ms  NRMSE=%.5f%s", label, it, t * 1000, err,
            err <= target ? "  <= target" : "")
        err <= target && return (it, t * 1000, err)
    end
    @warn "$label never reached target" target
    return nothing
end

for (key, meth, target, mrtreg, bartcmd, mrm, λdef) in (
        (
            :lowrank, "Global Low-Rank", TGT_LR, λ -> LowRank(λ; time_dim = :time),
            (λ, i) -> "pics -S -w 1 -m -F -n -i $i -u $CMP_RHO -C $CMP_CG_ITERS -b $Nd -R L:3:3:$λ",
            :lowrank, 0.01,
        ),
        (
            :llr, "Locally Low-Rank", TGT_LLR, λ -> LocallyLowRank(λ; block_size = (8, 8), time_dim = :time),
            (λ, i) -> "pics -S -w 1 -m -F -n -i $i -u $CMP_RHO -C $CMP_CG_ITERS -b 8 -R L:3:3:$λ",
            :llr, 0.01,
        ),
        (
            :ttv, "Temporal TV", TGT_TTV, λ -> TemporalTotalVariation(λ; time_dim = :time),
            (λ, i) -> "pics -S -w 1 -F -i $i -u $CMP_RHO -C $CMP_CG_ITERS -R T:32:0:$λ",
            nothing, 0.01,
        ),
    )
    println("--> $meth  (target NRMSE ≤ $target)")

    addrow(meth, target, FW, race_dyn("MRT $meth", LADDER, target, it -> begin
        t, _, x = time_reconstruction(() -> mrt_run(acq_dyn, mrtreg(load_lambda(key, "MRT", λdef)); maxit = it))
        (t, x)
    end))

    addrow(meth, target, BART_FW, race_dyn("BART $meth", LADDER_BART_ADMM, target, it -> begin
        t, _, r = time_bart(bartcmd(load_lambda(key, "BART", λdef), it), kbart_dyn, sbart_dyn)
        (t, dropdims(r, dims = (3, 4, 5)))
    end))

    mrm === nothing || addrow(meth, target, "MRIReco", race_dyn("MRIReco $meth", LADDER, target, it -> begin
        ms, x = mrireco_dynamic(mrm, ksp_dyn_z, cmap_dyn, (Nd, Nd); λ = load_lambda(key, "MRIReco", λdef), iterations = it)
        (ms / 1000, x)
    end))
end

write_section("accuracy_race")
