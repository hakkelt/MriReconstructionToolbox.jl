# Per-toolbox λ calibration for the matched-accuracy comparison.
#
# For each method (TV and L1-wavelet on the static phantom; global low-rank, locally low-rank and
# temporal TV on the 2D+t one) and each toolbox that implements it we sweep λ over a
# wide log grid, run the solver to (near) full convergence, and record NRMSE vs the synthetic
# ground truth. BART's regularisation weight is on a different internal scale than the others
# (it rescales the data internally), so a common λ is meaningless; instead the target NRMSE is
# what MRT reaches at its own best λ, and every other toolbox's λ is the grid point whose
# converged NRMSE is closest to that. `run_sparsity.jl` / `run_real.jl` read these λ back through
# `load_lambda` (k-space is unit-RMS normalised everywhere so the λ transfers), so the timing
# comparison is at matched accuracy rather than matched λ.
#
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/calibrate_lambda.jl --threads=N [--use-mkl]
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
using Random: MersenneTwister

const IT_CAL = parse(Int, get(ENV, "IT_CAL", "30"))   # above production's 20 — converged NRMSE(λ)
const NGRID = parse(Int, get(ENV, "NGRID", "8"))

img_mc, cmap = IMG_MC, CMAP
kspace_mc = add_noise(norm_ksp(KSPACE_MC); snr_db = CMP_SNR_DB)   # unit-RMS + noise so λ > 0 is optimal
smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))
mask_reg = rand(MersenneTwister(42), Bool, N, N)
mask_reg[(N ÷ 2 - 8):(N ÷ 2 + 8), :] .= true
acq_us = CartesianAcquisitionInfo(
    NamedDimsArray(kspace_mc[mask_reg, :], (:kxy, :coil));
    is3D = false, image_size = (N, N), sensitivity_maps = smaps_mc,
    shifted_image_dims = (:x, :y), subsampling = mask_reg,
)
ksp_z = copy(kspace_mc); ksp_z[.!mask_reg, :] .= 0
kbart = reshape(ComplexF32.(ksp_z), N, N, 1, Nc)
sbart = reshape(ComplexF32.(cmap), N, N, 1, Nc)

err(x) = mag_nrmse(x, img_mc)

# method => (MRT reg builder, MRT alg kind, sigpy sym, mrireco sym, BART cmd builder, λ centre)
METHODS = Dict(
    "tv" => (λ -> TotalVariation2D(λ), :admm, :tv, :tv,
        λ -> "pics -S -w 1 -F -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -R T:3:0:$λ", 0.01),
    "wavelet" => (mrt_wavelet, :fista, :wavelet, :wavelet,
        λ -> "pics -S -w 1 -e -i $IT_CAL -R W:3:0:$λ", 0.005),
    # TGV: MRT and BART only (SigPy and MRIReco have no TGV). Without this entry the section ran
    # both at the 0.01 fallback, which happens to be near BART's optimum and 5× past MRT's — MRT
    # measured NRMSE 0.0109 at λ=0.01 against 0.0032 at λ=0.002.
    "tgv" => (λ -> TotalGeneralizedVariation2D(λ; ratio = 2.0), :admm, nothing, nothing,
        λ -> "pics -S -w 1 -F -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -R G:3:0:$λ", 0.003),
)

# --- dynamic (2D+t) methods -------------------------------------------------------------------
# Same 64²×4-coil×8-frame phantom and mask as `run_dynamic.jl`, so the λ transfers. MRT and BART
# cover all three methods; MRIReco covers the two low-rank ones through `mrireco_dynamic` (frames as
# contrasts, `reco = "multiCoilMultiEcho"`) but cannot express temporal TV — see that function's
# docstring. SigPy ships no low-rank MRI app at all. The target NRMSE is still MRT's own best, and
# every other toolkit's λ is the grid point that matches it.
Nd, Ncd, Td = 64, 4, 8
img_dyn, kspace_dyn0, cmap_dyn = generate_dynamic_multicoil_brain(N = Nd, num_coils = Ncd, num_frames = Td)
# Noise is what makes λ > 0 optimal at all. On the noiseless dynamic phantom every regularizer is
# pure bias: measured NRMSE decreases monotonically as λ → 0 (LowRank 0.0810 at λ=1e-4 vs 0.0876 at
# λ=0.2, 20 ADMM iterations) and plain CG-SENSE beats all of them, so there is no operating point to
# calibrate. With CMP_SNR_DB the optimum is interior — LLR 0.0408 at λ=0.02 against CG-SENSE's
# 0.0990 — matching how the static sections are set up.
kspace_dyn = add_noise(norm_ksp(kspace_dyn0); snr_db = CMP_SNR_DB)
mask_pe = rand(MersenneTwister(42), Bool, Nd)
mask_pe[(Nd ÷ 2 - 4):(Nd ÷ 2 + 4)] .= true
acq_dyn = CartesianAcquisitionInfo(
    NamedDimsArray(permutedims(kspace_dyn[:, mask_pe, :, :], (1, 2, 4, 3)), (:kx, :ky, :coil, :time));
    is3D = false, image_size = (Nd, Nd), sensitivity_maps = NamedDimsArray(cmap_dyn, (:x, :y, :coil)),
    subsampling = (:, mask_pe), shifted_image_dims = (:x, :y),
)
kbart_dyn = zeros(ComplexF32, Nd, Nd, 1, Ncd, 1, Td)
for t in 1:Td
    kbart_dyn[:, mask_pe, 1, :, 1, t] .= ComplexF32.(kspace_dyn[:, mask_pe, t, :])
end
sbart_dyn = reshape(ComplexF32.(cmap_dyn), Nd, Nd, 1, Ncd)
err_dyn(x) = mag_nrmse(x, img_dyn)

# BART's `-R L` flags, all three verified by measurement — see `run_dynamic.jl` for the detail:
# `-m` forces ADMM (the low-rank path defaults to FISTA, which is a different algorithm from MRT's
# ADMM and converges far slower here), `-b` sets the block edge (default **8**, so a bare `-R L` is
# LLR, not global low-rank — `-b $Nd` is the single-block/global case), and `-n` switches off random
# cycle spinning to match MRT's default `shift = :none`.
# (MRT reg builder, BART cmd builder, MRIReco method or `nothing`, λ centre)
DYN_METHODS = Dict(
    "lowrank" => (λ -> LowRank(λ; time_dim = :time),
        λ -> "pics -S -w 1 -m -F -n -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -b $Nd -R L:3:3:$λ",
        :lowrank, 0.01),
    "llr" => (λ -> LocallyLowRank(λ; block_size = (8, 8), time_dim = :time),
        λ -> "pics -S -w 1 -m -F -n -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -b 8 -R L:3:3:$λ",
        :llr, 0.01),
    "ttv" => (λ -> TemporalTotalVariation(λ; time_dim = :time),
        λ -> "pics -S -w 1 -F -i $BART_BUDGET -u $CMP_RHO -C $CMP_CG_ITERS -R T:32:0:$λ",
        nothing, 0.01),
)

# Zero-filled `(nx, ny, time, coil)` frame stack for MRIReco.
ksp_dyn_z = zeros(ComplexF64, Nd, Nd, Td, Ncd)
for t in 1:Td
    ksp_dyn_z[:, mask_pe, t, :] .= kspace_dyn[:, mask_pe, t, :]
end

function sweep_dyn(method)
    mrtreg, bartcmd, mrm, λc = DYN_METHODS[method]
    grid = 10 .^ range(log10(λc) - 2, log10(λc) + 1.5, length = NGRID)
    curves = Dict{String, Vector{Tuple{Float64, Float64}}}()
    toolkits = ["MRT", "BART"]
    mrm === nothing || push!(toolkits, "MRIReco")
    for tb in toolkits
        pts = Tuple{Float64, Float64}[]
        for λ in grid
            e = try
                if tb == "MRT"
                    err_dyn(mrt_run(acq_dyn, mrtreg(λ); maxit = IT_CAL))
                elseif tb == "MRIReco"
                    err_dyn(mrireco_dynamic(mrm, ksp_dyn_z, cmap_dyn, (Nd, Nd); λ, iterations = IT_CAL)[2])
                else
                    err_dyn(dropdims(run_bart(1, bartcmd(λ), kbart_dyn, sbart_dyn), dims = (3, 4, 5)))
                end
            catch ex
                @warn "$tb $method λ=$λ failed" ex
                NaN
            end
            @info @sprintf("%-8s %-7s λ=%.4g  NRMSE=%.4f", method, tb, λ, e)
            push!(pts, (λ, e))
        end
        curves[tb] = pts
    end
    return curves
end

function sweep(method)
    mrtreg, mrtkind, spm, mrm, bartcmd, λc = METHODS[method]
    grid = 10 .^ range(log10(λc) - 2, log10(λc) + 1.5, length = NGRID)
    curves = Dict{String, Vector{Tuple{Float64, Float64}}}()
    toolkits = ["MRT", "BART"]
    mrm === nothing || push!(toolkits, "MRIReco")
    spm === nothing || push!(toolkits, "SigPy")
    for tb in toolkits
        pts = Tuple{Float64, Float64}[]
        for λ in grid
            e = try
                if tb == "MRT"
                    err(mrt_run(acq_us, mrtreg(λ); maxit = IT_CAL, kind = mrtkind))
                elseif tb == "MRIReco"
                    err(mrireco(mrm, ksp_z, cmap, (N, N); λ, iterations = IT_CAL)[2])
                elseif tb == "SigPy"
                    err(sigpy_recon(spm, ksp_z, cmap; λ, iterations = IT_CAL)[2])
                else
                    err(run_bart(1, bartcmd(λ), kbart, sbart)[:, :, 1])
                end
            catch ex
                @warn "$tb $method λ=$λ failed" ex
                NaN
            end
            @info @sprintf("%-8s %-7s λ=%.4g  NRMSE=%.4f", method, tb, λ, e)
            push!(pts, (λ, e))
        end
        curves[tb] = pts
    end
    return curves
end

"""Best (lowest) finite NRMSE on a curve."""
best(curve) = minimum(e for (_, e) in curve if isfinite(e); init = Inf)

"""λ on `curve` whose NRMSE is closest to `target`."""
function pick_lambda(curve, target)
    fin = [(λ, e) for (λ, e) in curve if isfinite(e)]
    isempty(fin) && return NaN
    return fin[argmin(abs(e - target) for (_, e) in fin)][1]
end

out = Dict{String, Any}()
sweeps = Dict{String, Any}()
for method in vcat(collect(keys(METHODS)), collect(keys(DYN_METHODS)))
    curves = haskey(METHODS, method) ? sweep(method) : sweep_dyn(method)
    # Target = the NRMSE MRT reaches at its own best λ (MRT is the reference implementation);
    # every other toolbox's λ is then chosen to match MRT's accuracy. If a toolbox cannot reach
    # that NRMSE anywhere on the grid, `pick_lambda` returns its closest (best) point.
    target = best(curves["MRT"])
    picks = Dict(tb => pick_lambda(c, target) for (tb, c) in curves)
    @info "calibrated" method target picks
    out[method] = picks
    sweeps[method] = Dict(tb => [[λ, e] for (λ, e) in c] for (tb, c) in curves)
    out[method]["_target_nrmse"] = target
end

path = normpath(joinpath(@__DIR__, "..", "results", "lambda_calibration.json"))
open(path, "w") do io
    JSON.print(io, Dict("lambda" => out, "sweeps" => sweeps,
            "meta" => Dict("N" => N, "Nc" => Nc, "iterations" => IT_CAL,
                "backend" => USE_MKL ? "mkl" : "openblas", "threads" => NUM_THREADS)), 4)
end
@info "wrote" path
