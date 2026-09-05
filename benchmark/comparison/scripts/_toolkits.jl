# Cross-toolkit reconstruction helpers shared by the section scripts. Each returns
# `(time_ms, image::Matrix{ComplexF64})` or throws; callers wrap in try/catch so a toolkit that
# cannot do a given method just drops out of that row rather than failing the section.
#
# Assumes `_setup.jl` has been included (`time_reconstruction`, `sp_app`, `run_bart`, MRIReco).

# MRIReco re-exports RegularizedLeastSquares' regularizers and solvers.
using MRIReco: AcquisitionData, L1Regularization, L2Regularization, TVRegularization,
    NuclearRegularization, LLRRegularization
# `ADMM` / `CGNR` / `FISTA` are exported by both MRIReco and MriReconstructionToolbox — always qualify.
const MR_ADMM = MRIReco.ADMM
const MR_CGNR = MRIReco.CGNR
const MR_FISTA = MRIReco.FISTA
const RLS = MRIReco.RegularizedLeastSquares

# --- matched problem conventions --------------------------------------------------------------
# `λ` is only comparable across toolkits once the *functional* each one minimises is the same.
# Two conventions have to be forced explicitly, both measured to matter:
#
#   * **Wavelet basis and depth.** Defaults disagree: MRT `WT.db2` at 2 levels, MRIReco `WT.db2`
#     at full depth, SigPy `db4`, BART `WAVELET_DAU2` at full depth. Aligning SigPy to `db2` moved
#     its converged NRMSE from 0.0083 to 0.0062, and putting MRT on full depth moved it from
#     0.0059 to 0.0061 — after which MRT / MRIReco / SigPy agree to 1.5%.
#   * **TV splitting** — see `mrireco`.
const CMP_WAVELET_LEVELS = parse(Int, get(ENV, "CMP_WAVELET_LEVELS", "3"))
const CMP_WAVELET_NAME = get(ENV, "CMP_WAVELET_NAME", "db2")
mrt_wavelet(λ) = L1Wavelet2D(λ; wavelet = MriReconstructionToolbox.WT.db2, levels = CMP_WAVELET_LEVELS)

# --- shared knobs ---------------------------------------------------------------------------
# Fixed ADMM penalty used by every toolkit's ADMM path, so ρ is not a hidden degree of freedom.
const CMP_RHO = 5.0e-2
# Outer iterations are capped at CMP_OUTER (20 is plenty for these 2D problems); inner CG at
# CMP_CG_ITERS (10). MRT, MRIReco and SigPy all run the full budget — `tol = 0` genuinely means
# "no early stop" in each (verified: MRT `cg.jl:349` `sqrt(r²) <= tol`, MRIReco `cg.jl:140`
# `tolerance = max(reltol*r₀, abstol)`, SigPy `alg.py:284` `resid <= tol`), so their effort is
# exactly `CMP_OUTER × (CMP_CG_ITERS + 1)` normal-operator applications. BART cannot be held to
# the same shape — its inner-CG tolerance is hardcoded and not CLI-settable — so it gets an
# equivalent *budget* instead; see `BART_BUDGET`.
const CMP_OUTER = parse(Int, get(ENV, "CMP_OUTER", "20"))
const CMP_CG_ITERS = parse(Int, get(ENV, "CMP_CG_ITERS", "10"))
const CMP_TOL_INNER = 0.0    # inner CG runs its full CMP_CG_ITERS budget, no early stop

"""
    CMP_FISTA_RHO_MRIRECO

Gradient step size for MRIReco's **FISTA** path (`rho` in `RegularizedLeastSquares.FISTA`, which is
the proximal-gradient step, *not* an ADMM penalty — the two share a keyword name).

`RegularizedLeastSquares.FISTA`'s own constructor default is `0.95 / power_iterations(AHA)`, and
that is *correct*: measured on the 128²×8 phantom, `λ_max(𝒜ᴴ𝒜) = 2.356`, so its intended step is
**0.403**.

It never gets used through the `reconstruction` wrapper. `defaultRecoParams()`
(`MRIReco/src/Reconstruction/RecoParameters.jl:9`) sets `params[:rho] = 5e-2` — an ADMM penalty —
and `reconstruction` merges those defaults into the params of *every* solver
(`Reconstruction.jl:35`), so `createLinearSolver` hands FISTA `rho = 0.05` and the constructor
default is shadowed. That is 8× below the intended step, and every user of the `reconstruction` API
with `solver = FISTA` hits it unless they override `:rho` themselves.

Measured at MRIReco's own optimal λ, 20 iterations, same objective throughout (`prox!` thresholds at
`rho·λ`, so holding λ fixed holds the problem fixed):

| `rho` | NRMSE @ 20 it |
|---|---|
| 0.05 (what the wrapper injects) | 0.0764 |
| 0.1 | 0.0391 |
| 0.2 | 0.0107 |
| **0.4** (≈ FISTA's own default) | **0.0067** |
| 0.6 | 0.0793 (unstable) |
| 0.8 | 0.5222 (diverged) |

The stability limit `2/L = 0.85` and the observed blow-up between 0.4 and 0.6 agree with
`λ_max = 2.356` once FISTA's momentum is accounted for. At 0.4 MRIReco tracks MRT's NRMSE trajectory
iteration for iteration (8 it: 0.0478 vs 0.0486; 16 it: 0.0070 vs 0.0073; 20 it: 0.0067 vs 0.0065)
and reaches the same floor (0.00655 vs 0.00612) — its apparent "slow convergence" was entirely this
one injected parameter, and must not be reported as a property of the algorithm.

So this knob exists only to undo the wrapper's default; the value to set is whatever
`0.95 / power_iterations(AHA)` gives for the data at hand, which is data-scale dependent. The
estimator itself is not the problem and needs no tuning: `power_iterations`' hardcoded `rtol = 1e-3`
returns 2.320 against a converged 2.356 (1.5% low), and even `rtol = 0.1` returns 2.00. Its `rtol` /
`maxiter` are not reachable from `FISTA`'s kwargs anyway — the constructor forwards only `verbose`.

MRT needs no equivalent knob: it normalizes the encoding operator to unit spectral norm and takes
`γ = 1/Lf` exactly.
"""
const CMP_FISTA_RHO_MRIRECO = parse(Float64, get(ENV, "CMP_FISTA_RHO_MRIRECO", "0.4"))

"""
    BART_BUDGET

What to pass as BART's `-i` for an **ADMM** recon (`-R T` / `-R G` / `-R L`).

`-i N` is *not* the outer iteration count: `src/iter/admm.c:453` breaks on
`nr_invokes > maxiter`, where `nr_invokes` accumulates ≈ (outer iterations + cumulative inner CG
iterations). So `-i N` is a budget of roughly N applications of the normal operator. Measured on
the 128²×8 phantom at ρ = 0.05: `-i 20` → 3 outer / 21 CG, `-i 80` → 29 outer / 52 CG, `-i 150` →
99 outer / 52 CG.

Matching `CMP_OUTER` outer iterations of `CMP_CG_ITERS` inner CG each therefore needs `-i` ≈ their
product. Note BART will usually *not* spend it as we would: its inner CG stops at
`1e-3 · ‖rhs‖` (`admm.c:143`, `cg_eps` — hardcoded at `iter.c:106`, **not** CLI-settable), so with
a good warm start it takes far fewer inner iterations and far more outer ones than MRT does. That
asymmetry is the reason `run_accuracy_race.jl` exists: a fixed budget cannot be made to mean the
same thing in both, but "wall time to a given NRMSE" can.

FISTA (`-R W`) is different — no inner CG, and `iter_fista_defaults.tol = 0` is never overridden,
so there `-i` *is* the iteration count and `CMP_OUTER` is passed directly.
"""
const BART_BUDGET = parse(Int, get(ENV, "BART_BUDGET", string(CMP_OUTER * CMP_CG_ITERS)))

"""
    norm_ksp(k) -> k scaled to unit RMS (‖k‖ = √length)

Every toolkit's λ is defined relative to the data-term scale, so a λ calibrated on the synthetic
phantom only transfers to real scanner data if both k-spaces are put on the same scale first.
Applied to every dataset before reconstruction; `mag_nrmse` is scale-invariant so references are
unaffected.
"""
norm_ksp(k) = k .* (sqrt(length(k)) / LinearAlgebra.norm(k))

"""
    add_noise(k; snr_db = 30, seed = 1) -> k + complex Gaussian noise

Additive complex white noise at `snr_db` relative to the RMS of `k`. The synthetic phantom is a
noiseless analytical Shepp–Logan, so without this every toolkit reconstructs it near-perfectly
and TV / wavelet regularisation only ever hurts — there is no non-trivial optimal λ to calibrate.
Real scanner data carries noise, so a λ calibrated on a noisy synthetic problem is the one that
transfers. Fixed `seed` so the calibration and the timing runs see the same realisation.
"""
function add_noise(k; snr_db::Real = 30, seed::Integer = 1)
    rng = Random.MersenneTwister(seed)
    rms = LinearAlgebra.norm(k) / sqrt(length(k))
    σ = rms * 10^(-snr_db / 20) / sqrt(2)
    return k .+ σ .* (randn(rng, ComplexF64, size(k)))
end
const CMP_SNR_DB = parse(Float64, get(ENV, "CMP_SNR_DB", "30"))

"""
    load_lambda(method::Symbol, toolbox::AbstractString, default::Real) -> Float64

Per-toolbox regularisation weight from `benchmark/comparison/results/lambda_calibration.json` (written by
`calibrate_lambda.jl`), falling back to `default` when the file or the entry is missing. The
calibration picks, per method, a λ for each toolbox that lands on a common NRMSE, so the timing
comparison is done at matched accuracy rather than matched λ.
"""
function load_lambda(method::Symbol, toolbox::AbstractString, default::Real)
    f = normpath(joinpath(@__DIR__, "..", "results", "lambda_calibration.json"))
    isfile(f) || return Float64(default)
    tbl = try
        JSON.parsefile(f)["lambda"]
    catch
        return Float64(default)
    end
    m = get(tbl, String(method), nothing)
    (m === nothing || !haskey(m, toolbox)) && return Float64(default)
    v = m[toolbox]
    return (v isa Real && isfinite(v)) ? Float64(v) : Float64(default)
end

# --- MRT --------------------------------------------------------------------------------------
"""
    mrt_admm(reg; rho = CMP_RHO, maxit) -> IterativeReconstruction

MRT reconstruction with a **fixed**-ρ ADMM (default is the adaptive
`SpectralRadiusApproximationPenalty`), a tight inner CG and no early stop, so it matches the
fixed-ρ ADMM the other toolkits are forced onto.
"""
function _mrt_alg(kind::Symbol, maxit::Int, rho::Real)
    if kind === :admm
        return MriReconstructionToolbox.ADMM(;
            rho = rho, maxit = maxit, tol = 0.0,
            cg_tol = CMP_TOL_INNER, cg_maxit = CMP_CG_ITERS
        )
    elseif kind === :fista
        return MriReconstructionToolbox.FISTA(; maxit = maxit, tol = 0.0)
    end
    error("unknown MRT algorithm $kind")
end

"""
    mrt_run(acq, reg; maxit, kind = :admm, rho = CMP_RHO) -> image

`reconstruct` with a fixed-ρ ADMM (`kind = :admm`, for TV / TGV / low-rank) or FISTA
(`kind = :fista`, for L1-wavelet — forcing wavelet through ADMM with a fixed ρ wrecks it).
`maxit` and `tol = 0` are set on `IterativeReconstruction` as well as on the algorithm object: the
method's own values win over the algorithm's, so both must agree to actually run the full count
with no early stop.
"""
mrt_run(acq, reg; maxit::Int, kind::Symbol = :admm, rho::Real = CMP_RHO) =
    reconstruct(acq, IterativeReconstruction(regularization = reg, algorithm = _mrt_alg(kind, maxit, rho); maxit = maxit, tol = 0.0); verbosity = Silent())

# --- MRIReco (Julia) ------------------------------------------------------------------------
# `MRIBase` accepts a 6D `(x, y, z, channel, echo, rep)` k-space array directly (`enc2D` for a
# 2D encode); unsampled entries must be zero. `ksp3` is the Julia `(nx, ny, coil)` layout.
_mrireco_acq(ksp3) = AcquisitionData(reshape(ComplexF64.(ksp3), size(ksp3, 1), size(ksp3, 2), 1, size(ksp3, 3), 1, 1); enc2D = true)

"""
    mrireco(method, ksp3, smaps3, reconSize; λ, iterations, ρ) -> (time_ms, image)

`method ∈ (:cgsense, :tv, :wavelet, :nuclear, :llr)`. TGV / temporal-TV are unsupported (throw).
Runs with `vary_rho = :none`, `iterationsCG = CMP_CG_ITERS` and zero tolerances so the full
iteration budget is spent (`RegularizedLeastSquares.filterKwargs` drops the keys a given solver
does not accept, so the same kwargs are safe for CGNR / ADMM / FISTA).
"""
function mrireco(
        method::Symbol, ksp3, smaps3, reconSize; λ = 0.0, iterations = 10,
        ρ = method === :wavelet ? CMP_FISTA_RHO_MRIRECO : CMP_RHO
    )
    # `regTrafo` stays `opEye` for everything except TV — see the TV branch.
    reg, solver, sparse, regTrafo = if method === :cgsense
        (L2Regularization(0.0), MR_CGNR, nothing, nothing)
    elseif method === :tv
        # RegularizedLeastSquares' own `ADMM` docstring (`src/ADMM.jl:74`) is explicit: "for a TV
        # penalty, you should NOT set `reg=TVRegularization`, but instead use
        # `reg=L1Regularization(λ), regTrafo=GradientOp(...)`". Passing `TVRegularization` as `reg`
        # leaves `regTrafo = opEye`, so the gradient never enters the ADMM splitting and the prox is
        # instead a nested 10-iteration fast-gradient-projection dual solve (`ProxTV.jl:39`; its
        # docstring wrongly says 20). That prox is inexact, which both slows convergence and caps
        # the reachable accuracy: it plateaus at NRMSE 0.0043 where the documented splitting reaches
        # 0.0027. The documented form is also the same splitting MRT, BART and SigPy use, so this is
        # what makes the comparison apples-to-apples.
        (
            L1Regularization(λ), MR_ADMM, nothing,
            RLS.GradientOp(ComplexF64; shape = reconSize, dims = 1:length(reconSize)),
        )
    elseif method === :wavelet
        # `rho` is FISTA's step size here, not a penalty — see `CMP_FISTA_RHO_MRIRECO`.
        (L1Regularization(λ), MR_FISTA, "Wavelet", nothing)
    elseif method === :nuclear
        (NuclearRegularization(λ), MR_ADMM, nothing, nothing)
    elseif method === :llr
        (LLRRegularization(λ; shape = reconSize, blockSize = (8, 8)), MR_ADMM, nothing, nothing)
    else
        error("MRIReco has no $method")
    end
    senseMaps = reshape(ComplexF64.(smaps3), reconSize..., 1, size(smaps3, 3))
    t, _, img = time_reconstruction() do
        rp = Dict{Symbol, Any}(
            :reco => "multiCoil", :reconSize => reconSize, :senseMaps => senseMaps,
            :solver => solver, :reg => reg, :iterations => iterations, :rho => ρ,
            :vary_rho => :none, :iterationsCG => CMP_CG_ITERS,
            :absTol => 0.0, :relTol => 0.0, :tolInner => CMP_TOL_INNER,
        )
        sparse !== nothing && (rp[:sparseTrafo] = sparse)
        regTrafo !== nothing && (rp[:regTrafo] = regTrafo)
        MRIReco.reconstruction(_mrireco_acq(ksp3), rp)[:, :, 1, 1, 1]
    end
    return t * 1000, Array{ComplexF64}(img)
end

"""
    mrireco_dynamic(method, ksp4, smaps3, reconSize; λ, iterations) -> (time_ms, image)

Dynamic (2D+t) reconstruction with MRIReco. `method ∈ (:lowrank, :llr)`; `ksp4` is the zero-filled
`(nx, ny, time, coil)` k-space.

The frames are handed to MRIReco as **contrasts (echoes)**, not repetitions, and the solve goes
through `reco = "multiCoilMultiEcho"` — `reconstruction_multiCoil` loops over repetitions and slices
and solves each independently (`IterativeReconstruction.jl:54`), which would decouple the frames and
make a temporal prior meaningless, while `reconstruction_multiCoilMultiEcho` builds one system over
all contrasts (`:273`) and applies the regularizer to the stacked volume. That is what MRT's
`LowRank` / `LocallyLowRank` do, so the two are comparable.

`RegularizedLeastSquares` needs the volume shape spelled out, since the prox reshapes a flat vector:
`NuclearRegularization` takes `svtShape = (prod(reconSize), n_frames)` — the Casorati matrix, i.e.
MRT's global `LowRank` — and `LLRRegularization` takes `shape` and a `blockSize` of the *same* arity
(`(8, 8, n_frames)`, i.e. blocks that span the whole time axis, matching MRT's `block_size = (8, 8)`
with all frames in the Casorati direction).

**Temporal TV is not reachable through this API and therefore has no MRIReco row.** Not for lack of
a prox — `L1Regularization` + a `GradientOp` along the time axis is the right formulation, and it is
what MRIReco's own TV path uses spatially — but `reconstruction_multiCoilMultiEcho` wraps whatever
`regTrafo` it is given in `DiagOp(repeat([trafo], numContr)...)` (`IterativeReconstruction.jl:304`),
i.e. it applies the transform *per contrast*. A per-frame block-diagonal transform cannot couple
frames, so a time-difference operator over the `(nx, ny, n_frames)` volume simply does not fit and
throws `LinearOperatorException("shape mismatch")`.
"""
function mrireco_dynamic(
        method::Symbol, ksp4, smaps3, reconSize; λ = 0.0, iterations = 10, ρ = CMP_RHO
    )
    nx, ny, nt, ncoil = size(ksp4)
    reg = if method === :lowrank
        RLS.NuclearRegularization(λ; svtShape = (prod(reconSize), nt))
    elseif method === :llr
        RLS.LLRRegularization(λ; shape = (reconSize..., nt), blockSize = (8, 8, nt))
    else
        error("MRIReco has no dynamic $method here (see the docstring on temporal TV)")
    end
    ksp6 = zeros(ComplexF64, nx, ny, 1, ncoil, nt, 1)
    for t in 1:nt
        ksp6[:, :, 1, :, t, 1] .= ComplexF64.(@view ksp4[:, :, t, :])
    end
    acq = AcquisitionData(ksp6; enc2D = true)
    senseMaps = reshape(ComplexF64.(smaps3), reconSize..., 1, ncoil)
    t, _, img = time_reconstruction() do
        rp = Dict{Symbol, Any}(
            :reco => "multiCoilMultiEcho", :reconSize => reconSize, :senseMaps => senseMaps,
            :solver => MR_ADMM, :reg => reg, :iterations => iterations, :rho => ρ,
            :vary_rho => :none, :iterationsCG => CMP_CG_ITERS,
            :absTol => 0.0, :relTol => 0.0, :tolInner => CMP_TOL_INNER,
        )
        MRIReco.reconstruction(acq, rp)
    end
    return t * 1000, reshape(Array{ComplexF64}(img), reconSize..., nt)
end

# --- SigPy (Python) ------------------------------------------------------------------------
# SigPy wants k-space `(coil, ky, kx)` and maps `(coil, y, x)`, returns `(y, x)`.
_sp_k(ksp3) = parent(permutedims(ComplexF64.(ksp3), (3, 2, 1)))
_sp_s(smaps3) = parent(permutedims(ComplexF64.(smaps3), (3, 2, 1)))

"""
    sigpy_recon(method, ksp3, smaps3; λ, iterations) -> (time_ms, image)

`method ∈ (:cgsense, :tv, :wavelet)`. TGV / low-rank unsupported here (throw).
Only **TV** is forced onto ADMM (`rho = CMP_RHO`, `max_cg_iter = CMP_CG_ITERS`) — matching the
ADMM the other toolkits use for TV; SigPy would otherwise default to PDHG. **L1-wavelet** keeps
SigPy's natural proximal-gradient solver (FISTA-like), as MRT / MRIReco / BART also use FISTA for
wavelet. `tol ≈ 0` so all `iterations` outer steps run.
"""
function sigpy_recon(method::Symbol, ksp3, smaps3; λ = 0.0, iterations = 10)
    y, mps = _sp_k(ksp3), _sp_s(smaps3)
    app = if method === :cgsense
        () -> sp_app.SenseRecon(y, mps; max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false).run()
    elseif method === :tv
        () -> sp_app.TotalVariationRecon(
            y, mps, λ; solver = "ADMM", rho = CMP_RHO,
            max_cg_iter = CMP_CG_ITERS, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false
        ).run()
    elseif method === :wavelet
        () -> sp_app.L1WaveletRecon(y, mps, λ; wave_name = CMP_WAVELET_NAME, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false).run()
    else
        error("SigPy has no $method here")
    end
    t, _, raw = time_reconstruction(app)
    return t * 1000, Array{ComplexF64}(permutedims(raw, (2, 1)))
end
