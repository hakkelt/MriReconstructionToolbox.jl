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
const CMP_WAVELET_LEVELS = BenchUtils.WAVELET_LEVELS
const CMP_WAVELET_NAME = get(ENV, "CMP_WAVELET_NAME", "db2")

# --- shared knobs ---------------------------------------------------------------------------
# The effort knobs are BenchUtils' (benchmark/utils/mrt_methods.jl), so MRT's rows here and the MRT
# harness run the same solve. Fixed ADMM penalty used by every toolkit's ADMM path, so ρ is not a
# hidden degree of freedom.
const CMP_RHO = ADMM_RHO
# Outer iterations are capped at CMP_OUTER (20 is plenty for these 2D problems); inner CG at
# CMP_CG_ITERS (10). MRT, MRIReco and SigPy all run the full budget — `tol = 0` genuinely means
# "no early stop" in each (verified: MRT `cg.jl:349` `sqrt(r²) <= tol`, MRIReco `cg.jl:140`
# `tolerance = max(reltol*r₀, abstol)`, SigPy `alg.py:284` `resid <= tol`), so their effort is
# exactly `CMP_OUTER × (CMP_CG_ITERS + 1)` normal-operator applications. BART cannot be held to
# the same shape — its inner-CG tolerance is hardcoded and not CLI-settable — so it gets an
# equivalent *budget* instead; see `BART_BUDGET`.
const CMP_OUTER = OUTER_ITERATIONS
const CMP_CG_ITERS = CG_ITERATIONS
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

**Which is why this is not what `mrireco` passes.** A hand-supplied `rho` is a free step size: it
takes the result of a power iteration without paying for one, while MRT's timing includes
`estimate_opnorm` on every regularized solve (`solve_core.jl:347`) — on the 128²×8 phantom that is
153 ms against a 300 ms wavelet solve, i.e. half the row. Measured on the same operator,
`power_iterations(AHA)` costs MRIReco **221 ms**. Passing 0.4 therefore hid, and credited to
MRIReco, more time than its entire reported wavelet row. `mrireco` now estimates the step itself,
inside the timed region (`ρ = nothing`); this constant remains only as an `ENV` escape hatch for
pinning the step by hand, and is no longer the default for any row.
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
    proxgrad_budget(outer) -> Int

How many **proximal-gradient** iterations equal `outer` ADMM iterations, in applications of the
normal operator: each ADMM iteration costs `CMP_CG_ITERS` inner CG applications plus the gradient,
so `outer * (CMP_CG_ITERS + 1)`.

MIRT ships no ADMM, so `mirt_lowrank` runs POGM — one normal-operator application per iteration.
Passing it `CMP_OUTER` therefore gave it **a tenth of the work** every other low-rank row spends,
and the row that came back was not a converged solve but an early-stopped one: measured on the
dynamic phantom at 20 iterations it sits at NRMSE 0.109 for every λ from 0.01 to 100, four orders
of magnitude over which nothing moves — the signature of an iterate that has barely left `x0`,
not of an optimum. The same solve at 60 iterations reaches 0.0800, below MRT's 0.0845.

This is the same correction `BART_BUDGET` makes for BART's `-i`, in the same unit. It changes what
λ means for the row, which is why MIRT's grid in `calibrate_lambda.jl` is swept at this budget too
and centred separately (`grid_centre`): early stopping is itself a regularizer, so a solver run to
convergence wants a larger λ than one stopped at 20 iterations.

Raising the budget is also what exposed [`_mirt_lipschitz`](@ref)'s job — a step size 1.24 % too
large survives 20 iterations and diverges over 220 — so the two fixes only make sense together.
"""
proxgrad_budget(outer::Int) = outer * (CMP_CG_ITERS + 1)

# Data scaling and noise (`norm_ksp`, `add_noise`) are applied once, when the catalog builds a case
# (benchmark/utils/noise.jl); λ comes from `load_lambda` in `_methods.jl`; MRT's own solve is
# `mrt_reconstructor` (benchmark/utils/mrt_methods.jl), the call the MRT harness times.

# --- MRIReco (Julia) ------------------------------------------------------------------------
# `MRIBase` accepts a 6D `(x, y, z, channel, echo, rep)` k-space array directly (`enc2D` for a
# 2D encode); unsampled entries must be zero. `ksp3` is the Julia `(nx, ny, coil)` layout.
_mrireco_acq(ksp3) = AcquisitionData(reshape(CMP_CTYPE.(ksp3), size(ksp3, 1), size(ksp3, 2), 1, size(ksp3, 3), 1, 1); enc2D = true)

"""
    _mrireco_normal_operator(acq, senseMaps, reconSize) -> AHA

The normal operator `(W∘E)ᴴ(W∘E)` that `reconstruction_multiCoil` builds internally
(`IterativeReconstruction.jl:238-240`) and hands to `createLinearSolver` as `AHA`.

Rebuilt here for one purpose: so a FISTA row can run the same `power_iterations(AHA)` step-size
estimate `FISTA`'s constructor would run by itself, and be timed for it. Only the power iteration
goes inside the timed region — the operator is built here, outside it, because `reconstruction`
builds its own copy anyway and timing this one would charge MRIReco for the construction twice.
"""
function _mrireco_normal_operator(acq, senseMaps, reconSize)
    E = MRIReco.encodingOps_parallel(acq, reconSize, senseMaps; slice = 1)
    W = MRIReco.WeightingOp(CMP_CTYPE; weights = MRIReco.samplingDensity(acq, reconSize)[1], rep = size(senseMaps, ndims(senseMaps)))
    return MRIReco.normalOperator(∘(W, E[1]))
end

"""
    mrireco(method, ksp3, smaps3, reconSize; λ, iterations, ρ) -> (time_ms, image)

`method ∈ (:cgsense, :tv, :wavelet, :nuclear, :llr)`. TGV / temporal-TV are unsupported (throw).
Runs with `vary_rho = :none`, `iterationsCG = CMP_CG_ITERS` and zero tolerances so the full
iteration budget is spent (`RegularizedLeastSquares.filterKwargs` drops the keys a given solver
does not accept, so the same kwargs are safe for CGNR / ADMM / FISTA).

`ρ = nothing` — the default for `:wavelet`, the one FISTA path — means *estimate the step size*:
`0.95 / power_iterations(AHA)`, FISTA's own constructor default, computed inside the timed region
because that is where MRT's equivalent `estimate_opnorm` is charged. See `CMP_FISTA_RHO_MRIRECO`
for why it is not passed as a constant. For the ADMM rows `ρ` is a penalty, not a step size,
and every toolkit is held to the same fixed `CMP_RHO`, so nothing is estimated there.
"""
function mrireco(
        method::Symbol, ksp3, smaps3, reconSize; λ = 0.0, iterations = 10,
        ρ::Union{Real, Nothing} = method === :wavelet ? nothing : CMP_RHO
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
            RLS.GradientOp(CMP_CTYPE; shape = reconSize, dims = 1:length(reconSize)),
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
    senseMaps = reshape(CMP_CTYPE.(smaps3), reconSize..., 1, size(smaps3, 3))
    # A separate `AcquisitionData` on purpose: the timed closure builds its own (as every other
    # row does), so this one costs the row nothing.
    AHA = ρ === nothing ? _mrireco_normal_operator(_mrireco_acq(ksp3), senseMaps, reconSize) : nothing
    t, _, img = time_reconstruction() do
        rp = Dict{Symbol, Any}(
            :reco => "multiCoil", :reconSize => reconSize, :senseMaps => senseMaps,
            :solver => solver, :reg => reg, :iterations => iterations,
            :rho => AHA === nothing ? ρ : 0.95 / RLS.power_iterations(AHA),
            :vary_rho => :none, :iterationsCG => CMP_CG_ITERS,
            :absTol => 0.0, :relTol => 0.0, :tolInner => CMP_TOL_INNER,
        )
        sparse !== nothing && (rp[:sparseTrafo] = sparse)
        regTrafo !== nothing && (rp[:regTrafo] = regTrafo)
        with_mrireco_blas() do
            MRIReco.reconstruction(_mrireco_acq(ksp3), rp)[:, :, 1, 1, 1]
        end
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
    ksp6 = zeros(CMP_CTYPE, nx, ny, 1, ncoil, nt, 1)
    for t in 1:nt
        ksp6[:, :, 1, :, t, 1] .= CMP_CTYPE.(@view ksp4[:, :, t, :])
    end
    acq = AcquisitionData(ksp6; enc2D = true)
    senseMaps = reshape(CMP_CTYPE.(smaps3), reconSize..., 1, ncoil)
    t, _, img = time_reconstruction() do
        rp = Dict{Symbol, Any}(
            :reco => "multiCoilMultiEcho", :reconSize => reconSize, :senseMaps => senseMaps,
            :solver => MR_ADMM, :reg => reg, :iterations => iterations, :rho => ρ,
            :vary_rho => :none, :iterationsCG => CMP_CG_ITERS,
            :absTol => 0.0, :relTol => 0.0, :tolInner => CMP_TOL_INNER,
        )
        with_mrireco_blas() do
            MRIReco.reconstruction(acq, rp)
        end
    end
    return t * 1000, reshape(Array{ComplexF64}(img), reconSize..., nt)
end

# --- SigPy (Python) ------------------------------------------------------------------------
# SigPy wants k-space `(coil, [kz,] ky, kx)` and maps `(coil, [z,] y, x)`, returns `([z,] y, x)`:
# every axis reversed against the Julia layout `(kx, ky, [kz,] coil)`.
_sp_rev(a) = parent(permutedims(CMP_CTYPE.(a), ndims(a):-1:1))
_sp_k(ksp) = _sp_rev(ksp)
_sp_s(smaps) = _sp_rev(smaps)

"""
    sigpy_recon(method, ksp, smaps; λ, iterations) -> (time_ms, image)

`ksp` is a zero-filled Cartesian `(kx, ky, coil)` or `(kx, ky, kz, coil)` array and `smaps` the
matching maps; SigPy's apps are dimension-agnostic, so the 3D volume goes through the same calls.
`method ∈ (:adjoint, :cgsense, :tv, :wavelet)`. TGV / low-rank unsupported here (throw).
Only **TV** is forced onto ADMM (`rho = CMP_RHO`, `max_cg_iter = CMP_CG_ITERS`) — matching the
ADMM the other toolkits use for TV; SigPy would otherwise default to PDHG. **L1-wavelet** keeps
SigPy's natural proximal-gradient solver (FISTA-like), as MRT / MRIReco / BART also use FISTA for
wavelet. `tol ≈ 0` so all `iterations` outer steps run. `:adjoint` is `Sense(mps)ᴴ y`.
"""
function sigpy_recon(method::Symbol, ksp3, smaps3; λ = 0.0, iterations = 10)
    y, mps = _sp_k(ksp3), _sp_s(smaps3)
    app = if method === :adjoint
        S = sp_mri.linop.Sense(mps)
        () -> S.H(y)
    elseif method === :cgsense
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
    return t * 1000, Array{ComplexF64}(permutedims(raw, ndims(raw):-1:1))
end

# --- MIRT.jl (Julia) --------------------------------------------------------------------------
# MIRT is Fessler's toolbox and is shaped differently from the other three: it ships system
# objects (`Asense`, `Anufft`) and generic solvers (`ncg`, `pogm_restart`) rather than
# reconstruction "apps", so each row is assembled here out of those pieces. That is the intended
# use, and it is why MIRT appears only in the rows whose functional needs no calibrated λ:
#
#   * `Asense` builds the Cartesian SENSE operator from a Boolean sampling mask, with `odim`
#     `(count(samp), ncoil)` — the samples in linear index order, one column per coil.
#   * CG-SENSE is `ncg` on `f(v) = ½‖v - y‖²` with `B = [A]`, whose MM line search reduces to
#     linear CG for this quadratic, so the iteration count means the same thing as everywhere else.
#   * `Asense` is not unitary by default, which leaves a global factor on the result; every row
#     here is scored with `mag_nrmse`, which normalises it away.
#
# L1-wavelet and TV are deliberately absent: MIRT would need its own entry in
# `lambda_calibration.json` for the comparison to stay at matched accuracy.
const MIRT = ComparisonHarness.MIRT

_mirt_samp(ksp3) = dropdims(any(!iszero, ksp3; dims = 3); dims = 3)
_mirt_y(ksp3, samp) = reduce(hcat, [ComplexF32.(ksp3[:, :, c])[samp] for c in axes(ksp3, 3)])

"""
    mirt_system(ksp3, smaps3) -> (A, y)

`Asense` for the sampling pattern implied by the zero-filled `ksp3` (nx, ny, coil), plus the
sampled data in the layout that operator produces.
"""
function mirt_system(ksp3, smaps3)
    samp = _mirt_samp(ksp3)
    A = MIRT.Asense(samp, ComplexF32.(smaps3))
    return A, _mirt_y(ksp3, samp)
end

"""
    mirt_recon(method, ksp3, smaps3; iterations) -> (time_ms, image)

`method ∈ (:adjoint, :cgsense)`; anything else throws so the caller drops the row.
"""
function mirt_recon(method::Symbol, ksp3, smaps3; iterations::Int = 10)
    A, y = mirt_system(ksp3, smaps3)
    f = if method === :adjoint
        () -> A' * y
    elseif method === :cgsense
        x0 = zeros(ComplexF32, size(smaps3, 1), size(smaps3, 2))
        () -> first(MIRT.ncg([A], [v -> v - y], [v -> 1.0f0], x0; niter = iterations))
    else
        error("MIRT has no $method here")
    end
    t, _, img = time_reconstruction(f)
    return t * 1000, Array{ComplexF64}(img)
end

"""
    mirt_gridding(kdata, traj, dcf, smaps3, image_size) -> (time_ms, image)

Density-compensated non-Cartesian adjoint: `Anufft` per coil, weighted by `dcf`, combined with
the conjugate sensitivities. `traj` is MRT's `(dim, k)` trajectory in cycles/sample, which MIRT
wants in radians; `n_shift` centres the image the way every other toolkit here does.
"""
function mirt_gridding(kdata, traj, dcf, smaps3, image_size)
    # (M, D), radians. Kept in Float64 and clamped: a radial trajectory reaches ±0.5 exactly, and
    # `Float32(2π * 0.5)` rounds just above π, which `nufft_init`'s `pi_error` check rejects.
    ω = clamp.(2π .* permutedims(Float64.(Array(traj)), (2, 1)), -π, π)
    A = MIRT.Anufft(ω, image_size; n_shift = collect(image_size) ./ 2)
    w = Float32.(vec(Array(dcf)))
    kd = ComplexF32.(Array(kdata))
    smap = ComplexF32.(Array(smaps3))
    function grid()
        acc = zeros(ComplexF32, image_size)
        for c in axes(kd, 2)
            acc .+= (A' * (w .* @view kd[:, c])) .* conj.(@view smap[:, :, c])
        end
        return acc
    end
    t, _, img = time_reconstruction(grid)
    return t * 1000, Array{ComplexF64}(img)
end

# --- global low-rank in SigPy and MIRT ----------------------------------------------------------
# Neither toolkit ships a low-rank MRI app, but both accept an arbitrary proximal operator, which is
# all a nuclear norm on the Casorati matrix needs. That makes the global low-rank row the one
# low-rank case they can both express faithfully; **locally** low rank is not, since it needs the
# block extraction and the cycle-spinning convention BART and MRT each have their own of, and
# comparing those would measure this file rather than the toolkits.
#
# Both rows solve exactly MRT's objective, ½‖Ax - y‖² + λ‖X‖_*, with the toolkit's own operator and
# its own solver: SigPy runs the same fixed-ρ ADMM as the other SigPy rows, MIRT runs POGM (Fessler's
# accelerated proximal gradient) since it ships no ADMM. The algorithm differs, so the iteration
# count is not comparable across those two rows the way it is between MRT and BART — that is what
# the calibrated λ and the accuracy column are for.

const sp = pyimport("sigpy")
const sp_linop = pyimport("sigpy.mri.linop")

# The prox lives in Python so SigPy's solver calls it without a round trip per iteration.
py"""
import numpy as np
import sigpy as sp

class _MrtSVT(sp.prox.Prox):
    '''Singular-value soft thresholding of the (frames x voxels) Casorati matrix.'''
    def __init__(self, shape, lamda):
        self.lamda = lamda
        super().__init__(shape)

    def _prox(self, alpha, input):
        m = input.reshape(input.shape[0], -1)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        s = np.maximum(s - alpha * self.lamda, 0)
        return (u @ (s[:, None] * vh)).reshape(input.shape)
"""

"""
    sigpy_lowrank(ksp4, smaps3, image_size; λ, iterations) -> (time_ms, image)

Global low-rank reconstruction of a zero-filled `(nx, ny, time, coil)` frame stack, through
`sigpy.app.LinearLeastSquares` on `sigpy.mri.linop.Sense` with the Casorati SVT prox above.
The solver settings match the other SigPy rows (`ADMM`, `rho = CMP_RHO`, `max_cg_iter =
CMP_CG_ITERS`). Returns the image as `(nx, ny, time)`.
"""
function sigpy_lowrank(ksp4, smaps3, image_size; λ = 0.0, iterations = 10)
    y = parent(permutedims(CMP_CTYPE.(ksp4), (3, 4, 2, 1)))          # (T, coil, ky, kx)
    mps = parent(permutedims(CMP_CTYPE.(smaps3), (3, 2, 1)))          # (coil, y, x)
    weights = Float64.(dropdims(sum(abs, y, dims = (1, 2)), dims = (1, 2)) .> 0)
    T = size(y, 1)
    ishape = (T, 1, image_size[2], image_size[1])
    # Built from primitives rather than through `sigpy.mri.linop.Sense`, which cannot express a
    # batched SENSE operator: given an explicit `ishape` it sets `img_ndim = len(ishape)` and then
    # transforms **every** axis, so a `(time, 1, y, x)` image gets Fourier-transformed over time and
    # coil as well. Same composition as `Sense` otherwise — P F S, with the `sqrt(weights)` that
    # SigPy's own apps apply to the operator and the data alike.
    S = sp.linop.Multiply(collect(ishape), mps)
    F = sp.linop.FFT(S.oshape, axes = (-2, -1))
    A = sp.linop.Multiply(F.oshape, sqrt.(weights)) * F * S
    y = y .* sqrt.(reshape(weights, 1, 1, size(weights)...))
    prox = py"_MrtSVT"(collect(ishape), λ)
    app = () -> sp.app.LinearLeastSquares(
        A, y; proxg = prox, solver = "ADMM", rho = CMP_RHO,
        max_cg_iter = CMP_CG_ITERS, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false
    ).run()
    t, _, raw = time_reconstruction(app)
    x = dropdims(Array{ComplexF64}(raw), dims = 2)                     # (T, y, x)
    return t * 1000, permutedims(x, (3, 2, 1))
end

"""
    _mirt_lipschitz(A, nx, ny; rtol, maxiter, safety) -> L

`ρ(A'A)` for POGM's step size `1/L`, as an **upper** bound.

POGM's worst-case rate is tight, so unlike FISTA it genuinely diverges when the step exceeds
`1/L` — and a power iteration converges *from below*. This operator makes the trap easy to fall
into: `A'A`'s leading eigenvalues are clustered, so convergence is slow. Traced on the dynamic
phantom (64², 4 coils), `ρ = 4926.95`:

| step | 10 | 30 | **50** | 100 | 150 | 200 |
|---|---|---|---|---|---|---|
| λmax | 4703.6 | 4820.5 | **4865.7** | 4910.5 | 4922.8 | 4926.9 |

A fixed 50 steps — what this function replaced — returns 4865.7, **1.24 % low**, and that was
enough to make the row diverge: at λ=10 it measured NRMSE 0.0789 at the true `ρ` but 0.5505 with
the 50-step estimate, and at 330 iterations the estimate produced 1.106. The failure is invisible
at the 20 iterations the row used to run, which is why it read as "MIRT converges to a worse
answer" rather than as an unstable step size.

So: iterate to a relative tolerance instead of a fixed count, and multiply by `safety` to land
above the limit rather than below it. The margin costs nothing measurable — at the matched budget
NRMSE is 0.0789 with `safety = 1.0` and 0.0789 with `safety = 1.2`, i.e. flat to four digits —
because POGM's step only has to be *valid*, not sharp. Seeded, so the row's accuracy does not
depend on the random draw.
"""
function _mirt_lipschitz(A, nx, ny; rtol = 1.0e-4, maxiter = 200, safety = 1.05)
    v = ComplexF32.(randn(Random.MersenneTwister(0), ComplexF64, nx, ny))
    v ./= Float32(norm(v))
    λ = 0.0
    for _ in 1:maxiter
        w = A' * (A * v)
        λnew = Float64(norm(w))
        v = w ./ Float32(λnew)
        converged = abs(λnew - λ) <= rtol * λnew
        λ = λnew
        converged && break
    end
    return safety * λ
end

# Singular-value soft thresholding of the Casorati matrix, the Julia side of the same prox.
function _svt(x::AbstractArray{<:Complex, 3}, τ::Real)
    nx, ny, nt = size(x)
    F = svd(reshape(x, nx * ny, nt))
    s = max.(F.S .- τ, 0)
    return reshape(F.U * (s .* F.Vt), nx, ny, nt)
end

"""
    mirt_lowrank(ksp4, smaps3; λ, iterations) -> (time_ms, image)

Global low-rank reconstruction with MIRT: one `Asense` for the (frame-independent) sampling
pattern, POGM with adaptive restart as the solver, and the Casorati SVT as its prox. POGM needs a
step size rather than the ρ the ADMM rows take; `f_L` comes from [`_mirt_lipschitz`](@ref), which
is an *upper* bound on `ρ(A'A)` and has to be — see its docstring.

`iterations` is a proximal-gradient count, so callers pass [`proxgrad_budget`](@ref) of the outer
count the ADMM rows use, not the outer count itself.
"""
function mirt_lowrank(ksp4, smaps3; λ = 0.0, iterations = 10)
    nx, ny, nt, nc = size(ksp4)
    samp = dropdims(any(!iszero, ksp4; dims = (3, 4)), dims = (3, 4))
    A = MIRT.Asense(samp, ComplexF32.(smaps3))
    y = [reduce(hcat, [ComplexF32.(ksp4[:, :, t, c])[samp] for c in 1:nc]) for t in 1:nt]

    x0 = zeros(ComplexF32, nx, ny, nt)
    f_grad = function (x)
        g = similar(x)
        for t in 1:nt
            g[:, :, t] = A' * (A * x[:, :, t] - y[t])
        end
        return g
    end
    # The SVD runs at the working precision, as SigPy's `_MrtSVT` and MRT's own prox do — promoting
    # to `ComplexF64` here would give MIRT a more accurate prox than the row it is compared against.
    g_prox = (z, c) -> _svt(z, λ * c)
    # `_mirt_lipschitz` is inside the timed closure, for the reason given on `CMP_FISTA_RHO_MRIRECO`:
    # the step size is part of what a solve costs, and MRT pays `estimate_opnorm` in its own timing.
    run = () -> first(
        MIRT.pogm_restart(
            x0, _ -> 0.0, f_grad, _mirt_lipschitz(A, nx, ny); niter = iterations, g_prox
        )
    )
    t, _, img = time_reconstruction(run)
    return t * 1000, Array{ComplexF64}(img)
end

# ================================================================= catalog adapters
# Everything below only rearranges a prepared `BenchCase` into each toolkit's layout and dispatches
# to the functions above. No data is generated or loaded here.

"""
    COMPETITORS

The toolkits every section compares MRT against, in row order.
"""
const COMPETITORS = (:bart, :sigpy, :mrireco, :mirt)

framework_label(tk::Symbol) = tk === :bart ? BART_FW : tk === :sigpy ? "SigPy" : tk === :mrireco ? "MRIReco" : "MIRT"
toolkit_key(tk::Symbol) = tk === :bart ? "BART" : tk === :sigpy ? "SigPy" : tk === :mrireco ? "MRIReco" : "MIRT"

"""
    supports(tk, c::BenchCase, method) -> Bool

Whether toolkit `tk` has a faithful implementation of `method` on case `c` here. A `false` is a
skip, logged by nothing: it is a property of the toolkit or of this file, not a failure.

| toolkit | Cartesian | non-Cartesian |
|---|---|---|
| BART | everything (multislice as a per-slice loop) | gridding, CG-SENSE, TV; cine: low-rank, LLR, temporal TV |
| SigPy | adjoint, CG-SENSE, TV, L1-wavelet (2D, per slice, 3D); cine: global low-rank | gridding, CG-SENSE, TV (2D) |
| MRIReco | adjoint, CG-SENSE, TV, L1-wavelet (2D, per slice); cine: global / locally low-rank | gridding, CG-SENSE (2D) |
| MIRT | adjoint, CG-SENSE (2D multichannel); cine: global low-rank | gridding (2D) |
"""
function supports(tk::Symbol, c::BenchCase, m::Symbol)
    m in applicable_methods(c) || return false
    cart = c.trajectory === :cartesian
    fam = c.family
    if tk === :bart
        cart && return true
        return fam === :cine || m in (:gridding, :cgsense, :tv)
    elseif tk === :sigpy
        cart && fam === :cine && return m === :lowrank
        cart && return m in (:adjoint, :cgsense, :tv, :wavelet)
        return fam === :single_slice && m in (:gridding, :cgsense, :tv)
    elseif tk === :mrireco
        cart && fam === :cine && return m in (:lowrank, :llr)
        cart && fam in (:single_slice, :multislice) && return m in (:adjoint, :cgsense, :tv, :wavelet)
        return !cart && fam === :single_slice && m in (:gridding, :cgsense)
    elseif tk === :mirt
        cart && fam === :cine && return m === :lowrank
        cart && fam === :single_slice && return ncoils(c) > 1 && m in (:adjoint, :cgsense)
        return !cart && fam === :single_slice && m === :gridding
    end
    return false
end

"""
    toolkit_run(tk, c, method; λ, maxit, runs) -> (time_ms, image)

Reconstruct case `c` by `method` with toolkit `tk`, timed over `runs` runs after a warm-up. The
image comes back in the case's reference layout.
"""
function toolkit_run(tk::Symbol, c::BenchCase, m::Symbol; λ::Real, maxit::Int, runs::Int = timed_runs(c))
    RUNS[] = runs
    tk === :bart && return bart_run(c, m; λ, maxit)
    tk === :sigpy && return sigpy_run(c, m; λ, maxit)
    tk === :mrireco && return mrireco_run(c, m; λ, maxit)
    tk === :mirt && return mirt_run(c, m; λ, maxit)
    throw(ArgumentError("unknown toolkit $tk"))
end

# Sensitivity maps of a single-slice case, ones for a single-channel one.
cart_maps(c::BenchCase) = c.smaps === nothing ? ones(ComplexF32, c.image_size..., 1) : c.smaps

# A multislice case as a loop of independent 2D problems: `f(ksp3, smaps3) -> (ms, image)`.
function per_slice(f, c::BenchCase)
    nz = size(c.kspace, 4)
    out = Array{ComplexF64}(undef, c.image_size..., nz)
    total = 0.0
    for z in 1:nz
        ms, x = f(c.kspace[:, :, :, z], c.smaps[:, :, :, z])
        total += ms
        out[:, :, z] = x
    end
    return total, out
end

# `(nx, ny, time, coil)` zero-filled cine stack, the layout the dynamic helpers above take.
cine_stack(c::BenchCase) = permutedims(c.kspace, (1, 2, 4, 3))

# ---------------------------------------------------------------- BART

"""
    bart_cmd(c, method, λ, maxit; budget = maxit * CMP_CG_ITERS) -> String

The `pics` command line for `method` on `c` (see `BART_BUDGET` for why an ADMM `-i` is a budget).
Regularizer flags cover the spatial axes (3, or 7 for the volume); the cine time axis is BART
dimension 5 (flag 32). `-b` is the LLR block edge (8) or, for global low rank, the whole image.
"""
function bart_cmd(c::BenchCase, m::Symbol, λ::Real, maxit::Int; budget::Int = maxit * CMP_CG_ITERS)
    sp = c.family === :volume ? 7 : 3
    admm = "-F -i $budget -u $CMP_RHO -C $CMP_CG_ITERS"
    m === :cgsense && return "pics -S -w 1 -i $maxit"
    m === :tv && return "pics -S -w 1 $admm -R T:$sp:0:$λ"
    m === :wavelet && return "pics -S -w 1 -e -i $maxit -R W:$sp:0:$λ"
    m === :tgv && return "pics -S -w 1 $admm -R G:3:0:$λ"
    m === :lowrank && return "pics -S -w 1 -m $admm -n -b $(maximum(c.image_size)) -R L:3:3:$λ"
    m === :llr && return "pics -S -w 1 -m $admm -n -b 8 -R L:3:3:$λ"
    m === :ttv && return "pics -S -w 1 $admm -R T:32:0:$λ"
    throw(ArgumentError("BART has no $m here"))
end

"""
    bart_inputs(c) -> (inputs...,)

`c` in BART's layout: `(x, y, z, coil, maps, TE, ...)`, frames of a cine along dimension 5, and for
a non-Cartesian case the trajectory first, in pixel units (`k · n`, a zero third coordinate).
"""
function bart_inputs(c::BenchCase)
    nx, ny = c.image_size[1:2]
    nc = ncoils(c)
    if c.trajectory === :cartesian
        c.family === :volume && return (c.kspace, c.smaps)
        s = reshape(cart_maps(c), nx, ny, 1, nc)
        c.family === :cine && return (reshape(c.kspace, nx, ny, 1, nc, 1, size(c.kspace, 4)), s)
        return (reshape(c.kspace, nx, ny, 1, nc), s)
    end
    ns, nsp = size(c.traj, 2), size(c.traj, 3)
    traj = cat(c.traj[1:1, :, :] .* nx, c.traj[2:2, :, :] .* ny, zeros(Float32, 1, ns, nsp); dims = 1)
    s = reshape(c.smaps, nx, ny, 1, nc)
    c.family === :cine && return (ComplexF32.(traj), reshape(c.kspace, 1, ns, nsp, nc, 1, size(c.kspace, 4)), s)
    return (ComplexF32.(traj), reshape(c.kspace, 1, ns, nsp, nc), s)
end

# pics output → the case's image layout.
function bart_image(c::BenchCase, r)
    c.family === :volume && return r[:, :, :]
    c.family === :cine && return reshape(r, c.image_size..., size(c.reference, 3))
    return r[:, :, 1]
end

function bart_run(c::BenchCase, m::Symbol; λ, maxit, budget = maxit * CMP_CG_ITERS)
    nx, ny = c.image_size[1:2]
    nc = ncoils(c)
    if c.family === :multislice
        return per_slice(c) do k, s
            one = BenchCase(; id = c.id, family = :single_slice, trajectory = :cartesian, reference = c.reference[:, :, 1], smaps = s, kspace = k, image_size = c.image_size)
            bart_run(one, m; λ, maxit, budget)
        end
    end
    inputs = bart_inputs(c)
    # A direct reconstruction takes a few ms in-process against ~100 ms of BART process spawn and
    # file I/O with ±30 ms jitter, so its time is not measurable (see run_base.jl's header): it is
    # recorded as NaN and only its agreement is scored.
    # Coil images come back with the coil axis at BART dimension 4 in every layout, and the maps
    # broadcast over a cine's frames.
    if m === :adjoint
        k, s = inputs
        imgs = run_bart(1, "fft -i $(c.family === :volume ? 7 : 3)", k)
        return NaN, reshape(sum(imgs .* conj.(s); dims = 4), size(c.reference))
    elseif m === :gridding
        traj, k, s = inputs
        imgs = run_bart(1, "nufft -a -d $nx:$ny:1 -t", traj, k .* reshape(c.dcf, 1, size(c.dcf)...))
        return NaN, reshape(sum(imgs .* conj.(s); dims = 4), size(c.reference))
    end
    cmd = bart_cmd(c, m, λ, maxit; budget)
    c.trajectory === :noncartesian && (cmd *= " -t")
    tb, _, r = time_bart(cmd, inputs...; num_runs = RUNS[])
    return 1000 * tb, bart_image(c, r)
end

# ---------------------------------------------------------------- SigPy

function sigpy_run(c::BenchCase, m::Symbol; λ, maxit)
    if c.trajectory === :cartesian
        c.family === :multislice && return per_slice((k, s) -> sigpy_recon(m, k, s; λ, iterations = maxit), c)
        c.family === :cine && return sigpy_lowrank(cine_stack(c), c.smaps, c.image_size; λ, iterations = maxit)
        return sigpy_recon(m, c.kspace, c.family === :volume ? c.smaps : cart_maps(c); λ, iterations = maxit)
    end
    return sigpy_noncartesian(m, c; λ, iterations = maxit)
end

"""
    sigpy_noncartesian(method, c; λ, iterations) -> (time_ms, image)

SigPy's NUFFT path for a single-slice non-Cartesian case: `coord` is `(spoke, sample, 2)` in pixel
units with the last axis ordered like SigPy's image axes, `(y, x)`. `:gridding` is
`Sense(mps, coord)ᴴ (dcf · y)`; `:cgsense` and `:tv` are the SigPy apps given `coord`.
"""
function sigpy_noncartesian(m::Symbol, c::BenchCase; λ = 0.0, iterations = 10)
    nx, ny = c.image_size
    ns, nsp = size(c.traj, 2), size(c.traj, 3)
    coord = Array{Float64}(undef, nsp, ns, 2)
    for j in 1:nsp, i in 1:ns
        coord[j, i, 1] = c.traj[2, i, j] * ny
        coord[j, i, 2] = c.traj[1, i, j] * nx
    end
    y = parent(permutedims(CMP_CTYPE.(c.kspace), (3, 2, 1)))                 # (coil, spoke, sample)
    mps = _sp_s(c.smaps)
    app = if m === :gridding
        S = sp_mri.linop.Sense(mps; coord)
        w = reshape(permutedims(c.dcf, (2, 1)), 1, nsp, ns)
        yw = y .* w
        () -> S.H(yw)
    elseif m === :cgsense
        () -> sp_app.SenseRecon(y, mps; coord, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false).run()
    elseif m === :tv
        () -> sp_app.TotalVariationRecon(
            y, mps, λ; coord, solver = "ADMM", rho = CMP_RHO,
            max_cg_iter = CMP_CG_ITERS, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false
        ).run()
    else
        error("SigPy has no non-Cartesian $m here")
    end
    t, _, raw = time_reconstruction(app)
    return t * 1000, Array{ComplexF64}(permutedims(raw, (2, 1)))
end

# ---------------------------------------------------------------- MRIReco

function mrireco_run(c::BenchCase, m::Symbol; λ, maxit)
    if c.trajectory === :noncartesian
        return mrireco_noncartesian(m, c; iterations = maxit)
    elseif c.family === :cine
        return mrireco_dynamic(m, cine_stack(c), c.smaps, c.image_size; λ, iterations = maxit)
    end
    f = (k, s) -> m === :adjoint ? mrireco_adjoint(k, s) : mrireco(m, k, s, c.image_size; λ, iterations = maxit)
    c.family === :multislice && return per_slice(f, c)
    return f(c.kspace, cart_maps(c))
end

"""
    mrireco_adjoint(ksp3, smaps3) -> (time_ms, image)

MRIReco's `direct` reconstruction (per-coil images) followed by the conjugate-sensitivity coil
combination, inside the timed region: MRT's direct row includes the sensitivity adjoint.
"""
function mrireco_adjoint(ksp3, smaps3)
    nx, ny, nc = size(ksp3)
    acq = _mrireco_acq(ksp3)
    s = CMP_CTYPE.(smaps3)
    rp = Dict{Symbol, Any}(:reco => "direct", :reconSize => (nx, ny), :senseMaps => reshape(s, nx, ny, 1, nc))
    t, _, x = time_reconstruction() do
        imgs = with_mrireco_blas(() -> MRIReco.reconstruction(acq, rp)[:, :, 1, 1, :])
        dropdims(sum(imgs .* conj.(s); dims = 3); dims = 3)
    end
    return t * 1000, Array{ComplexF64}(x)
end

"""
    mrireco_noncartesian(method, c; iterations) -> (time_ms, image)

MRIReco on the case's own trajectory (`MRIBase.Trajectory` from the `(2, sample, spoke)` nodes,
`circular = true` as for radial). `:gridding` is its `direct` reconstruction, which applies
MRIReco's own density compensation, followed by the conjugate-sensitivity combination; `:cgsense`
is `multiCoil` CGNR, which MRIReco weights by its sampling density.
"""
function mrireco_noncartesian(m::Symbol, c::BenchCase; iterations = 10)
    nx, ny = c.image_size
    ns, nsp = size(c.traj, 2), size(c.traj, 3)
    nc = ncoils(c)
    tr = MRIReco.Trajectory(Float32.(reshape(c.traj, 2, :)), nsp, ns; circular = true)
    acq = AcquisitionData(tr, fill(reshape(CMP_CTYPE.(c.kspace), ns * nsp, nc), 1, 1, 1))
    s = CMP_CTYPE.(c.smaps)
    smap = reshape(s, nx, ny, 1, nc)
    if m === :gridding
        rp = Dict{Symbol, Any}(:reco => "direct", :reconSize => (nx, ny), :senseMaps => smap)
        t, _, x = time_reconstruction() do
            imgs = with_mrireco_blas(() -> MRIReco.reconstruction(acq, rp)[:, :, 1, 1, :])
            dropdims(sum(imgs .* conj.(s); dims = 3); dims = 3)
        end
        return t * 1000, Array{ComplexF64}(x)
    elseif m === :cgsense
        rp = Dict{Symbol, Any}(
            :reco => "multiCoil", :reconSize => (nx, ny), :senseMaps => smap, :solver => MR_CGNR,
            :reg => L2Regularization(0.0), :iterations => iterations, :absTol => 0.0, :relTol => 0.0,
        )
        t, _, x = time_reconstruction(() -> with_mrireco_blas(() -> MRIReco.reconstruction(acq, rp)[:, :, 1, 1, 1]))
        return t * 1000, Array{ComplexF64}(x)
    end
    error("MRIReco has no non-Cartesian $m here")
end

# ---------------------------------------------------------------- MIRT

function mirt_run(c::BenchCase, m::Symbol; λ, maxit)
    c.family === :cine && return mirt_lowrank(cine_stack(c), c.smaps; λ, iterations = proxgrad_budget(maxit))
    if c.trajectory === :noncartesian
        nc = ncoils(c)
        return mirt_gridding(reshape(c.kspace, :, nc), reshape(c.traj, 2, :), vec(c.dcf), c.smaps, c.image_size)
    end
    return mirt_recon(m, c.kspace, c.smaps; iterations = maxit)
end
