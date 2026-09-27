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
# harness run the same solve. `CMP_RHO` is the ADMM penalty every competitor's ADMM path falls back
# to when a case has no calibrated one (`load_rho`); the value is absolute, in each toolkit's own
# operator scaling. MRT's own rows fall back to `admm_rho(c)` instead, which is relative to `‖𝒜‖²`,
# so the one number means different things: `calibrate_lambda.jl` fits ρ per toolkit for that reason.
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
# 2D encode, a 3D encode otherwise); unsampled entries must be zero, and each echo's sampling
# pattern is read from its own nonzero samples. `ksp` is the Julia `(nx, ny, coil)` or
# `(nx, ny, nz, coil)` layout.
function _mrireco_acq(ksp)
    ndims(ksp) == 3 || return AcquisitionData(reshape(CMP_CTYPE.(ksp), size(ksp)..., 1, 1))
    return AcquisitionData(reshape(CMP_CTYPE.(ksp), size(ksp, 1), size(ksp, 2), 1, size(ksp, 3), 1, 1); enc2D = true)
end

# A zero-filled `(nx, ny, time, coil)` cine stack, the frames as echoes (see `mrireco_dynamic`).
function _mrireco_cine_acq(ksp4)
    nx, ny, nt, ncoil = size(ksp4)
    ksp6 = zeros(CMP_CTYPE, nx, ny, 1, ncoil, nt, 1)
    for t in 1:nt
        ksp6[:, :, 1, :, t, 1] .= CMP_CTYPE.(@view ksp4[:, :, t, :])
    end
    return AcquisitionData(ksp6; enc2D = true)
end

"""
    _mrireco_nc_acq(c) -> AcquisitionData

A non-Cartesian case on its own trajectory (`MRIBase.Trajectory` from the `(2, sample, spoke)`
nodes), a cine's frames as echoes on the shared trajectory. `circular = false` although the
trajectory is radial: a circular trajectory makes `reconstruction` zero the image outside the
inscribed circle after the solve (`circularShutter!`, `IterativeReconstruction.jl:72,244`), a
post-processing step no other toolkit here applies. `encodingSize` is given explicitly because the
constructor otherwise infers the encoding dimension from the trajectory vector (1).
"""
function _mrireco_nc_acq(c::BenchCase)
    ns, nsp = size(c.traj, 2), size(c.traj, 3)
    nc = ncoils(c)
    nt = c.family === :cine ? size(c.kspace, 4) : 1
    tr = MRIReco.Trajectory(Float32.(reshape(c.traj, 2, :)), nsp, ns; circular = false)
    frame(t) = c.family === :cine ? c.kspace[:, :, :, t] : c.kspace
    kdata = [reshape(CMP_CTYPE.(frame(t)), ns * nsp, nc) for t in 1:nt, _ in 1:1, _ in 1:1]
    return AcquisitionData(fill(tr, nt), kdata; encodingSize = c.image_size)
end

# Sensitivity maps as MRIReco takes them: `(x, y, 1, coil)` in 2D, `(x, y, z, coil)` in 3D.
_mrireco_maps(smaps, reconSize) =
    reshape(CMP_CTYPE.(smaps), reconSize..., ntuple(_ -> 1, 3 - length(reconSize))..., size(smaps, ndims(smaps)))

# A single-contrast `reconstruction` result `(x, y, z or slice, echo, coil, rep)` as a
# `reconSize` image; any further trailing axis (frames, coils) follows it.
_mrireco_image(img, reconSize, rest...) = reshape(Array{ComplexF64}(img), reconSize..., rest...)

"""
    MRIRECO_UNWEIGHTED

Parameters every MRIReco iterative row passes: `densityWeighting = false`.

MRIReco's iterative reconstructions weight the data term by `W = WeightingOp(samplingDensity)`
(`IterativeReconstruction.jl:233,309`). On a Cartesian acquisition that density is the uniform
`1/√N`, but on a non-Cartesian one it is the square root of an iterative density compensation
(`MRIBase` `samplingDensity`, `sdc(plan, iters = 10)`), so the solve minimises the density-weighted
`½‖W(Ax - y)‖²` rather than the `½‖Ax - y‖²` MRT, BART and SigPy minimise: a different problem,
whose CG is also preconditioned by `W`. Measured on the small radial phantom at 10 iterations,
weighted CG-SENSE reaches NRMSE 0.348 and unweighted 0.412, against MRT's 0.420. With
`densityWeighting = false` MRIReco uses the uniform `1/√N` for every trajectory
(`RecoParameters.jl:94-102`), the same weights as its Cartesian path, so every MRIReco row solves the
unweighted problem up to a constant factor, which λ and ρ absorb.
"""
const MRIRECO_UNWEIGHTED = (; densityWeighting = false)

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
    mrireco(method, mkacq, smaps, reconSize; λ, iterations, ρ) -> (time_ms, image)

A static (2D or 3D) reconstruction of the acquisition `mkacq()` builds, with maps `smaps`
(`(x, y, [z,] coil)`). `method ∈ (:cgsense, :atv, :wavelet, :nuclear, :llr)`. TGV / temporal-TV are
unsupported (throw). The acquisition is built inside the timed region, as `reconstruction`'s input.
Runs with `vary_rho = :none`, `iterationsCG = CMP_CG_ITERS` and zero tolerances so the full
iteration budget is spent (`RegularizedLeastSquares.filterKwargs` drops the keys a given solver
does not accept, so the same kwargs are safe for CGNR / ADMM / FISTA).

`ρ = nothing` for `:wavelet`, the one FISTA path, means *estimate the step size*:
`0.95 / power_iterations(AHA)`, FISTA's own constructor default, computed inside the timed region
because that is where MRT's equivalent `estimate_opnorm` is charged. See `CMP_FISTA_RHO_MRIRECO`
for why it is not passed as a constant. For the ADMM rows `ρ` is a fixed penalty, not a step
size: the case's calibrated one (`load_rho`) or `CMP_RHO`, so nothing is estimated there.
"""
function mrireco(
        method::Symbol, mkacq, smaps, reconSize; λ = 0.0, iterations = 10,
        ρ::Union{Real, Nothing} = method === :wavelet ? nothing : CMP_RHO
    )
    # `regTrafo` stays `opEye` for everything except TV — see the TV branch.
    reg, solver, sparse, regTrafo = if method === :cgsense
        (L2Regularization(0.0), MR_CGNR, nothing, nothing)
    elseif method === :atv
        # RegularizedLeastSquares' TV is anisotropic in every form: `L1Regularization` of
        # `GradientOp`, and `TVRegularization`, whose dual projection clips each difference on its
        # own (`tv_restrictMagnitude!`). `L21Regularization` groups `x[i:n:end]` of the stacked
        # differences, which are not the differences of one pixel, so isotropic TV has no row.
        #
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
        (LLRRegularization(λ; shape = reconSize, blockSize = (8, 8), randshift = false), MR_ADMM, nothing, nothing)
    else
        error("MRIReco has no $method")
    end
    senseMaps = _mrireco_maps(smaps, reconSize)
    # A separate `AcquisitionData` on purpose: the timed closure builds its own (as every other
    # row does), so this one costs the row nothing.
    AHA = ρ === nothing ? _mrireco_normal_operator(mkacq(), senseMaps, reconSize) : nothing
    t, _, img = time_reconstruction() do
        rp = Dict{Symbol, Any}(
            :reco => "multiCoil", :reconSize => reconSize, :senseMaps => senseMaps,
            :solver => solver, :reg => reg, :iterations => iterations,
            :rho => AHA === nothing ? ρ : 0.95 / RLS.power_iterations(AHA),
            :vary_rho => :none, :iterationsCG => CMP_CG_ITERS,
            :absTol => 0.0, :relTol => 0.0, :tolInner => CMP_TOL_INNER, pairs(MRIRECO_UNWEIGHTED)...,
        )
        sparse !== nothing && (rp[:sparseTrafo] = sparse)
        regTrafo !== nothing && (rp[:regTrafo] = regTrafo)
        with_mrireco_blas() do
            MRIReco.reconstruction(mkacq(), rp)
        end
    end
    return t * 1000, _mrireco_image(img, reconSize)
end

"""
    mrireco_dynamic(method, acq, smaps3, reconSize, nt; λ, iterations, ρ) -> (time_ms, image)

Dynamic (2D+t) reconstruction with MRIReco of `acq`, a cine with its `nt` frames as echoes
(`_mrireco_cine_acq` or `_mrireco_nc_acq`). `method ∈ (:adjoint, :gridding, :cgsense, :lowrank,
:llr)`: the first two are the `direct` reconstruction of every frame with the conjugate-sensitivity
combination (see `mrireco_direct`), CG-SENSE is CGNR on the joint system of all frames, as MRT and
BART solve it. Returns `(nx, ny, nt)`.

The frames are handed to MRIReco as **contrasts (echoes)**, not repetitions, and the solve goes
through `reco = "multiCoilMultiEcho"` — `reconstruction_multiCoil` loops over repetitions and slices
and solves each independently (`IterativeReconstruction.jl:54`), which would decouple the frames and
make a temporal prior meaningless, while `reconstruction_multiCoilMultiEcho` builds one system over
all contrasts (`:273`) and applies the regularizer to the stacked volume. That is what MRT's
`LowRank` / `LocallyLowRank` do, so the two are comparable.

`RegularizedLeastSquares` needs the volume shape spelled out, since the prox reshapes a flat vector:
`NuclearRegularization` takes `svtShape = (prod(reconSize), n_frames)` — the Casorati matrix, i.e.
MRT's global `LowRank` — and `LLRRegularization` takes the *spatial* `shape = reconSize` with
`blockSize = (8, 8)`. Its prox reshapes the vector to `(shape..., K)` and thresholds the singular
values of each block's `(64, K)` Casorati matrix, so the frames must be the trailing `K`: given
`shape = (nx, ny, n_frames)` and `blockSize = (8, 8, n_frames)`, as this row once did, `K = 1` and
each block is a single `64·n_frames` vector whose "SVT" merely shrinks its norm — a group-sparsity
penalty, which measured 2.6× MRT's NRMSE on the Cartesian cine. `randshift = false` tiles the blocks
at fixed positions, as MRT's `LocallyLowRank` and BART's `-n` do; the default shifts them randomly
every iteration, a different objective.

**Temporal TV is not reachable through this API and therefore has no MRIReco row.** Not for lack of
a prox — `L1Regularization` + a `GradientOp` along the time axis is the right formulation, and it is
what MRIReco's own TV path uses spatially — but `reconstruction_multiCoilMultiEcho` wraps whatever
`regTrafo` it is given in `DiagOp(repeat([trafo], numContr)...)` (`IterativeReconstruction.jl:304`),
i.e. it applies the transform *per contrast*. A per-frame block-diagonal transform cannot couple
frames, so a time-difference operator over the `(nx, ny, n_frames)` volume simply does not fit and
throws `LinearOperatorException("shape mismatch")`.
"""
function mrireco_dynamic(
        method::Symbol, acq, smaps3, reconSize, nt; λ = 0.0, iterations = 10, ρ = CMP_RHO
    )
    method in (:adjoint, :gridding) && return mrireco_direct(acq, smaps3, reconSize; frames = nt)
    solver, reg = if method === :cgsense
        MR_CGNR, L2Regularization(0.0)
    elseif method === :lowrank
        MR_ADMM, RLS.NuclearRegularization(λ; svtShape = (prod(reconSize), nt))
    elseif method === :llr
        MR_ADMM, RLS.LLRRegularization(λ; shape = reconSize, blockSize = (8, 8), randshift = false)
    else
        error("MRIReco has no dynamic $method here (see the docstring on temporal TV)")
    end
    senseMaps = _mrireco_maps(smaps3, reconSize)
    t, _, img = time_reconstruction() do
        rp = Dict{Symbol, Any}(
            :reco => "multiCoilMultiEcho", :reconSize => reconSize, :senseMaps => senseMaps,
            :solver => solver, :reg => reg, :iterations => iterations, :rho => ρ,
            :vary_rho => :none, :iterationsCG => CMP_CG_ITERS,
            :absTol => 0.0, :relTol => 0.0, :tolInner => CMP_TOL_INNER, pairs(MRIRECO_UNWEIGHTED)...,
        )
        with_mrireco_blas() do
            MRIReco.reconstruction(acq, rp)
        end
    end
    return t * 1000, _mrireco_image(img, reconSize, nt)
end

# --- Python (SigPy, MRpro) -----------------------------------------------------------------
# Python code of the harness's own (`_IsoTVRecon`, `_MrtSVT`, `_mrpro_solve`, ...) is executed into
# `PY` by the `pyexec` blocks below and looked up there by name.
const PY = pydict()
const np = pyimport("numpy")

# A Julia array as a NumPy array of the same shape sharing its memory (column-major strides);
# anything else as PythonCall converts it.
_np(a::AbstractArray) = np.asarray(a)
_np(x) = x

# The Python reconstruction `f` with its NumPy result copied into a Julia array of the same shape,
# inside the timed region, so that a toolkit's time includes handing its image back.
_jl(f) = () -> pyconvert(Array, f())

# --- SigPy (Python) ------------------------------------------------------------------------
# SigPy wants k-space `(coil, [kz,] ky, kx)` and maps `(coil, [z,] y, x)`, returns `([z,] y, x)`:
# every axis reversed against the Julia layout `(kx, ky, [kz,] coil)`.
_sp_rev(a) = parent(permutedims(CMP_CTYPE.(a), ndims(a):-1:1))
_sp_k(ksp) = _np(_sp_rev(ksp))
_sp_s(smaps) = _np(_sp_rev(smaps))

# Isotropic TV in SigPy. SigPy ships only the anisotropic one (`TotalVariationRecon`: an `L1Reg`
# prox on `FiniteDifference`, whose output stacks the per-axis differences along axis 0) and no
# joint (group) threshold. `_IsoTVRecon` is `TotalVariationRecon` with that prox replaced by the
# joint soft threshold over axis 0, the prox of `λ Σ ‖∇x‖₂`; data weighting, operator and solver
# are built exactly as `TotalVariationRecon` builds them.
pyexec(
    """
    import numpy as np
    import sigpy as sp
    import sigpy.mri as spm
    from sigpy.mri.app import _estimate_weights

    class _JointL1(sp.prox.Prox):
        '''Soft thresholding of the l2 norm over axis 0: the prox of lamda * sum ||x[:, i]||_2.'''
        def __init__(self, shape, lamda):
            self.lamda = lamda
            super().__init__(shape)

        def _prox(self, alpha, input):
            n = np.sqrt(np.sum(np.abs(input) ** 2, axis=0, keepdims=True))
            return input * np.maximum(1 - alpha * self.lamda / np.maximum(n, 1e-30), 0)

    def _IsoTVRecon(y, mps, lamda, coord=None, **kwargs):
        weights = _estimate_weights(y, None, coord)
        if weights is not None:
            y = y * weights ** 0.5
        A = spm.linop.Sense(mps, coord=coord, weights=weights)
        G = sp.linop.FiniteDifference(A.ishape)
        return sp.app.LinearLeastSquares(A, y, proxg=_JointL1(G.oshape, lamda), G=G, **kwargs)
    """, PY,
)

"""
    sigpy_recon(method, ksp, smaps; λ, iterations) -> (time_ms, image)

`ksp` is a zero-filled Cartesian `(kx, ky, coil)` or `(kx, ky, kz, coil)` array and `smaps` the
matching maps; SigPy's apps are dimension-agnostic, so the 3D volume goes through the same calls.
`method ∈ (:adjoint, :cgsense, :tv, :atv, :tv_pd, :atv_pd, :wavelet)`: `:atv` is SigPy's
`TotalVariationRecon`, `:tv` the isotropic `_IsoTVRecon`; `:tv_pd` / `:atv_pd` are the same apps
with `solver = "PrimalDualHybridGradient"` at SigPy's own step sizes. TGV / low-rank unsupported
here (throw).
Only **TV** is forced onto ADMM (`rho = ρ`, `max_cg_iter = CMP_CG_ITERS`) — matching the
ADMM the other toolkits use for TV; SigPy would otherwise default to PDHG. **L1-wavelet** keeps
SigPy's natural proximal-gradient solver (FISTA-like), as MRT / MRIReco / BART also use FISTA for
wavelet. `tol ≈ 0` so all `iterations` outer steps run. `:adjoint` is `Sense(mps)ᴴ y`, with the
`Sense` operator built inside the timed region, as every app builds its own.
"""
function sigpy_recon(method::Symbol, ksp3, smaps3; λ = 0.0, iterations = 10, ρ = CMP_RHO)
    y, mps = _sp_k(ksp3), _sp_s(smaps3)
    app = if method === :adjoint
        () -> sp_mri.linop.Sense(mps).H(y)
    elseif method === :cgsense
        () -> sp_app.SenseRecon(y, mps; max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false).run()
    elseif method in (:tv, :atv)
        tv = method === :tv ? PY["_IsoTVRecon"] : sp_app.TotalVariationRecon
        () -> tv(
            y, mps, λ; solver = "ADMM", rho = ρ,
            max_cg_iter = CMP_CG_ITERS, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false
        ).run()
    elseif method in (:tv_pd, :atv_pd)
        tv = method === :tv_pd ? PY["_IsoTVRecon"] : sp_app.TotalVariationRecon
        () -> tv(
            y, mps, λ; solver = "PrimalDualHybridGradient", max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false
        ).run()
    elseif method === :wavelet
        () -> sp_app.L1WaveletRecon(y, mps, λ; wave_name = CMP_WAVELET_NAME, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false).run()
    else
        error("SigPy has no $method here")
    end
    t, _, raw = time_reconstruction(_jl(app))
    return t * 1000, Array{ComplexF64}(permutedims(raw, ndims(raw):-1:1))
end

# --- MIRT.jl (Julia) --------------------------------------------------------------------------
# MIRT is Fessler's toolbox and is shaped differently from the other three: it ships system
# objects (`Asense`, `Anufft`) and generic solvers (`ncg`, `pogm_restart`) rather than
# reconstruction "apps", so each row is assembled here out of those pieces. That is the intended
# use:
#
#   * `Asense` builds the Cartesian SENSE operator from a Boolean sampling mask, with `odim`
#     `(count(samp), ncoil)` — the samples in linear index order, one column per coil. The
#     non-Cartesian one is `Anufft` composed with the maps (`_mirt_sense_nufft`).
#   * CG-SENSE is `ncg` on `f(v) = ½‖v - y‖²` with `B = [A]`, whose MM line search reduces to
#     linear CG for this quadratic, so the iteration count means the same thing as everywhere else.
#   * `Asense` is not unitary by default, which leaves a global factor on the result; every row
#     here is scored with `mag_nrmse`, which normalises it away.
#
# L1-wavelet and TV are absent because MIRT ships no prox for either; one written here would be
# this file's, not MIRT's.
const MIRT = ComparisonHarness.MIRT

const LinearMapAA = MIRT.LinearMapAA

# The sampling pattern of a zero-filled `(nx, ny, [nz,] coil)` k-space, and its samples in the
# `(count(samp), ncoil)` layout `Asense` produces.
_mirt_samp(ksp) = dropdims(any(!iszero, ksp; dims = ndims(ksp)); dims = ndims(ksp))
_mirt_y(ksp, samp) = reduce(hcat, [ComplexF32.(selectdim(ksp, ndims(ksp), c))[samp] for c in axes(ksp, ndims(ksp))])

"""
    mirt_problem(ksp, smaps) -> (build, y)

A function `build()` returning `Asense` for the sampling pattern implied by the zero-filled `ksp`
(`(nx, ny, [nz,] coil)`), plus the sampled data in the layout that operator produces. The rows call
`build` inside their timed region: MRT, MRIReco, MRpro and BART all build their operator inside
theirs.
"""
function mirt_problem(ksp, smaps)
    samp = _mirt_samp(ksp)
    maps = ComplexF32.(smaps)
    return () -> MIRT.Asense(samp, maps), _mirt_y(ksp, samp)
end

"""
    mirt_system(args...) -> (A, y)

[`mirt_problem`](@ref) with the operator built.
"""
function mirt_system(args...)
    build, y = mirt_problem(args...)
    return build(), y
end

"""
    mirt_problem(c::BenchCase) -> (build, y)

The SENSE system of a single-slice, volume or cine case as one `LinearMapAA` (built by `build()`)
and its data:

  * Cartesian: `Asense`, one per frame for a cine, since each frame has its own sampling pattern;
  * non-Cartesian: `Anufft` on the trajectory composed with the maps (`_mirt_sense_nufft`), the
    same operator for every frame, as the frames share the trajectory.

A cine's frames are stacked into one block-diagonal operator over `(nx, ny, time)`
(`_mirt_frames`), so that its CG-SENSE is one CG over all frames, as MRT and BART solve it.
"""
function mirt_problem(c::BenchCase)
    if c.trajectory === :cartesian
        c.family === :cine || return mirt_problem(c.kspace, c.family === :volume ? c.smaps : cart_maps(c))
        frames = [mirt_problem(c.kspace[:, :, :, t], c.smaps) for t in axes(c.kspace, 4)]
        builds = first.(frames)
        return () -> _mirt_frames([b() for b in builds]), reduce(vcat, vec.(last.(frames)))
    end
    traj = reshape(c.traj, 2, :)
    build = () -> _mirt_sense_nufft(traj, c.smaps, c.image_size)
    c.family === :cine || return build, ComplexF32.(reshape(c.kspace, :, ncoils(c)))
    nt = size(c.kspace, 4)
    return () -> _mirt_frames(fill(build(), nt)), ComplexF32.(vec(c.kspace))
end

# MRT's `(dim, k)` trajectory in cycles/sample as MIRT's `(k, dim)` in radians. Kept in Float64 and
# clamped: a radial trajectory reaches ±0.5 exactly, and `Float32(2π * 0.5)` rounds just above π,
# which `nufft_init`'s `pi_error` check rejects. `n_shift` centres the image the way every other
# toolkit here does.
function _mirt_nufft(traj, image_size)
    ω = clamp.(2π .* permutedims(Float64.(Array(traj)), (2, 1)), -π, π)
    A = MIRT.Anufft(ω, image_size; n_shift = collect(image_size) ./ 2)
    _cap_nfft_fft_threads!(A._prop.p)
    return A
end

"""
    MIRT_FFTW_THREADS

FFTW threads MIRT's transforms run with: `NUM_THREADS`, at most 8. On Julia 1.13 a transform
planned with 16 FFTW threads and executed from the root task segfaults intermittently inside
FFTW.jl's task callback (12 did not in 6 runs); MIRT runs its operators from the root task, so its
transforms are capped. See [`with_mirt_fftw`](@ref).
"""
const MIRT_FFTW_THREADS = min(NUM_THREADS, 8)

"""
    with_mirt_fftw(f)

Run `f()` with FFTW's global thread count at [`MIRT_FFTW_THREADS`](@ref), which the FFTs `Asense`
plans use, and restore `NUM_THREADS` afterwards, including on exception.
"""
function with_mirt_fftw(f)
    MIRT_FFTW_THREADS == NUM_THREADS && return f()
    FFTW.set_num_threads(MIRT_FFTW_THREADS)
    try
        return f()
    finally
        FFTW.set_num_threads(NUM_THREADS)
    end
end

# NFFT.jl plans its FFTs with `Threads.nthreads()` threads whatever FFTW's global count is, so the
# plan's two FFTs are replaced by `MIRT_FFTW_THREADS`-thread plans with NFFT's default flags.
function _cap_nfft_fft_threads!(p)
    Threads.nthreads() > MIRT_FFTW_THREADS || return p
    p.forwardFFT = plan_fft!(p.tmpVec, p.dims; num_threads = MIRT_FFTW_THREADS)
    p.backwardFFT = plan_bfft!(p.tmpVec, p.dims; num_threads = MIRT_FFTW_THREADS)
    return p
end

"""
    _mirt_sense_nufft(traj, smaps3, image_size) -> LinearMapAA

Non-Cartesian SENSE out of MIRT's pieces: `Anufft` applied to each coil image `sᶜ ⊙ x`, output
`(sample, coil)`. MIRT ships no non-Cartesian SENSE object, but a `LinearMapAA` of a forward and an
adjoint function is how its system models are composed.
"""
function _mirt_sense_nufft(traj, smaps3, image_size)
    F = _mirt_nufft(traj, image_size)
    s = ComplexF32.(Array(smaps3))
    nc = size(s, 3)
    M = size(traj, 2)
    forw = x -> reduce(hcat, [F * (x .* @view s[:, :, j]) for j in 1:nc])
    back = y -> sum(j -> (F' * y[:, j]) .* conj.(@view s[:, :, j]), 1:nc)
    return LinearMapAA(forw, back, (M * nc, prod(image_size)); idim = Tuple(image_size), odim = (M, nc), T = ComplexF32)
end

# The block-diagonal operator of per-frame operators `ops` over `(idim..., frame)`, its output the
# frames' outputs concatenated as one vector.
function _mirt_frames(ops)
    idim = ops[1]._idim
    offs = cumsum([0; [prod(A._odim) for A in ops]])
    frame(x, t) = copy(selectdim(x, length(idim) + 1, t))
    forw = x -> reduce(vcat, [vec(ops[t] * frame(x, t)) for t in eachindex(ops)])
    back = y -> stack([ops[t]' * reshape(y[(offs[t] + 1):offs[t + 1]], ops[t]._odim) for t in eachindex(ops)])
    return LinearMapAA(forw, back, (offs[end], prod(idim) * length(ops)); idim = (idim..., length(ops)), odim = (offs[end],), T = ComplexF32)
end

"""
    mirt_recon(method, build, y; iterations) -> (time_ms, image)

`method ∈ (:adjoint, :cgsense)` on the system `build()` (a `LinearMapAA`, see `mirt_problem`) and
data `y`; anything else throws so the caller drops the row. The image has the system's input
shape. The timed region includes building the system.
"""
function mirt_recon(method::Symbol, build, y; iterations::Int = 10)
    f = if method === :adjoint
        () -> build()' * y
    elseif method === :cgsense
        function ()
            A = build()
            x0 = zeros(ComplexF32, A._idim)
            return first(MIRT.ncg([A], [v -> v - y], [v -> 1.0f0], x0; niter = iterations))
        end
    else
        error("MIRT has no $method here")
    end
    t, _, img = time_reconstruction(f)
    return t * 1000, Array{ComplexF64}(img)
end

"""
    mirt_gridding(c) -> (time_ms, image)

Density-compensated non-Cartesian adjoint of case `c`: `Anufft` per coil (and frame), weighted by
the case's `dcf`, combined with the conjugate sensitivities. The timed region includes planning the
NUFFT.
"""
function mirt_gridding(c::BenchCase)
    traj = reshape(c.traj, 2, :)
    w = Float32.(vec(c.dcf))
    nc = ncoils(c)
    nt = c.family === :cine ? size(c.kspace, 4) : 1
    kd = ComplexF32.(reshape(c.kspace, :, nc, nt))
    smap = ComplexF32.(c.smaps)
    function grid()
        A = _mirt_nufft(traj, c.image_size)
        acc = zeros(ComplexF32, c.image_size..., nt)
        for t in 1:nt, j in 1:nc
            acc[:, :, t] .+= (A' * (w .* @view kd[:, j, t])) .* conj.(@view smap[:, :, j])
        end
        return acc
    end
    t, _, img = time_reconstruction(grid)
    return t * 1000, Array{ComplexF64}(nt == 1 ? img[:, :, 1] : img)
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
pyexec(
    """
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
    """, PY,
)

"""
    _sp_cine_system(c) -> (A, y, ishape)

The SENSE system of cine case `c` over all frames at once: the image is `(time, 1, y, x)`, the data
`(time, coil, ky, kx)` for a Cartesian case and `(time, coil, spoke, sample)` for a radial one.

Built from primitives rather than through `sigpy.mri.linop.Sense`, which cannot express a batched
SENSE operator: given an explicit `ishape` it sets `img_ndim = len(ishape)` and then transforms
**every** axis, so a `(time, 1, y, x)` image gets Fourier-transformed over time and coil as well.
Same composition as `Sense` otherwise: `P F S`, `F` the centred FFT over the two image axes or the
NUFFT on the shared trajectory. `P` is each frame's own sampling mask, read from that frame's
nonzero samples: the frames of a Cartesian cine are sampled differently, and the union of their
masks would treat every position another frame sampled as a measured zero.
"""
function _sp_cine_system(c::BenchCase)
    nx, ny = c.image_size
    y = parent(permutedims(CMP_CTYPE.(c.kspace), (4, 3, 2, 1)))
    mps = _sp_s(c.smaps)                                                 # (coil, y, x)
    ishape = (size(y, 1), 1, ny, nx)
    S = sp.linop.Multiply(ishape, mps)
    if c.trajectory === :cartesian
        mask = Float32.(sum(abs, y; dims = 2) .> 0)                      # (time, 1, ky, kx)
        A = sp.linop.Multiply(S.oshape, _np(mask)) * sp.linop.FFT(S.oshape, axes = (-2, -1)) * S
        return A, y .* mask, ishape
    end
    return sp.linop.NUFFT(S.oshape, _np(_sp_coord(c))) * S, y, ishape
end

"""
    sigpy_dynamic(method, c; λ, iterations, ρ) -> (time_ms, image)

Cine reconstruction with SigPy on the joint system of all frames (`_sp_cine_system`), returned as
`(nx, ny, time)`. `method ∈ (:adjoint, :gridding, :cgsense, :lowrank, :ttv)`:

  * `:adjoint` / `:gridding` apply `Aᴴ`, the latter to the DCF-weighted data;
  * `:cgsense` is `sigpy.app.LinearLeastSquares` with no prox, i.e. conjugate gradient;
  * `:lowrank` gives it the Casorati SVT prox above;
  * `:ttv` is `λ‖D_t x‖₁` with `G = FiniteDifference` along time and an `L1Reg` prox, the same
    composition `sigpy.mri.app.TotalVariationRecon` builds over the image axes. SigPy's finite
    difference is circular (`np.roll`), so it also penalises the last frame against the first,
    which MRT's and BART's temporal TV do not.

The regularized rows use the settings of the other SigPy rows (`ADMM`, `rho = ρ`,
`max_cg_iter = CMP_CG_ITERS`); `:ttv_pd` is `:ttv` with `solver = "PrimalDualHybridGradient"`.
"""
function sigpy_dynamic(m::Symbol, c::BenchCase; λ = 0.0, iterations = 10, ρ = CMP_RHO)
    A, y, ishape = _sp_cine_system(c)
    admm = (; solver = "ADMM", rho = ρ, max_cg_iter = CMP_CG_ITERS, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false)
    yn = _np(y)
    app = if m === :adjoint
        () -> A.H(yn)
    elseif m === :gridding
        yw = _np(y .* reshape(permutedims(c.dcf, (2, 1)), 1, 1, size(c.dcf, 2), size(c.dcf, 1)))
        () -> A.H(yw)
    elseif m === :cgsense
        () -> sp.app.LinearLeastSquares(A, yn; max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false).run()
    elseif m === :lowrank
        prox = PY["_MrtSVT"](ishape, λ)
        () -> sp.app.LinearLeastSquares(A, yn; proxg = prox, admm...).run()
    elseif m in (:ttv, :ttv_pd)
        G = sp.linop.FiniteDifference(ishape; axes = (0,))
        prox = sp.prox.L1Reg(G.oshape, λ)
        solver = m === :ttv ? admm :
            (; solver = "PrimalDualHybridGradient", max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false)
        () -> sp.app.LinearLeastSquares(A, yn; proxg = prox, G, solver...).run()
    else
        error("SigPy has no dynamic $m here")
    end
    t, _, raw = time_reconstruction(_jl(app))
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
function _mirt_lipschitz(A; rtol = 1.0e-4, maxiter = 200, safety = 1.05)
    v = ComplexF32.(randn(Random.MersenneTwister(0), ComplexF64, A._idim))
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
    mirt_lowrank(build, y; λ, iterations) -> (time_ms, image)

Global low-rank reconstruction with MIRT on a cine's joint system `build()` over `(nx, ny, time)`
and its data `y` (`mirt_problem`), the system built inside the timed region: POGM with adaptive
restart as the solver, and the Casorati SVT as its
prox. POGM needs a step size rather than the ρ the ADMM rows take; `f_L` comes from
[`_mirt_lipschitz`](@ref), which is an *upper* bound on `ρ(A'A)` and has to be — see its docstring.

`iterations` is a proximal-gradient count, so callers pass [`proxgrad_budget`](@ref) of the outer
count the ADMM rows use, not the outer count itself.
"""
function mirt_lowrank(build, y; λ = 0.0, iterations = 10)
    # The SVD runs at the working precision, as SigPy's `_MrtSVT` and MRT's own prox do — promoting
    # to `ComplexF64` here would give MIRT a more accurate prox than the row it is compared against.
    g_prox = (z, c) -> _svt(z, λ * c)
    # `_mirt_lipschitz` is inside the timed closure, for the reason given on `CMP_FISTA_RHO_MRIRECO`:
    # the step size is part of what a solve costs, and MRT pays `estimate_opnorm` in its own timing.
    function run()
        A = build()
        x0 = zeros(ComplexF32, A._idim)
        f_grad = x -> A' * (A * x - y)
        return first(
            MIRT.pogm_restart(
                x0, _ -> 0.0, f_grad, _mirt_lipschitz(A); niter = iterations, g_prox
            )
        )
    end
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
const COMPETITORS = (:bart, :sigpy, :mrireco, :mirt, :mrpro)

framework_label(tk::Symbol) = tk === :bart ? BART_FW : toolkit_key(tk)
toolkit_key(tk::Symbol) =
    tk === :bart ? "BART" : tk === :sigpy ? "SigPy" : tk === :mrireco ? "MRIReco" : tk === :mrpro ? "MRpro" : "MIRT"

"""
    supports(tk, c::BenchCase, method) -> Bool

Whether toolkit `tk` has a faithful implementation of `method` on case `c` here. A `false` is a
skip, logged by nothing: it is a property of the toolkit or of this file, not a failure. Only the
methods the case admits (`applicable_methods`) are considered, and every row below covers both
trajectories of its family.

| toolkit | static (2D, per slice, 3D) | cine |
|---|---|---|
| BART | everything | everything |
| SigPy | adjoint / gridding, CG-SENSE, isotropic / anisotropic TV (ADMM and PDHG), L1-wavelet | adjoint / gridding, CG-SENSE, global low-rank, temporal TV (ADMM and PDHG) |
| MRIReco | adjoint / gridding, CG-SENSE, anisotropic TV, L1-wavelet | adjoint / gridding, CG-SENSE, global / locally low-rank |
| MIRT | adjoint / gridding, CG-SENSE | adjoint / gridding, CG-SENSE, global low-rank |
| MRpro | adjoint / gridding, CG-SENSE, isotropic / anisotropic TV (PDHG), L1-wavelet | adjoint / gridding, CG-SENSE, global low-rank, temporal TV (PDHG) |

What is absent and why: TGV exists only in BART and MRT. SigPy and MIRT have no locally low-rank
prox, and building one here would compare this file's block convention rather than the toolkits
(see the low-rank section). MRIReco applies a regularizer's transform per frame, so it cannot express
temporal TV (see `mrireco_dynamic`). MIRT ships neither a TV nor a wavelet prox. Every TV of
RegularizedLeastSquares is anisotropic, so MRIReco has no isotropic TV row (see `mrireco`); BART's
anisotropic row sums one `-R T` term per axis, and SigPy's isotropic one replaces the prox of its
`TotalVariationRecon` with a joint threshold (`_IsoTVRecon`). The PDHG rows (`PDHG_METHODS`) are
absent from MRIReco, whose `PrimalDualSolver` takes only a dense matrix, and from MIRT. MRpro has
no ADMM, so of the TV rows it has only the PDHG ones, and no locally low-rank or TGV row.
"""
function supports(tk::Symbol, c::BenchCase, m::Symbol)
    m in applicable_methods(c) || return false
    cart = c.trajectory === :cartesian
    fam = c.family
    if tk === :bart
        cart && return true
        return fam === :cine || m in (:gridding, :cgsense, :tv, :atv, :tv_pd, :atv_pd)
    elseif tk === :sigpy
        fam === :cine && return m in (:adjoint, :gridding, :cgsense, :lowrank, :ttv, :ttv_pd)
        return m in (:adjoint, :gridding, :cgsense, :tv, :atv, :tv_pd, :atv_pd, :wavelet)
    elseif tk === :mrireco
        fam === :cine && return m in (:adjoint, :gridding, :cgsense, :lowrank, :llr)
        return m in (:adjoint, :gridding, :cgsense, :atv, :wavelet)
    elseif tk === :mirt
        fam === :cine && return m in (:adjoint, :gridding, :cgsense, :lowrank)
        return m in (:adjoint, :gridding, :cgsense)
    elseif tk === :mrpro
        fam === :cine && return m in (:adjoint, :gridding, :cgsense, :lowrank, :ttv_pd)
        return m in (:adjoint, :gridding, :cgsense, :tv_pd, :atv_pd, :wavelet)
    end
    return false
end

"""
    uses_admm(tk, c::BenchCase, method) -> Bool

Whether toolkit `tk` (`:mrt` included) solves `method` on `c` with a fixed-penalty ADMM, i.e.
whether a ρ is a parameter of its row. L1-wavelet runs FISTA everywhere, CG-SENSE CG, and MIRT's
only regularized row POGM.
"""
function uses_admm(tk::Symbol, c::BenchCase, m::Symbol)
    # A PDHG row (`PDHG_METHODS`) has step sizes, not a penalty, and falls through here.
    m in (:tv, :atv, :tgv, :lowrank, :llr, :ttv) || return false
    tk === :mrt && return true
    tk === :bart && return true
    tk === :sigpy && return m in (:tv, :atv, :lowrank, :ttv)
    tk === :mrireco && return m in (:atv, :lowrank, :llr)
    return false
end

"""
    toolkit_run(tk, c, method; λ, maxit, runs, ρ = nothing) -> (time_ms, image)

Reconstruct case `c` by `method` with toolkit `tk`, timed over `runs` runs after a warm-up. The
image comes back in the case's reference layout. `ρ` is the ADMM penalty of a row that
[`uses_admm`](@ref), `nothing` meaning `CMP_RHO`; other rows ignore it.
"""
function toolkit_run(
        tk::Symbol, c::BenchCase, m::Symbol; λ::Real, maxit::Int, runs::Int = timed_runs(c), ρ = nothing,
    )
    RUNS[] = runs
    ρ = something(ρ, CMP_RHO)
    tk === :bart && return bart_run(c, m; λ, maxit, ρ)
    tk === :sigpy && return sigpy_run(c, m; λ, maxit, ρ)
    tk === :mrireco && return mrireco_run(c, m; λ, maxit, ρ)
    tk === :mirt && return mirt_run(c, m; λ, maxit)
    tk === :mrpro && return mrpro_run(c, m; λ, maxit)
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
    bart_cmd(c, method, λ, maxit; budget = maxit * CMP_CG_ITERS, ρ = CMP_RHO) -> String

The `pics` command line for `method` on `c` (see `BART_BUDGET` for why an ADMM `-i` is a budget).
Regularizer flags cover the spatial axes (3, or 7 for the volume); the cine time axis is BART
dimension 5 (flag 32). `-b` is the LLR block edge (8) or, for global low rank, the whole image.
`ρ` is the ADMM penalty (`-u`).
"""
function bart_cmd(
        c::BenchCase, m::Symbol, λ::Real, maxit::Int; budget::Int = maxit * CMP_CG_ITERS, ρ::Real = CMP_RHO,
    )
    sp = c.family === :volume ? 7 : 3
    admm = "-F -i $budget -u $ρ -C $CMP_CG_ITERS"
    m === :cgsense && return "pics -S -w 1 -i $maxit"
    m === :tv && return "pics -S -w 1 $admm -R T:$sp:0:$λ"
    # `-R T` thresholds jointly over its gradient axis (`src/grecon/optreg.c`), so it is isotropic
    # over the axes of its flags; one term per axis leaves a size-1 gradient axis, whose joint
    # threshold is the plain L1 of that axis' differences.
    m === :atv && return "pics -S -w 1 $admm " * join(("-R T:$(1 << (d - 1)):0:$λ" for d in 1:(c.family === :volume ? 3 : 2)), " ")
    m === :wavelet && return "pics -S -w 1 -e -i $maxit -R W:$sp:0:$λ"
    m === :tgv && return "pics -S -w 1 $admm -R G:3:0:$λ"
    m === :lowrank && return "pics -S -w 1 -m $admm -n -b $(maximum(c.image_size)) -R L:3:3:$λ"
    m === :llr && return "pics -S -w 1 -m $admm -n -b 8 -R L:3:3:$λ"
    m === :ttv && return "pics -S -w 1 $admm -R T:32:0:$λ"
    # `-a` is the primal-dual (Chambolle-Pock) solver. It takes the data term only as one more prox
    # in its stack (`iter2_chambolle_pock` asserts there is no normal-equation operator), which
    # `--precond` sets up (`opt_precond_configure`); `-e` sizes the steps by a power method over the
    # whole stack. Its tolerance is fixed at 1e-4 (`src/grecon/italgo.c`), so it may stop before `-i`.
    haskey(PDHG_METHODS, m) || throw(ArgumentError("BART has no $m here"))
    return replace(bart_cmd(c, penalty_of(m), λ, maxit; budget, ρ), admm => "-a -e --precond -i $maxit")
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

function bart_run(c::BenchCase, m::Symbol; λ, maxit, budget = maxit * CMP_CG_ITERS, ρ = CMP_RHO)
    nx, ny = c.image_size[1:2]
    nc = ncoils(c)
    if c.family === :multislice
        return per_slice(c) do k, s
            one = BenchCase(; id = c.id, family = :single_slice, trajectory = :cartesian, reference = c.reference[:, :, 1], smaps = s, kspace = k, image_size = c.image_size)
            bart_run(one, m; λ, maxit, budget, ρ)
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
    cmd = bart_cmd(c, m, λ, maxit; budget, ρ)
    c.trajectory === :noncartesian && (cmd *= " -t")
    tb, _, r = time_bart(cmd, inputs...; num_runs = RUNS[])
    return 1000 * tb, bart_image(c, r)
end

# ---------------------------------------------------------------- SigPy

function sigpy_run(c::BenchCase, m::Symbol; λ, maxit, ρ = CMP_RHO)
    c.family === :cine && return sigpy_dynamic(m, c; λ, iterations = maxit, ρ)
    if c.trajectory === :cartesian
        c.family === :multislice && return per_slice((k, s) -> sigpy_recon(m, k, s; λ, iterations = maxit, ρ), c)
        return sigpy_recon(m, c.kspace, c.family === :volume ? c.smaps : cart_maps(c); λ, iterations = maxit, ρ)
    end
    return sigpy_noncartesian(m, c; λ, iterations = maxit, ρ)
end

# SigPy's `coord` of a non-Cartesian case: `(spoke, sample, 2)` in pixel units, the last axis
# ordered like SigPy's image axes, `(y, x)`.
function _sp_coord(c::BenchCase)
    nx, ny = c.image_size
    ns, nsp = size(c.traj, 2), size(c.traj, 3)
    coord = Array{Float64}(undef, nsp, ns, 2)
    for j in 1:nsp, i in 1:ns
        coord[j, i, 1] = c.traj[2, i, j] * ny
        coord[j, i, 2] = c.traj[1, i, j] * nx
    end
    return coord
end

"""
    sigpy_noncartesian(method, c; λ, iterations) -> (time_ms, image)

SigPy's NUFFT path for a single-slice non-Cartesian case, on `_sp_coord(c)`. `:gridding` is
`Sense(mps, coord)ᴴ (dcf · y)`, the operator built inside the timed region; `:cgsense` and the TV
rows are the apps of `sigpy_recon` given
`coord`.
"""
function sigpy_noncartesian(m::Symbol, c::BenchCase; λ = 0.0, iterations = 10, ρ = CMP_RHO)
    ns, nsp = size(c.traj, 2), size(c.traj, 3)
    coord = _np(_sp_coord(c))
    yj = parent(permutedims(CMP_CTYPE.(c.kspace), (3, 2, 1)))                # (coil, spoke, sample)
    y = _np(yj)
    mps = _sp_s(c.smaps)
    app = if m === :gridding
        w = reshape(permutedims(c.dcf, (2, 1)), 1, nsp, ns)
        yw = _np(yj .* w)
        () -> sp_mri.linop.Sense(mps; coord).H(yw)
    elseif m === :cgsense
        () -> sp_app.SenseRecon(y, mps; coord, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false).run()
    elseif m in (:tv, :atv)
        tv = m === :tv ? PY["_IsoTVRecon"] : sp_app.TotalVariationRecon
        () -> tv(
            y, mps, λ; coord, solver = "ADMM", rho = ρ,
            max_cg_iter = CMP_CG_ITERS, max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false
        ).run()
    elseif m in (:tv_pd, :atv_pd)
        tv = m === :tv_pd ? PY["_IsoTVRecon"] : sp_app.TotalVariationRecon
        () -> tv(
            y, mps, λ; coord, solver = "PrimalDualHybridGradient", max_iter = iterations, tol = CMP_TOL_INNER, show_pbar = false
        ).run()
    else
        error("SigPy has no non-Cartesian $m here")
    end
    t, _, raw = time_reconstruction(_jl(app))
    return t * 1000, Array{ComplexF64}(permutedims(raw, (2, 1)))
end

# ---------------------------------------------------------------- MRIReco

function mrireco_run(c::BenchCase, m::Symbol; λ, maxit, ρ = CMP_RHO)
    if c.family === :cine
        acq = c.trajectory === :cartesian ? _mrireco_cine_acq(cine_stack(c)) : _mrireco_nc_acq(c)
        return mrireco_dynamic(m, acq, c.smaps, c.image_size, size(c.reference, 3); λ, iterations = maxit, ρ)
    end
    # The FISTA row's `ρ` is its step size, estimated when `nothing` (see `mrireco`).
    ρm = m === :wavelet ? nothing : ρ
    if c.trajectory === :noncartesian
        m === :gridding && return mrireco_direct(_mrireco_nc_acq(c), c.smaps, c.image_size)
        return mrireco(m, () -> _mrireco_nc_acq(c), c.smaps, c.image_size; λ, iterations = maxit, ρ = ρm)
    end
    f = function (k, s)
        m === :adjoint && return mrireco_direct(_mrireco_acq(k), s, c.image_size)
        return mrireco(m, () -> _mrireco_acq(k), s, c.image_size; λ, iterations = maxit, ρ = ρm)
    end
    c.family === :multislice && return per_slice(f, c)
    return f(c.kspace, c.family === :volume ? c.smaps : cart_maps(c))
end

"""
    mrireco_direct(acq, smaps, reconSize; frames = 1) -> (time_ms, image)

MRIReco's `direct` reconstruction (per-coil images) followed by the conjugate-sensitivity coil
combination, inside the timed region: MRT's direct row includes the sensitivity adjoint. On a
non-Cartesian `acq` this is the gridding row, and `direct` applies MRIReco's own density
compensation rather than the case's ramp DCF. `frames > 1` is a cine with its frames as echoes,
returned as `(reconSize..., frames)`.
"""
function mrireco_direct(acq, smaps, reconSize; frames::Int = 1)
    nc = size(smaps, ndims(smaps))
    s = CMP_CTYPE.(smaps)
    rp = Dict{Symbol, Any}(:reco => "direct", :reconSize => reconSize, :senseMaps => _mrireco_maps(s, reconSize))
    # `direct` returns `(x, y, z or slice, echo, coil, rep)`; the maps broadcast over the frames.
    sb = reshape(s, reconSize..., 1, nc)
    t, _, x = time_reconstruction() do
        imgs = with_mrireco_blas(() -> MRIReco.reconstruction(acq, rp))
        dropdims(sum(reshape(Array(imgs), reconSize..., frames, nc) .* conj.(sb); dims = length(reconSize) + 2); dims = length(reconSize) + 2)
    end
    x = frames == 1 ? reshape(x, reconSize) : x
    return t * 1000, Array{ComplexF64}(x)
end

# ---------------------------------------------------------------- MRpro

# MRpro (PyTorch) composes each reconstruction from its operators, functionals and optimizers, as
# its own examples do; it ships no reconstruction "app" for these problems. The system is built
# from tensors in MRpro's `(other, coil, z, y, x)` layout, never from its `KData`: masking ∘ FFT ∘
# sensitivities on zero-filled Cartesian k-space, the NUFFT (`FourierOp`) ∘ sensitivities on a
# radial trajectory in grid units, a cine's frames along `other`.
#
# MRpro has no ADMM, so it joins the PDHG rows and none of the ADMM ones. Its `L1Norm` is the
# complex modulus, so the anisotropic row has MRT's penalty; the isotropic one needs the joint
# threshold over the stacked differences, `_MrproJointL1`, the counterpart of SigPy's `_JointL1`.
# Temporal TV uses a circular difference along the frames, as SigPy's does. L1-wavelet is FISTA
# (`pgd`) on the synthesis form `‖A Wᴴ z - y‖² + λ‖z‖₁`: MRpro's `WaveletOp` is a Parseval frame
# (`WᴴW = I`) with a few boundary coefficients more than pixels, whose analysis prox has no closed
# form. Global low rank is `pgd` with a Casorati singular-value threshold, `_MrproSVT`. The step size
# `1/(2‖A‖²)` of both `pgd` rows comes from a power method inside the timed region, as MRT's does.
#
# Every functional here is MRpro's `‖·‖²` without the ½, and its FFT is orthonormal; the per-toolkit
# λ calibration absorbs both.
pyexec(
    """
    import numpy as np
    import torch
    from mrpro.operators import (CartesianMaskingOp, FastFourierOp, SensitivityOp, FiniteDifferenceOp,
                                 FourierOp, WaveletOp, LinearOperatorMatrix, ProximableFunctional)
    from mrpro.operators.functionals import L1Norm, L2NormSquared
    from mrpro.algorithms.optimizers import pdhg, pgd, cg
    from mrpro.data import SpatialDimension, KTrajectory

    def _mrpro_tensor(a):
        return None if a is None else torch.from_numpy(np.ascontiguousarray(a))

    class _MrproJointL1(ProximableFunctional):
        '''lam * sum over voxels of the 2-norm along axis 0, the stacked finite differences.'''
        def __init__(self, lam):
            super().__init__()
            self.lam = lam

        def forward(self, x):
            return (self.lam * x.abs().square().sum(0).sqrt().sum(),)

        def prox(self, x, sigma=1.0):
            n = x.abs().square().sum(0, keepdim=True).sqrt()
            return (x * torch.clamp(1 - self.lam * sigma / n.clamp_min(1e-30), min=0),)

        def prox_convex_conj(self, x, sigma=1.0):
            n = x.abs().square().sum(0, keepdim=True).sqrt()
            return (x / torch.clamp(n / self.lam, min=1),)

    class _MrproSVT(ProximableFunctional):
        '''lam * nuclear norm of the (frames x voxels) Casorati matrix.'''
        def __init__(self, lam):
            super().__init__()
            self.lam = lam

        def forward(self, x):
            return (self.lam * torch.linalg.svdvals(x.reshape(x.shape[0], -1)).sum(),)

        def prox(self, x, sigma=1.0):
            u, s, vh = torch.linalg.svd(x.reshape(x.shape[0], -1), full_matrices=False)
            s = torch.clamp(s - self.lam * sigma, min=0).to(u.dtype)
            return (((u * s[None, :]) @ vh).reshape(x.shape),)

        def prox_convex_conj(self, x, sigma=1.0):
            return (x - sigma * self.prox(x / sigma, 1.0 / sigma)[0],)

    def _mrpro_system(csm, mask, kx, ky, ndim):
        S = SensitivityOp(csm)
        if kx is None:
            return CartesianMaskingOp(mask) @ FastFourierOp(dim=tuple(range(-ndim, 0))) @ S
        ny, nx = csm.shape[-2], csm.shape[-1]
        traj = KTrajectory(torch.zeros(1, 1, 1, 1, 1), ky, kx)
        return FourierOp(recon_matrix=SpatialDimension(1, ny, nx), encoding_matrix=SpatialDimension(1, ny, nx), traj=traj) @ S

    def _mrpro_solve(method, y, csm, mask, kx, ky, dcf, lam, maxit, ndim, wavelet, levels):
        y, csm, mask, kx, ky, dcf = map(_mrpro_tensor, (y, csm, mask, kx, ky, dcf))
        A = _mrpro_system(csm, mask, kx, ky, ndim)
        if method == 'adjoint':
            return A.H(y)[0].numpy()
        if method == 'gridding':
            return A.H(y * dcf)[0].numpy()
        (b,) = A.H(y)
        if method == 'cgsense':
            return cg(A.gram, b, max_iterations=maxit, tolerance=0.0)[0].numpy()
        if method in ('tv_pd', 'atv_pd', 'ttv_pd'):
            if method == 'ttv_pd':
                D = FiniteDifferenceOp(dim=(-5,), mode='forward', pad_mode='circular')
            else:
                D = FiniteDifferenceOp(dim=tuple(range(-ndim, 0)), mode='forward')
            g = _MrproJointL1(lam) if method == 'tv_pd' else L1Norm(weight=lam)
            K = LinearOperatorMatrix(((A,), (D,)))
            return pdhg(f=L2NormSquared(target=y) | g, g=None, operator=K,
                        initial_values=(torch.zeros_like(b),), max_iterations=maxit)[0].numpy()
        torch.manual_seed(0)
        L2 = 1.05 * float(A.operator_norm(torch.randn_like(b), dim=None, max_iterations=30)) ** 2
        if method == 'wavelet':
            W = WaveletOp(domain_shape=tuple(b.shape[-ndim:]), dim=tuple(range(-ndim, 0)), wavelet_name=wavelet, level=levels)
            (z0,) = W(torch.zeros_like(b))
            (z,) = pgd(f=L2NormSquared(target=y) @ A @ W.H, g=L1Norm(weight=lam), initial_value=z0,
                       stepsize=0.5 / L2, max_iterations=maxit)
            return W.H(z)[0].numpy()
        if method == 'lowrank':
            return pgd(f=L2NormSquared(target=y) @ A, g=_MrproSVT(lam), initial_value=torch.zeros_like(b),
                       stepsize=0.5 / L2, max_iterations=maxit)[0].numpy()
        raise ValueError('MRpro has no ' + method + ' here')
    """, PY,
)

"""
    mrpro_inputs(c) -> (y, csm, mask, kx, ky, dcf, ndim)

Case `c` in MRpro's `(other, coil, z, y, x)` layout — every axis of the case's `(x, y, [z,] coil,
[time])` or `(sample, spoke, coil, [time])` arrays reversed, singleton axes inserted: k-space, maps,
the per-frame sampling mask of a Cartesian case, the radial trajectory in grid units
`(1, 1, 1, spoke, sample)` with its DCF, and the number of image axes. Absent pieces are `nothing`.
"""
function mrpro_inputs(c::BenchCase)
    nx, ny = c.image_size[1:2]
    nt = c.family === :cine ? size(c.reference, 3) : 1
    nc = ncoils(c)
    smaps = c.family === :volume ? c.smaps : cart_maps(c)
    ndim = c.family === :volume ? 3 : 2
    csm = reshape(parent(permutedims(CMP_CTYPE.(smaps), ndims(smaps):-1:1)), 1, nc, (ndim == 3 ? size(smaps, 3) : 1), ny, nx)
    if c.trajectory === :cartesian
        k = parent(permutedims(CMP_CTYPE.(c.kspace), ndims(c.kspace):-1:1))            # (time?, coil, [z,] y, x)
        y = reshape(k, nt, nc, (ndim == 3 ? size(k, 2 + (nt > 1)) : 1), ny, nx)
        mask = Float32.(sum(abs, y; dims = 2) .> 0)
        return y, csm, mask, nothing, nothing, nothing, ndim
    end
    ns, nsp = size(c.traj, 2), size(c.traj, 3)
    k = parent(permutedims(CMP_CTYPE.(c.kspace), ndims(c.kspace):-1:1))                # (time?, coil, spoke, sample)
    y = reshape(k, nt, nc, 1, nsp, ns)
    kx = reshape(permutedims(Float32.(c.traj[1, :, :] .* nx), (2, 1)), 1, 1, 1, nsp, ns)
    ky = reshape(permutedims(Float32.(c.traj[2, :, :] .* ny), (2, 1)), 1, 1, 1, nsp, ns)
    dcf = reshape(permutedims(Float32.(c.dcf), (2, 1)), 1, 1, 1, nsp, ns)
    return y, csm, nothing, kx, ky, dcf, ndim
end

# MRpro's `(other, 1, z, y, x)` result → the case's image layout.
function mrpro_image(c::BenchCase, raw)
    c.family === :cine && return permutedims(Array{ComplexF64}(raw[:, 1, 1, :, :]), (3, 2, 1))
    c.family === :volume && return permutedims(Array{ComplexF64}(raw[1, 1, :, :, :]), (3, 2, 1))
    return permutedims(Array{ComplexF64}(raw[1, 1, 1, :, :]), (2, 1))
end

"""
    mrpro_run(c, method; λ, maxit) -> (time_ms, image)

MRpro's reconstruction of case `c` by `method` (see the section comment above for how each row is
composed). The inputs are handed to Python before the clock starts; the solve, the construction of
its operators and the step-size estimate are timed. A multislice case runs slice by slice. Global
low rank runs `proxgrad_budget(maxit)` iterations, as MIRT's proximal-gradient row does.
"""
function mrpro_run(c::BenchCase, m::Symbol; λ, maxit)
    if c.family === :multislice
        return per_slice(c) do k, s
            one = BenchCase(; id = c.id, family = :single_slice, trajectory = :cartesian, reference = c.reference[:, :, 1], smaps = s, kspace = k, image_size = c.image_size)
            mrpro_run(one, m; λ, maxit)
        end
    end
    y, csm, mask, kx, ky, dcf, ndim = map(_np, mrpro_inputs(c))
    it = m === :lowrank ? proxgrad_budget(maxit) : maxit
    solve = PY["_mrpro_solve"]
    t, _, raw = time_reconstruction(
        _jl(() -> solve(String(m), y, csm, mask, kx, ky, dcf, Float64(λ), it, ndim, CMP_WAVELET_NAME, CMP_WAVELET_LEVELS))
    )
    return t * 1000, mrpro_image(c, raw)
end

# ---------------------------------------------------------------- MIRT

mirt_run(c::BenchCase, m::Symbol; λ, maxit) = with_mirt_fftw(() -> _mirt_run(c, m; λ, maxit))

function _mirt_run(c::BenchCase, m::Symbol; λ, maxit)
    c.family === :multislice && return per_slice((k, s) -> mirt_recon(m, mirt_problem(k, s)...; iterations = maxit), c)
    m === :gridding && return mirt_gridding(c)
    build, y = mirt_problem(c)
    m === :lowrank && return mirt_lowrank(build, y; λ, iterations = proxgrad_budget(maxit))
    return mirt_recon(m, build, y; iterations = maxit)
end
