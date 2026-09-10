# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,src//jl:percent
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Julia 1.12.7
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # 6 — Algorithms and configuration
#
# Two knobs decide how a reconstruction runs: the *method* (what problem is solved, and with
# which solver) and the *run configuration* (scaling, output, threading, task splitting). MRT
# keeps them strictly separate — everything only a method can act on lives on the method, and
# `ReconstructionConfig` rejects such a keyword rather than silently ignoring it.
#
# **Contents**
# 1. Which solver, and why
# 2. The solvers: CG/CGNR, ISTA/FISTA, ADMM, Douglas–Rachford
# 3. `maxit`, `tol` and early stopping
# 4. Verbosity
# 5. Data scaling
# 6. Warm starts
# 7. Operator-norm and normal-operator options
# 8. Task splitting and threading
# 9. A convergence comparison
# 10. `ReconstructionConfig` and run settings

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using MriReconstructionToolbox: DEFAULT_ALGORITHMS, get_encoding_operator
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
using MIRTjim: jim
using Plots
using NamedDims
using ProximalAlgorithms: get_assumptions
using AbstractOperators: estimate_opnorm
using Random

Random.seed!(0)

nx, ny, nc = 128, 128, 8
x_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
x_noisy = x_true + 0.02f0 * randn(ComplexF32, nx, ny)
acq = AcquisitionInfo(;
    is3D = false,
    image_size = (nx, ny),
    sensitivity_maps = coil_sensitivities(nx, ny, nc),
    subsampling = create_sampling_pattern(
        VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), (nx, ny)
    ),
)
data = simulate_acquisition(x_noisy, acq)
nrmse1(x̂) = nrmse(x̂, x_true)          # one-argument closure over the ground truth

# %% [markdown]
# ## 1. Which solver, and why
#
# `IterativeReconstruction` is handed a *tuple* of candidate algorithms and takes the first one
# whose assumptions the parsed problem satisfies. `DEFAULT_ALGORITHMS` is that tuple when you say
# nothing, and each entry declares the shape of model it can take — `ProximalAlgorithms`'
# `get_assumptions` is where those declarations live, so the table below is read off the solvers
# themselves rather than copied out of them.
#
# In the shapes: `ls(Ax - b)` is a least-squares data term, `f`/`g`/`gᵢ` are arbitrary functions
# subject to the stated properties, and `Bᵢ` are linear operators (a regularizer's transform).

# %%
# One row per entry of `DEFAULT_ALGORITHMS`, in the order they are tried. The model shape comes
# from the solver's own `get_assumptions`; only the last column is prose.
const SELECTION_NOTES = Dict(
    :CGIteration => ("CG", "quadratic, square operator: least squares plus at most an L2 penalty, and 𝒜 maps image to image (single coil, fully sampled, or a KSpaceToImage model)"),
    :CGNRIteration => ("CGNR", "quadratic, any operator: the normal-equation form, so a rectangular 𝒜 is fine — this is where an unregularized or L2-only model lands"),
    :FastForwardBackwardIteration => ("FISTA", "smooth + prox: one smooth data term and exactly one term the parser can reduce to a single proximal map (wavelets, temporal Fourier, low rank)"),
    :ADMMIteration => ("ADMM", "least squares + proximable terms behind linear operators: several regularizers, or one whose transform is not tight (finite differences), so the prox cannot be composed with it"),
    :DouglasRachfordIteration => ("DouglasRachford", "two proximable terms and nothing smooth: data consistency as a constraint rather than a penalty"),
)

function algorithm_table(algorithms)
    rows = map(algorithms) do alg
        iteration_type = typeof(alg).parameters[1]
        key = nameof(iteration_type)
        name, why = get(SELECTION_NOTES, key, (string(key), "—"))
        shape = first(split(sprint(show, get_assumptions(alg)), " where "))
        return (name, strip(shape), why)
    end
    wname = maximum(length(r[1]) for r in rows)
    wshape = maximum(length(r[2]) for r in rows)
    println(rpad("algorithm", wname), " | ", rpad("model shape", wshape), " | picked when")
    println(repeat("-", wname), "-+-", repeat("-", wshape), "-+", repeat("-", 12))
    for (name, shape, why) in rows
        println(rpad(name, wname), " | ", rpad(shape, wshape), " | ", why)
    end
    return nothing
end

algorithm_table(DEFAULT_ALGORITHMS)

# %% [markdown]
# Two consequences worth spelling out:
#
# - A model with **no** non-smooth term is a plain least-squares problem, so it reaches CG or
#   CGNR — no step size, no proximal map, few iterations.
# - A single non-smooth term reaches FISTA only if the parser can evaluate its proximal map
#   directly. That works when the term's operator has a diagonal normal operator
#   (`is_AAc_diagonal` — orthogonal and tight-frame transforms: wavelets, temporal Fourier), and
#   fails for finite differences, which is why TV lands on ADMM.

# %% [markdown]
# ## 2. The solvers
#
# ### CGNR — smooth problems
#
# Least squares with an optional quadratic penalty. No step size to tune, few iterations needed.

# %%
x_cgnr = reconstruct(
    data, IterativeReconstruction(L2Image(1.0f-4); algorithm = CGNR(), maxit = 20); verbosity = Silent()
)
println("CGNR    NRMSE ", round(nrmse1(x_cgnr), digits = 4))

# %% [markdown]
# ### ISTA / FISTA — one non-smooth term
#
# Proximal gradient, with (FISTA) and without (ISTA) Nesterov acceleration. The acceleration is
# free, so FISTA is the one in `DEFAULT_ALGORITHMS`; `ISTA()` has to be named explicitly.

# %%
x_ista = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); algorithm = ISTA(), maxit = 60); verbosity = Silent()
)
x_fista = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); algorithm = FISTA(), maxit = 60); verbosity = Silent()
)
println("ISTA    NRMSE ", round(nrmse1(x_ista), digits = 4))
println("FISTA   NRMSE ", round(nrmse1(x_fista), digits = 4))

# %% [markdown]
# ### ADMM — several terms, or a non-tight operator
#
# Splits the problem into pieces with easy proximal maps. Slower per iteration, but it is what
# makes TV and multi-term models solvable.

# %%
x_admm = reconstruct(
    data,
    IterativeReconstruction(L1Wavelet2D(1.5f-3), TotalVariation2D(5.0f-4); algorithm = ADMM(), maxit = 50);
    verbosity = Silent()
)
println("ADMM    NRMSE ", round(nrmse1(x_admm), digits = 4))

# %% [markdown]
# ### Douglas–Rachford — two proximable terms
#
# The natural solver when data consistency is a *constraint* rather than a penalty:
# `HardConsistency()` projects onto $\{x : \mathcal{A}x = y\}$, and the regularizer supplies the
# second proximal map.
#
# The NRMSE below is poor, and that is a property of the *model*, not of the solver: an equality
# constraint forces the reconstruction to reproduce the measured k-space including its noise, and
# under heavy SENSE undersampling the projection amplifies that noise along the directions where
# $\mathcal{A}\mathcal{A}^*$ is nearly singular. Notebook 05 §4 measures this in detail (a
# better-converged projection makes it worse, not better) and says where `HardConsistency` does
# belong. Douglas–Rachford itself is fine — give it a model whose two proximal maps are
# well-conditioned.

# %%
x_dr = reconstruct(
    data,
    IterativeReconstruction(
        L1Wavelet2D(2.0f-3);
        fidelity = HardConsistency(maxit = 20), algorithm = DouglasRachford(), maxit = 40
    );
    verbosity = Silent()
)
println("DR + hard consistency NRMSE ", round(nrmse1(x_dr), digits = 4))

# %%
side_by_side(
    x_cgnr, x_fista, x_admm;
    titles = ("CGNR (L2)", "FISTA (wavelet)", "ADMM (wavelet+TV)")
)

# %% [markdown]
# ### Letting MRT choose
#
# **The recommended form is to leave `algorithm` out entirely.** The default tuple already covers
# every model this package can build, in a sensible order, and the choice then tracks whatever
# regularizers you happen to have combined.

# %%
x_auto = reconstruct(
    data,
    IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 60);   # no `algorithm`
    verbosity = Silent()
)
println("auto-selected NRMSE ", round(nrmse1(x_auto), digits = 4), "  (equals FISTA: ", x_auto ≈ x_fista, ")")

# %% [markdown]
# Passing a tuple is the second-choice form, and it is for *restricting* or *extending* the
# candidate set rather than for picking a solver (a bare `algorithm = FISTA()` does that).
#
# - **Restrict** when more than one default applies and you want the other one. An ℓ₁-wavelet
#   model is accepted by FISTA *and* by ADMM; `algorithm = (ADMM(),)` keeps the automatic
#   behaviour of erroring out on a model it cannot parse, while making sure the model that *can*
#   go to FISTA does not.
# - **Extend** when the solver you want is not in the defaults. `ISTA()` is not: the
#   unaccelerated proximal gradient is only reachable by naming it. That matters when the
#   proximal map changes from iteration to iteration — `LocallyLowRank(; shift = :random)` in
#   notebook 07 is the example — because momentum and line search both assume a fixed objective.

# %%
x_forced = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); algorithm = (ADMM(),), maxit = 60); verbosity = Silent()
)
x_extended = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); algorithm = (ISTA(), ADMM()), maxit = 60); verbosity = Silent()
)
println("restricted to ADMM      NRMSE ", round(nrmse1(x_forced), digits = 4))
println(
    "extended with ISTA      NRMSE ", round(nrmse1(x_extended), digits = 4),
    "  (equals ISTA: ", x_extended ≈ x_ista, ")"
)

# %% [markdown]
# ## 3. `maxit`, `tol` and early stopping
#
# Both belong to the *method*, because only a method has iterations.
#
# - **`maxit` is the iteration budget** — an upper bound on the work the solve may do, and the
#   guarantee that it terminates. Default `100`.
# - **`tol` is a relative tolerance that controls early stopping** — how close to a fixed point
#   the iterate must be before the solver stops ahead of the budget. Default `1e-4`.
#
# Mechanically: MRT's `tol` is *relative*, `ProximalAlgorithms`' is absolute, and MRT converts
# between them. The absolute threshold handed to the solver is
#
# ```julia
# max(10 * eps(real(eltype(x₀))), tol * maximum(abs, x₀))
# ```
#
# where `x₀` is the initial guess (the direct reconstruction unless you pass one). `tol = 0`
# switches the test off entirely, which is what you want when comparing convergence curves.
#
# > **Note.** Setting either to `nothing` leaves the corresponding keyword out of the `solve`
# > call altogether, so the algorithm's own value survives. That is the only way
# > `algorithm = FISTA(maxit = 500)` becomes reachable: passed together with a `maxit` on the
# > method, the method's value would be merged in last and overwrite it.

# %%
x_loose = reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 200, tol = 1.0f-3); verbosity = Silent())
x_tight = reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 200, tol = 1.0f-6); verbosity = Silent())
println("tol 1e-3 NRMSE ", round(nrmse1(x_loose), digits = 4), "   (stopped early)")
println("tol 1e-6 NRMSE ", round(nrmse1(x_tight), digits = 4), "   (ran further)")

# %% [markdown]
# The looser tolerance gives the *lower* NRMSE, which is not a mistake: stopping early is itself
# a form of regularization, and the extra iterations the tighter run buys are spent fitting the
# noise the data term is asking it to fit. `tol` controls how faithfully the *objective* is
# minimized; whether that objective's minimizer is the best image is λ's job.

# %%
# Defer to the algorithm's own iteration count.
x_alg = reconstruct(
    data,
    IterativeReconstruction(L1Wavelet2D(2.0f-3); algorithm = FISTA(maxit = 30), maxit = nothing);
    verbosity = Silent()
)
println("algorithm-owned maxit: NRMSE ", round(nrmse1(x_alg), digits = 4))

# %%
# Passing an iteration parameter to `reconstruct` is an error rather than being ignored.
try
    reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3)); maxit = 10)
catch e
    println(sprint(showerror, e))
end

# %% [markdown]
# ### What early stopping actually tests
#
# **The quantity.** Every solver compares one scalar against the threshold, and in each case it
# measures *how far the iterate still is from a fixed point*, not how good the image is:
#
# | algorithm | quantity compared against the threshold |
# |---|---|
# | ISTA / FISTA | `‖x - z‖∞ / γ` — the proximal-gradient step divided by the step size, i.e. the fixed-point residual |
# | Douglas–Rachford | the same residual form, divided by its `γ` |
# | ADMM | the iterate change `‖Δx‖`, *and* the primal residual against `tol·εᵖʳⁱ`, *and* the dual residual against `tol·εᵈᵘᵃ` — all three must hold |
# | CG / CGNR | `‖r‖₂`, the norm of the (normal-equation) residual |
#
# None of these is the objective value, and none of them is an error against a ground truth — the
# solver has no ground truth. A small residual says the iterate has stopped moving; whether that
# point is a *good image* is the regularizer's business, not the tolerance's.
#
# **When it is evaluated.** At the end of every iteration, after the update has been applied and
# after any `on_iteration` callback has fired, and before the next iteration begins. The check is
# `k >= maxit || stop(iter, state)`, so the budget is tested first and the two can coincide.
#
# **Why relative to `maximum(abs, x₀)`.** The residual carries the units of the image. The same
# absolute number means "converged" for data scaled so that the image peaks near 1, and "nowhere
# near" for raw scanner data peaking at 10⁶. Dividing by the initial estimate's own peak turns
# `tol` into a *fraction of the image's dynamic range*, so a `tol` tuned on one dataset transfers
# to the next. The `10 * eps` floor stops the threshold from dropping below the level where the
# residual is floating-point noise and the test could never pass.
#
# **When it never triggers.** Nothing happens: the loop runs the full `maxit` and returns the
# last iterate. There is no warning and no error — it is a perfectly ordinary outcome — but it
# means the *budget*, not the tolerance, decided the answer, and a larger `maxit` would have
# changed it.
#
# **Telling the two apart from the output.** `Verbose` prints one row every `freq` iterations
# *and always one final row at the iteration the loop actually stopped on*. Read the iteration
# index in that last row: equal to `maxit` means the budget ran out; smaller than `maxit` — and
# usually off the printing grid — means the tolerance fired.

# %%
println("tol = 1e-3, maxit = 200 — watch the last row's index:")
reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 200, tol = 1.0f-3);
    verbosity = Verbose(; timing = false)
);

# %%
println("tol = 0, maxit = 20 — the tolerance is switched off, so the budget decides:")
reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 20, tol = 0.0);
    verbosity = Verbose(; timing = false, freq = 5)
);

# %% [markdown]
# ## 4. Verbosity
#
# Three mutually exclusive output modes, given to `reconstruct` (they are run settings, not
# method parameters). `verbosity` also accepts `true`/`false` and the symbols `:verbose`,
# `:progress`, `:silent`.

# %%
reconstruct(data, IterativeReconstruction(L2Image(1.0f-4); maxit = 5); verbosity = Silent());

# %%
reconstruct(data, IterativeReconstruction(L2Image(1.0f-4); maxit = 5); verbosity = ProgressBar());

# %% [markdown]
# `freq` is **optional**. Left out, MRT derives a printing frequency from the method's `maxit`
# (roughly twenty rows over the run, rounded to one of 1, 5, 10, 20, 50, 100), which is what you
# want almost always. Pass it only to override that: `freq = 1` for every iteration, `freq = 0`
# for a single end-of-run summary line, `freq = -1` to drop the solver's output while keeping
# MRT's own phase log.

# %%
println("--- Verbose(): frequency chosen from maxit = 60 ---")
reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 60); verbosity = Verbose(; timing = false));

# %%
println("--- Verbose(; freq = 1): overridden, every iteration ---")
reconstruct(data, IterativeReconstruction(L2Image(1.0f-4); maxit = 5); verbosity = Verbose(; freq = 1, timing = false));

# %%
# The log can be redirected anywhere — here into a vector, e.g. for a dashboard or a test.
messages = String[]
reconstruct(
    data, IterativeReconstruction(L2Image(1.0f-4); maxit = 5);
    verbosity = Verbose(; printfunc = (args...) -> push!(messages, string(args...)))
)
println(length(messages), " messages captured; first: ", first(messages))

# %% [markdown]
# ## 5. Data scaling
#
# The absolute magnitude of the k-space is not neutral, for two reasons that have nothing to do
# with each other:
#
# 1. **λ is scale-dependent.** The objective is $\tfrac12\|\mathcal{A}x - y\|^2 + \lambda R(x)$.
#    Multiply the data by $c$ and the data term grows like $c^2$ while an ℓ₁-type $R$ grows like
#    $c$, so the balance between them moves and the λ you tuned no longer means the same thing.
#    Bringing every dataset to a common scale first is what makes a λ transferable between
#    scans, scanners and vendors.
# 2. **The arithmetic has finite range.** These reconstructions run in `Float32`, and the data
#    term squares whatever comes off the scanner; raw k-space that lives around `1e-6` or `1e6`
#    spends that range on the exponent instead of on the image.
#
# The stopping tolerance, by contrast, is already scale-free: it is relative to
# `maximum(abs, x₀)` (§3), so it needs no help from the scaling.
#
# `BartScaling` divides by a high quantile of the direct reconstruction (the convention BART
# uses), `MeasurementBasedScaling` derives the factor from the measurements,
# `FixedScaling` takes a number you supply, and `NoScaling` leaves the data alone. The output is
# scaled back unless you ask otherwise, so the choice does not change the units you get out — it
# changes the units the solver works in, and therefore what λ means.

# %%
for scaling in (NoScaling(), BartScaling(), MeasurementBasedScaling())
    x̂ = reconstruct(
        data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40);
        scaling = scaling, verbosity = Silent()
    )
    println(
        rpad(string(typeof(scaling).name.name), 24), " NRMSE ", round(nrmse1(x̂), digits = 4),
        "   max|x| ", round(maximum(abs, x̂), digits = 3)
    )
end

# %% [markdown]
# The spread is small here because the simulated data is already close to unit scale, so there
# is little for a scaling to fix. Multiply the k-space by 1000 — the kind of factor that
# separates one scanner's raw units from another's — and point 1 becomes unmissable: with
# `NoScaling` the same λ now under-regularizes badly, while `BartScaling` returns bit-for-bit the
# reconstruction it gave on the original data.

# %%
for factor in (1.0f0, 1.0f3)
    data_scaled = AcquisitionInfo(data; kspace_data = data.kspace_data .* factor)
    for scaling in (NoScaling(), BartScaling())
        x̂ = reconstruct(
            data_scaled, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40);
            scaling = scaling, verbosity = Silent()
        )
        println(
            "k-space × ", rpad(factor, 8), rpad(string(typeof(scaling).name.name), 14),
            " NRMSE ", round(nrmse1(x̂ ./ factor), digits = 5)
        )
    end
end

# %%
# Keep the internally scaled units instead of mapping back.
x_scaled_back = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40);
    scaling = BartScaling(), verbosity = Silent()
)
x_unscaled = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40);
    scaling = BartScaling(), disable_inverse_scale_output = true, verbosity = Silent()
)
println("max|x| scaled back:  ", round(maximum(abs, x_scaled_back), digits = 3))
println("max|x| left scaled:  ", round(maximum(abs, x_unscaled), digits = 3))

# %% [markdown]
# ## 6. Warm starts
#
# `x₀` seeds the solver. Useful for parameter sweeps, staged reconstructions, and the non-convex
# terms whose result depends on where they start. It is also what the stopping threshold is
# measured against (§3), so a warm start changes the tolerance as well as the starting point.

# %%
x_stage1 = reconstruct(data, IterativeReconstruction(L1Wavelet2D(5.0f-3); maxit = 40); verbosity = Silent())
x_stage2 = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 40); x₀ = x_stage1, verbosity = Silent()
)
println("stage 1 NRMSE ", round(nrmse1(x_stage1), digits = 4))
println("stage 2 NRMSE ", round(nrmse1(x_stage2), digits = 4))

# %% [markdown]
# ## 7. Operator-norm and normal-operator options
#
# ### What `‖𝒜‖` is used for
#
# MRT estimates $\|\mathcal{A}\|$ with a 20-step power iteration and hands the algorithm
# $L_f = n\|\mathcal{A}\|^2$ — the Lipschitz constant of the data term's gradient, with $n$ the
# number of optimization variables (one, unless the model has `Component`s). It does **not**
# rescale $\mathcal{A}$: that used to be the implementation, and it silently multiplied the
# effective λ and the returned image by $\|\mathcal{A}\|$. λ therefore means what it says, in the
# data's own units.
#
# ### Which algorithms actually need it
#
# | algorithm | needs `Lf`? | what it uses it for |
# |---|---|---|
# | ISTA / FISTA | yes | the step size `γ = 1/Lf`; without it the algorithm backtracks to find one |
# | Douglas–Rachford | yes | its default `γ = 1/Lf` |
# | ADMM | **no** | its step is the penalty `ρ`, chosen adaptively; `Lf` is discarded |
# | CG / CGNR | **no** | Krylov methods are scale invariant and derive everything themselves |
#
# MRT already skips the estimate where it can prove it is useless — a *pure, unregularized*
# CG/CGNR solve — so the flags below matter only in the remaining cases. Note the gap in that
# rule: an ADMM run still pays for an estimate it will discard, which is a case where
# `disable_operator_normalization = true` costs nothing at all.

# %%
𝒜 = get_encoding_operator(data)
estimate_opnorm(𝒜)                                                       # warm up
t_estimate = minimum(@elapsed(estimate_opnorm(𝒜)) for _ in 1:5)
println("the 20-step estimate costs ", round(1000 * t_estimate, digits = 1), " ms")

for kwargs in ((;), (; disable_operator_normalization = true))
    m = IterativeReconstruction(TotalVariation2D(5.0f-4); algorithm = ADMM(), maxit = 30, kwargs...)
    x̂ = reconstruct(data, m; verbosity = Silent())
    println(
        rpad(isempty(kwargs) ? "ADMM, default" : "ADMM, no estimate", 20),
        " NRMSE ", round(nrmse1(x̂), digits = 5)
    )
end

# %% [markdown]
# ### The three options
#
# **`exact_opnorm = true`** — replace `estimate_opnorm`'s 20 power iterations with
# `LinearAlgebra.opnorm`, run to convergence.
# *Costs* a longer setup: the power iteration keeps applying $\mathcal{A}$ and
# $\mathcal{A}^*$ until it stops moving instead of stopping at 20.
# *Reach for it* when you need the true constant rather than a lower bound — the estimate
# converges from below, so `1/Lf` is a slightly **larger** step than the theory guarantees, and
# on an awkwardly conditioned operator that can cost monotonicity.
#
# **Supplying the constant yourself** — `disable_operator_normalization` is not the only
# alternative to estimating. Because MRT fills `Lf` in only when the algorithm does not already
# carry one, an `Lf` you pass to the algorithm wins; combine it with
# `disable_operator_normalization = true` and the estimate is skipped as well.
# *Costs* nothing, and *reach for it* whenever you already know $\|\mathcal{A}\|$ — a parameter
# sweep over λ on one fixed operator computes it once and reuses it.
#
# **`disable_operator_normalization = true`** — skip the estimate and pass no `Lf`. The
# forward-backward iteration then switches to `adaptive = true`: it seeds `γ` from its own cheap
# lower bound on the smoothness constant and re-checks a descent condition every iteration,
# halving `γ` whenever the check fails.
# *Costs* one extra evaluation of the smooth term per iteration — the descent check — rather
# than one power iteration up front, so it trades a fixed setup cost for a per-iteration one.
# *Reach for it* for an operator whose norm you cannot estimate cheaply, for the ADMM and
# CG cases above where `Lf` is discarded anyway, or to check that a suspicious step size is not
# the cause of a bad reconstruction.
#
# **`disable_normalop_optimization = true`** — the substitution this disables is not about
# "least-squares models" in general. MRT builds the data term as `normalop_ls(𝒜x - y)` — which
# precomputes $\mathcal{A}^*\mathcal{A}$ once and applies that instead of $\mathcal{A}$ followed
# by $\mathcal{A}^*$ — exactly when **all three** hold: the fidelity is `L2Loss`, the model has a
# single image variable (not `Component`s), and no regularizer contributed auxiliary variables
# (`TotalGeneralizedVariation2D` does, and then the stored normal operator would not span the
# solver's variable tuple). Otherwise plain `ls(𝒜x - y)` is used and the flag changes nothing.
# *Costs*, when disabled, an extra operator application per gradient.
# *Reach for it* when debugging a custom operator whose adjoint or normal operator you suspect,
# since the two forms should agree to round-off.

# %%
# `Lf = n‖𝒜‖²` with n = 1: computed once here (the same `𝒜` as above), then handed to the
# algorithm.
L = estimate_opnorm(𝒜)
println("‖A‖ = ", round(L, digits = 6))

x_default = reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40); verbosity = Silent())
x_manual = reconstruct(
    data,
    IterativeReconstruction(
        L1Wavelet2D(2.0f-3);
        algorithm = FISTA(Lf = L^2), disable_operator_normalization = true, maxit = 40
    );
    verbosity = Silent()
)
println("hand-supplied Lf reproduces the default run exactly: ", x_manual ≈ x_default)

# %%
for (label, kwargs) in (
        ("default", (;)),
        ("exact_opnorm", (; exact_opnorm = true)),
        ("no normalization", (; disable_operator_normalization = true)),
        ("no normal-op", (; disable_normalop_optimization = true)),
    )
    m = IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40, kwargs...)
    reconstruct(data, m; verbosity = Silent())                                   # warm up
    t = minimum(@elapsed(reconstruct(data, m; verbosity = Silent())) for _ in 1:3)
    x̂ = reconstruct(data, m; verbosity = Silent())
    println(rpad(label, 20), " NRMSE ", round(nrmse1(x̂), digits = 5), "   ", round(t, digits = 3), " s")
end

# %% [markdown]
# Two rows in that table are worth explaining, because neither is obvious. (Timings on a shared
# machine move by tens of percent between runs; the *reasons* below are the part to keep.)
#
# **Why `exact_opnorm` costs so much wall-clock time.** Not because of the iterations — `maxit`
# is unchanged — but because of the setup. The 20-step estimate took a few tens of milliseconds
# in the cell above; running the power iteration to convergence takes roughly an order of
# magnitude longer, while the whole 40-iteration solve is only a couple of hundred
# milliseconds. The setup, not the solve, is what grew.
#
# **Why "no normalization" costs extra time too, and where it goes.** Not to a longer setup —
# there is none — but to the loop. Without `Lf` the iteration runs in adaptive mode and performs
# a descent check every iteration, one extra evaluation of the smooth term each time. In this
# problem the check always passes and `γ` never actually shrinks (it stays at the value the
# algorithm seeded it with, close to the `1/Lf` the estimate would have given), so the price is
# paid for information MRT could have supplied once.
#
# **Why `exact_opnorm` gives the *worse* NRMSE at this budget.** It does not converge to a worse
# image — it converges to the same one, a little more slowly. `estimate_opnorm` stops after 20
# power iterations and so returns a slight **under**-estimate of `‖𝒜‖` (here about 0.6 % low),
# which makes `Lf` too small and therefore `γ = 1/Lf` slightly *too large*. A larger step means
# more progress per iteration, and at a truncated `maxit = 40` more progress is a lower NRMSE.
# The exact norm gives the theoretically safe, slightly smaller step, and it lands a little
# further back along the same trajectory. λ is not involved: since `‖𝒜‖` no longer rescales
# `𝒜`, the objective being minimized is identical in both runs. The cell below checks that
# directly by giving both enough iterations to converge.

# %%
for (label, kwargs) in (("default", (;)), ("exact_opnorm", (; exact_opnorm = true)))
    x̂ = reconstruct(
        data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 300, tol = 0.0, kwargs...);
        verbosity = Silent()
    )
    println(rpad(label, 14), " NRMSE after 300 iterations ", round(nrmse1(x̂), digits = 5))
end

# %% [markdown]
# ## 8. Task splitting and threading
#
# When the data has batch dimensions that *nothing couples* — slices, contrasts, echoes, or time
# if there is no temporal regularizer — `reconstruct` splits the problem into one independent
# solve per batch element and runs them in parallel. This is **task splitting**, and it is a run
# setting: `task_executor` chooses how the tasks are run, `disable_task_splitting` turns the
# whole mechanism off.
#
# (Do not confuse it with *image decomposition* — `Component`, L+S — which splits one image into
# additive parts inside a single solve. Notebook 07 covers that.)

# %%
n_ms, nslices, nc_ms = 128, 32, 4
vol = create_shepp_logan_phantom(n_ms, n_ms, nslices; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps_ms = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(n_ms, n_ms, nc_ms))

acq_ms = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky, :coil, :slice)}(zeros(ComplexF32, n_ms, n_ms, nc_ms, nslices));
    is3D = false, sensitivity_maps = smaps_ms
)
data_ms = simulate_acquisition(NamedDimsArray{(:x, :y, :slice)}(vol), acq_ms)
println("multi-slice k-space: ", size(data_ms.kspace_data), " ", dimnames(data_ms.kspace_data))
println("Julia threads: ", Threads.nthreads())

# %% [markdown]
# 32 slices of 128² with 4 coils, 150 FISTA iterations each: large enough that the per-slice
# solve dominates the fork/join overhead, which a 64² × 8-slice problem does not.
#
# Timings on a shared machine swing by 30–60 % from run to run, so each configuration below is
# measured three times and the **best** time is reported; treat the ratios, not the absolute
# seconds, as the result.

# %%
method_ms = IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 150, tol = 0.0)
warmup_ms = IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 2)

function best_of(kwargs; reps = 3)
    reconstruct(data_ms, warmup_ms; verbosity = Silent(), kwargs...)   # compile everything first
    times = Float64[]
    local x̂
    for _ in 1:reps
        push!(times, @elapsed x̂ = reconstruct(data_ms, method_ms; verbosity = Silent(), kwargs...))
    end
    return minimum(times), x̂
end

t_par, x_par = best_of((; task_executor = MultiThreadingExecutor()))
t_seq, x_seq = best_of((; task_executor = SequentialExecutor()))
t_none, x_none = best_of((; disable_task_splitting = true))

println("split, threaded over slices : ", round(t_par, digits = 2), " s")
println("split, one slice at a time  : ", round(t_seq, digits = 2), " s   (", round(t_seq / t_par, digits = 2), "× slower)")
println("not split at all            : ", round(t_none, digits = 2), " s   (", round(t_none / t_par, digits = 2), "× slower)")
println("same answer all three: ", x_par ≈ x_seq && x_par ≈ x_none)

# %% [markdown]
# The unsplit run is the slowest of the three even though it is doing the same arithmetic: it
# solves one big problem whose iterate is the whole stack, so every slice is dragged along until
# the *last* one converges, and the operators work on 32× larger arrays.

# %% [markdown]
# ### Which dimensions can be split
#
# `get_affected_dims(reg, acq_or_nothing, image_dims)` is the interface function that decides
# this, and it is what a custom regularizer implements. The rule is a single line of
# `get_task_splitting_plan`: start from the non-Fourier image dimensions and remove every
# dimension any term in the model affects. What is left over is split.

# %%
using MriReconstructionToolbox: get_affected_dims

image_dims = (:x, :y, :slice, :time)
batch_dims = (:slice, :time)      # the non-Fourier dimensions of this layout

regularizers = (
    L1Image(1.0f-3),
    L1Wavelet2D(1.0f-3),
    TotalVariation2D(1.0f-3),
    NonNegative(),
    L1Wavelet3D(1.0f-3),
    TotalVariation3D(1.0f-3),
    L1TemporalFourier(1.0f-2; time_dim = :time),
    TemporalTotalVariation(1.0f-2; time_dim = :time),
    LowRank(1.0f-1; time_dim = :time),
    LocallyLowRank(1.0f-1; block_size = 8, time_dim = :time),
    L0Image(; count = 200),
)

println(rpad("regularizer", 26), rpad("couples", 26), "leaves splittable")
println(repeat("-", 78))
for reg in regularizers
    coupled = get_affected_dims(reg, nothing, image_dims)
    splittable = Tuple(d for d in batch_dims if d ∉ coupled)
    println(
        rpad(string(typeof(reg).name.name), 26),
        rpad(isempty(coupled) ? "(nothing)" : string(coupled), 26),
        isempty(splittable) ? "(nothing)" : string(splittable)
    )
end

# %% [markdown]
# Reading the rows, and *why* each one couples what it does:
#
# | term | couples | because |
# |---|---|---|
# | `L1Image`, `NonNegative` | nothing at all | the penalty is a sum over voxels; every voxel is independent of every other, so both `:slice` and `:time` stay splittable |
# | `L1Wavelet2D`, `TotalVariation2D` | `:x`, `:y` | a 2D transform mixes neighbouring pixels within a frame, and nothing across frames |
# | `L1Wavelet3D`, `TotalVariation3D` | `:x`, `:y`, `:slice` | the third axis of the transform is the slice axis, so slices can no longer be solved apart — `:time` still can |
# | `L1TemporalFourier`, `TemporalTotalVariation` | `:time` | the penalty is defined on differences (or a Fourier transform) *along* time, so a frame's value constrains its neighbours' — slices stay independent |
# | `LowRank`, `LocallyLowRank` | everything | the Casorati matrix is space × time; a nuclear norm on it is a joint property of all voxels and all frames at once, and cannot be evaluated on a piece of it |
# | `L0Image(; count = k)` | everything | a *budget* of k non-zeros is a statement about the whole coefficient array — split it in two and each half would get its own budget of k. The threshold form, `L0Image(; threshold = λ)`, is a per-voxel test and couples nothing |
#
# Note that the coupled dimensions include image dimensions like `:x` and `:y`. Those were never
# candidates for splitting in the first place (they are Fourier-encoded), so a 2D regularizer
# leaves both batch dimensions free.

# %% [markdown]
# ### Threading notes
#
# MRT parallelizes *across* slices and keeps each slice's work single-threaded, because a 128²
# slice is small enough that splitting it costs more than it saves; a low-rank prox, whose SVDs
# are level-3 BLAS, is the documented exception and keeps its threaded budget. The library-level
# thread pools underneath (BLAS, FFTW, NFFT) are managed for you during the solve — do not call
# `BLAS.set_num_threads` yourself.
#
# Two things are yours to set:
#
# - `julia -t N` with `N` = the number of physical cores you actually have (and, on Slurm,
#   `--cpus-per-task` to match, plus an explicit `--mem`).
# - `export KMP_BLOCKTIME=0` **before** starting Julia, if you use MKL. It cannot be set from
#   inside Julia, and without it MKL's spinning worker threads crowd out the reconstruction.

# %%
using LinearAlgebra: BLAS
using FFTW

@show Threads.nthreads()
@show BLAS.get_num_threads()
@show FFTW.get_num_threads()
@show get(ENV, "KMP_BLOCKTIME", "unset")
@show MriReconstructionToolbox.serial_blas_threshold_bytes()

# %% [markdown]
# ## 9. A convergence comparison
#
# `on_iteration` calls back once per solver iteration with the current image estimate, already
# inverse-scaled and in the shape `reconstruct` will return. `IterationTrace(reduction)` is the
# collector to use: it applies `reduction` to that estimate and records the result together with
# the iteration index and a wall-clock reading from a monotonic clock started *after* the
# operator build and the operator-norm estimate. Three runs, three traces, and the NRMSE is
# computed after every single iteration rather than by re-solving at a ladder of budgets.
#
# `tol = 0` matters here: with the default tolerance a solver would stop early and truncate its
# own curve.

# %%
traces = Dict{String, IterationTrace}()

for (label, alg) in (("ISTA", ISTA()), ("FISTA", FISTA()), ("ADMM", ADMM()))
    trace = IterationTrace(nrmse1)
    reconstruct(
        data,
        IterativeReconstruction(
            L1Wavelet2D(2.0f-3); algorithm = alg, maxit = 60, tol = 0.0, on_iteration = trace
        );
        verbosity = Silent()
    )
    traces[label] = trace
    println(
        rpad(label, 6), " ", length(trace.values), " iterations in ",
        round(trace.times[end], digits = 2), " s   final NRMSE ", round(trace.values[end], digits = 4)
    )
end

# %% [markdown]
# **Plot it against both axes.** One ADMM iteration costs several times what one FISTA
# iteration costs — it solves an inner linear system and updates a set of dual variables every
# step — as the printed times above show for the same iteration count. A plot against iteration
# number charges every algorithm the same price for a step, so it flatters whichever does the
# most work per step; a plot against wall-clock time is what you actually pay for. Neither plot
# alone is the answer, which is why the figure has both.

# %%
p_iter = plot(; xlabel = "iteration", ylabel = "NRMSE", yscale = :log10, title = "per iteration")
p_time = plot(; xlabel = "wall-clock time (s)", ylabel = "NRMSE", yscale = :log10, title = "per second")
for label in ("ISTA", "FISTA", "ADMM")
    trace = traces[label]
    plot!(p_iter, trace.iterations, trace.values; label = label, lw = 2)
    plot!(p_time, trace.times, trace.values; label = label, lw = 2)
end
plot(p_iter, p_time; layout = (1, 2), size = (950, 380))

# %% [markdown]
# `trace.metrics` carries whatever the algorithm itself computed, so the same run also yields the
# solver's own convergence diagnostics — and which fields exist is a property of the algorithm,
# not something to guess at.

# %%
for label in ("ISTA", "FISTA", "ADMM")
    println(rpad(label, 6), " metric fields: ", keys(traces[label].metrics[1]))
end

# %%
plot(
    [m.fixed_point_residual for m in traces["FISTA"].metrics];
    label = "FISTA fixed-point residual", lw = 2, yscale = :log10, xlabel = "iteration"
)
plot!([m.primal_residual for m in traces["ADMM"].metrics]; label = "ADMM primal residual", lw = 2)
plot!(
    [m.dual_residual for m in traces["ADMM"].metrics];
    label = "ADMM dual residual", lw = 2, size = (700, 380), ylabel = "residual"
)

# %% [markdown]
# ## 10. `ReconstructionConfig` and run settings
#
# Everything in §4, §5 and §8 — and only those — is a *run setting*: it describes how a
# reconstruction is executed, not what problem is solved. `ReconstructionConfig` bundles them
# into one reusable object:
#
# | field | default | section |
# |---|---|---|
# | `verbosity` | `Verbose()` | §4 |
# | `scaling` | `BartScaling()` | §5 |
# | `disable_inverse_scale_output` | `false` | §5 |
# | `threaded` | `Threads.nthreads() > 1` | §8 |
# | `task_executor` | `nothing` (chosen from the problem size) | §8 |
# | `disable_task_splitting` | `false` | §8 |
#
# The method-owned parameters stay on the method and are deliberately *rejected* here rather
# than ignored: `maxit`, `tol`, `algorithm`, `on_iteration` (§3, §9), and the operator-norm
# switches `exact_opnorm`, `disable_operator_normalization`, `disable_normalop_optimization`
# (§7), all of which are constructor keywords of `IterativeReconstruction`.

# %%
config = ReconstructionConfig(;
    verbosity = Silent(),
    scaling = BartScaling(),
    task_executor = SequentialExecutor(),
)

x1 = reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 30); config = config)
x2 = reconstruct(data, IterativeReconstruction(TotalVariation2D(1.0f-3); maxit = 30); config = config)
println("reused config for two methods: ", round(nrmse1(x1), digits = 4), ", ", round(nrmse1(x2), digits = 4))

# %%
# An existing config can be extended, and individual keywords still win over it per call.
config_loud = ReconstructionConfig(config; verbosity = Verbose())
x3 = reconstruct(
    data, IterativeReconstruction(L2Image(1.0f-4); maxit = 5);
    config = config_loud, verbosity = Silent()
)
println("silenced despite a verbose config")

# %%
# A method-owned keyword aimed at the run settings says where it belongs, rather than being
# quietly dropped.
try
    reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3)); on_iteration = IterationTrace())
catch e
    println(sprint(showerror, e))
end

# %% [markdown]
# ## Environment

# %%
print_versions()
