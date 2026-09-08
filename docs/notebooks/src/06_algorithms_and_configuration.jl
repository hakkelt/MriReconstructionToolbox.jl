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
# # 5 — Algorithms and configuration
#
# Two knobs decide how a reconstruction runs: the *method* (what problem is solved, and with
# which solver) and the *run configuration* (scaling, output, threading, decomposition). MRT
# keeps them strictly separate — everything only a method can act on lives on the method.
#
# **Contents**
# 1. Which solver, and why
# 2. The solvers: CG/CGNR, ISTA/FISTA, ADMM, Douglas–Rachford
# 3. `maxit` and `tol`
# 4. Verbosity
# 5. `ReconstructionConfig` and run settings
# 6. Data scaling
# 7. Warm starts
# 8. Operator-norm and normal-operator options
# 9. A convergence comparison

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using MriReconstructionToolbox: DEFAULT_ALGORITHMS
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
using MIRTjim: jim
using Plots
using LinearAlgebra: norm
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
nrmse(x̂) = norm(abs.(x̂) - abs.(x_true)) / norm(abs.(x_true))

# %% [markdown]
# ## 1. Which solver, and why
#
# Left alone, `IterativeReconstruction` picks from `DEFAULT_ALGORITHMS` based on the structure of
# the problem:
#
# ```
# smooth objective (no ℓ₁, TV, …)?
# ├─ yes → CG / CGNR
# └─ no  → exactly one non-smooth term whose operator has a diagonal normal operator?
#          ├─ yes → FISTA
#          └─ no  → ADMM
# ```
#
# "Diagonal normal operator" (`is_AAc_diagonal`) covers orthogonal and tight-frame transforms —
# wavelets, temporal Fourier — but not finite differences, which is why TV lands on ADMM.

# %%
DEFAULT_ALGORITHMS

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
println("CGNR    NRMSE ", round(nrmse(x_cgnr), digits = 4))

# %% [markdown]
# ### ISTA / FISTA — one non-smooth term
#
# Proximal gradient, with (FISTA) and without (ISTA) Nesterov acceleration. The acceleration is
# free, so FISTA is the default of the pair.

# %%
x_ista = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); algorithm = ISTA(), maxit = 60); verbosity = Silent()
)
x_fista = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); algorithm = FISTA(), maxit = 60); verbosity = Silent()
)
println("ISTA    NRMSE ", round(nrmse(x_ista), digits = 4))
println("FISTA   NRMSE ", round(nrmse(x_fista), digits = 4))

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
println("ADMM    NRMSE ", round(nrmse(x_admm), digits = 4))

# %% [markdown]
# ### Douglas–Rachford — two proximable terms
#
# The natural solver when data consistency is a *constraint* rather than a penalty:
# `HardConsistency()` projects onto $\{x : \mathcal{A}x = y\}$, and the regularizer supplies the
# second proximal map.

# %%
x_dr = reconstruct(
    data,
    IterativeReconstruction(
        L1Wavelet2D(2.0f-3);
        fidelity = HardConsistency(maxit = 20), algorithm = DouglasRachford(), maxit = 40
    );
    verbosity = Silent()
)
println("DR + hard consistency NRMSE ", round(nrmse(x_dr), digits = 4))

# %%
jim(
    jim(x_cgnr; title = "CGNR (L2)"),
    jim(x_fista; title = "FISTA (wavelet)"),
    jim(x_admm; title = "ADMM (wavelet+TV)"),
    jim(x_dr; title = "DR (hard consistency)");
    layout = (2, 2), size = (800, 700)
)

# %% [markdown]
# ### Letting MRT choose
#
# A tuple of candidates is filtered by problem structure; the first applicable one wins.

# %%
x_auto = reconstruct(
    data,
    IterativeReconstruction(L1Wavelet2D(2.0f-3); algorithm = (CG(), FISTA(), ADMM()), maxit = 60);
    verbosity = Silent()
)
println("auto-selected NRMSE ", round(nrmse(x_auto), digits = 4), "  (equals FISTA: ", x_auto ≈ x_fista, ")")

# %% [markdown]
# ## 3. `maxit` and `tol`
#
# Both belong to the *method*. `tol` is relative: the threshold handed to the solver is
# `max(10*eps, tol * maximum(abs, x₀))`. Setting either to `nothing` defers to the algorithm's
# own value — which is how `algorithm = FISTA(maxit = 500)` becomes reachable.

# %%
x_loose = reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 200, tol = 1.0f-3); verbosity = Silent())
x_tight = reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 200, tol = 1.0f-6); verbosity = Silent())
println("tol 1e-3 NRMSE ", round(nrmse(x_loose), digits = 4))
println("tol 1e-6 NRMSE ", round(nrmse(x_tight), digits = 4))

# %%
# Defer to the algorithm's own iteration count.
x_alg = reconstruct(
    data,
    IterativeReconstruction(L1Wavelet2D(2.0f-3); algorithm = FISTA(maxit = 30), maxit = nothing);
    verbosity = Silent()
)
println("algorithm-owned maxit: NRMSE ", round(nrmse(x_alg), digits = 4))

# %%
# Passing an iteration parameter to `reconstruct` is an error rather than being ignored.
try
    reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3)); maxit = 10)
catch e
    println(sprint(showerror, e))
end

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

# %%
reconstruct(data, IterativeReconstruction(L2Image(1.0f-4); maxit = 5); verbosity = Verbose(; freq = 1));

# %%
# The log can be redirected anywhere — here into a vector, e.g. for a dashboard or a test.
messages = String[]
reconstruct(
    data, IterativeReconstruction(L2Image(1.0f-4); maxit = 5);
    verbosity = Verbose(; printfunc = (args...) -> push!(messages, string(args...)))
)
println(length(messages), " messages captured; first: ", first(messages))

# %% [markdown]
# ## 5. `ReconstructionConfig`
#
# Run settings can be bundled into a reusable object, and individual fields overridden per call.

# %%
config = ReconstructionConfig(; verbosity = Silent(), scaling = BartScaling())

x1 = reconstruct(data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 30); config = config)
x2 = reconstruct(data, IterativeReconstruction(TotalVariation2D(1.0f-3); maxit = 30); config = config)
println("reused config for two methods: ", round(nrmse(x1), digits = 4), ", ", round(nrmse(x2), digits = 4))

# %%
# Keywords win over the config object.
x3 = reconstruct(
    data, IterativeReconstruction(L2Image(1.0f-4); maxit = 5);
    config = ReconstructionConfig(; verbosity = Verbose()), verbosity = Silent()
)
println("silenced despite a verbose config")

# %% [markdown]
# ## 6. Data scaling
#
# The solver behaves better when the data is O(1). `BartScaling` divides by a high quantile of
# the direct reconstruction (the convention BART uses), `MeasurementBasedScaling` derives the
# factor from the measurements, `FixedScaling` takes a number you supply, and `NoScaling` leaves
# the data alone. The output is scaled back unless you ask otherwise.

# %%
for scaling in (NoScaling(), BartScaling(), MeasurementBasedScaling())
    x̂ = reconstruct(
        data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40);
        scaling = scaling, verbosity = Silent()
    )
    println(rpad(string(typeof(scaling).name.name), 24), " NRMSE ", round(nrmse(x̂), digits = 4),
        "   max|x| ", round(maximum(abs, x̂), digits = 3))
end

# %%
# Keep the internally scaled units instead of mapping back.
x_unscaled = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40);
    scaling = BartScaling(), disable_inverse_scale_output = true, verbosity = Silent()
)
println("max|x| scaled back:  ", round(maximum(abs, x1), digits = 3))
println("max|x| left scaled:  ", round(maximum(abs, x_unscaled), digits = 3))

# %% [markdown]
# ## 7. Warm starts
#
# `x₀` seeds the solver. Useful for parameter sweeps, staged reconstructions, and the non-convex
# terms whose result depends on where they start.

# %%
x_stage1 = reconstruct(data, IterativeReconstruction(L1Wavelet2D(5.0f-3); maxit = 40); verbosity = Silent())
x_stage2 = reconstruct(
    data, IterativeReconstruction(L1Wavelet2D(1.0f-3); maxit = 40); x₀ = x_stage1, verbosity = Silent()
)
println("stage 1 NRMSE ", round(nrmse(x_stage1), digits = 4))
println("stage 2 NRMSE ", round(nrmse(x_stage2), digits = 4))

# %% [markdown]
# ## 8. Operator-norm and normal-operator options
#
# A proximal method needs the Lipschitz constant of the data term, i.e. $\|\mathcal{A}\|$. MRT
# estimates it with 20 power iterations by default.
#
# - `exact_opnorm = true` — run the power iteration to convergence (the estimate converges from
#   below, so it is a slight under-estimate).
# - `disable_operator_normalization = true` — skip the estimate and let the algorithm find its
#   own step size by backtracking.
# - `disable_normalop_optimization = true` — do not substitute $\mathcal{A}^*\mathcal{A}$ in
#   least-squares models (useful when debugging a custom operator).

# %%
for (label, kwargs) in (
        ("default", (;)),
        ("exact_opnorm", (; exact_opnorm = true)),
        ("no normalization", (; disable_operator_normalization = true)),
        ("no normal-op", (; disable_normalop_optimization = true)),
    )
    t = @elapsed x̂ = reconstruct(
        data, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40, kwargs...); verbosity = Silent()
    )
    println(rpad(label, 20), " NRMSE ", round(nrmse(x̂), digits = 4), "   ", round(t, digits = 2), " s")
end

# %% [markdown]
# ## 9. A convergence comparison
#
# Running the same problem for a growing iteration budget shows the accelerated method pulling
# ahead of the plain one, and ADMM paying more per iteration for its generality.

# %%
budgets = [5, 10, 20, 40, 80]
curves = Dict{String, Vector{Float64}}()

for (label, alg, reg) in (
        ("ISTA", ISTA(), (L1Wavelet2D(2.0f-3),)),
        ("FISTA", FISTA(), (L1Wavelet2D(2.0f-3),)),
        ("ADMM", ADMM(), (L1Wavelet2D(2.0f-3),)),
    )
    curves[label] = [
        nrmse(
            reconstruct(
                data, IterativeReconstruction(reg...; algorithm = alg, maxit = k, tol = 0.0);
                verbosity = Silent()
            )
        ) for k in budgets
    ]
end

plot(
    budgets, [curves["ISTA"] curves["FISTA"] curves["ADMM"]];
    label = ["ISTA" "FISTA" "ADMM"], xlabel = "iterations", ylabel = "NRMSE",
    marker = :circle, lw = 2, xscale = :log10, size = (600, 380), title = "convergence, ℓ₁-wavelet"
)
