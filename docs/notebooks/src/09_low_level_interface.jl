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
# # 9 — The low-level interface
#
# `reconstruct` is a convenience layer over three packages that can be driven directly:
# `AbstractOperators.jl` (linear operators), `ProximalOperators.jl` (proximal maps) and
# `StructuredOptimization.jl` (problem syntax + solvers). This notebook opens the box.
#
# **Contents**
# 1. The encoding operator and its parts
# 2. Adjoint and operator-norm checks
# 3. `build_model` — what `reconstruct` builds
# 4. Writing the optimization problem by hand
# 5. Proximal operators directly
# 6. Adding a regularizer of your own

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator,
    get_sensitivity_map_operator, get_subsampling_operator, build_model, materialize,
    get_operator, get_affected_dims, calculate, Regularization, NamedDimsOp
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
using MIRTjim: jim
using Plots
using AbstractOperators
using StructuredOptimization
using ProximalOperators
using ProximalAlgorithms
using WaveletOperators: WaveletOp, WT, wavelet
using LinearAlgebra
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. The encoding operator and its parts
#
# $\mathcal{A} = \mathcal{P}\,\mathcal{F}\,\mathcal{S}$: sensitivities, Fourier transform,
# sampling. Each factor is available separately, and they compose with `*`.

# %%
nx, ny, nc = 128, 128, 8
x_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps = coil_sensitivities(nx, ny, nc)
pattern = create_sampling_pattern(
    VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), (nx, ny)
)

acq = AcquisitionInfo(;
    is3D = false, image_size = (nx, ny), sensitivity_maps = smaps, subsampling = pattern
)
data = simulate_acquisition(x_true + 0.01f0 * randn(ComplexF32, nx, ny), acq)

𝒮 = get_sensitivity_map_operator(data)
ℱ = get_fourier_operator(data)
𝒫 = get_subsampling_operator(data)
𝒜 = get_encoding_operator(data)

for (name, op) in ("𝒮" => 𝒮, "ℱ" => ℱ, "𝒫" => 𝒫, "𝒜" => 𝒜)
    println(rpad(name, 3), " : ", size(op, 2), " → ", size(op, 1))
end

# %%
# Composed by hand, the product reproduces the encoding operator.
𝒜_manual = 𝒫 * ℱ * 𝒮
y = data.kspace_data
println("‖𝒜ᴴy − (𝒫ℱ𝒮)ᴴy‖ / ‖𝒜ᴴy‖ = ", norm(𝒜' * y - 𝒜_manual' * y) / norm(𝒜' * y))

x_adj = 𝒜' * y
jim(x_adj; title = "𝒜ᴴy — the direct reconstruction, by hand", size = (400, 350))

# %% [markdown]
# ## 2. Adjoint and operator-norm checks
#
# Two things worth knowing when you build or wrap operators yourself.
#
# First, the FFT convention. MRT's Fourier operator is the `fft`/`ifft` pair, not the unitary one:
# the forward transform is unnormalized and `'` carries the `1/(nx·ny)` factor. So `𝒜'` is the
# *inverse-scaled* adjoint, and the dot-product identity
# $\langle \mathcal{A}u, v\rangle = \langle u, \mathcal{A}^H v\rangle$ holds only up to that
# factor — as the ratio below shows. Multiply by `nx*ny` for the true Hermitian adjoint. (This is
# also why `‖𝒜‖ ≈ 1` rather than something of the order of the image size.)

# %%
u = randn(ComplexF32, nx, ny)
v = randn(ComplexF32, size(y)...)

lhs = dot(𝒜 * u, v)
rhs = dot(u, 𝒜' * v)
println("⟨𝒜u, v⟩   = ", lhs)
println("⟨u, 𝒜'v⟩  = ", rhs)
println("ratio      = ", round(real(lhs / rhs), digits = 3), "   (nx·ny = ", nx * ny, ")")

# %% [markdown]
# Second, the operator norm: it sets the step size of every proximal algorithm, and MRT estimates
# it with 20 power iterations before each solve.

# %%
L_est = AbstractOperators.estimate_opnorm(𝒜)
println("‖𝒜‖ (20 power iterations): ", round(L_est, digits = 5))

# The power iteration converges from below, so more iterations only increase the estimate.
L_exact = AbstractOperators.estimate_opnorm(𝒜; maxit = 1000, tol = 1.0e-10)
println("‖𝒜‖ (converged):           ", round(L_exact, digits = 5))

# %% [markdown]
# ## 3. `build_model` — what `reconstruct` builds
#
# `build_model` returns the `StructuredOptimization` problem `reconstruct` would solve, so you can
# inspect it, hand it to a solver yourself, or modify it.

# %%
terms = build_model(𝒜, y, (L1Wavelet2D(2.0f-3),))
for t in terms
    println(t)
end

# %%
# The variant that also returns the variables is what you need when a regularizer introduces
# auxiliary variables of its own (`TotalGeneralizedVariation2D` does): the image is then not at a
# predictable position in the solver's variable tuple.
terms_tgv, x_var, auxiliaries = MriReconstructionToolbox.build_model_with_variables(
    𝒜, y, (TotalGeneralizedVariation2D(1.0f-3),)
)
println("image variable:      ", size(~x_var))
println("auxiliary variables: ", length(auxiliaries), " → ", map(a -> size(~a), auxiliaries))

# %% [markdown]
# ## 4. Writing the optimization problem by hand
#
# `Variable`, `ls`, `norm` and `@minimize` are the whole syntax. Below is the compressed-sensing
# problem of notebook 1, written out.

# %%
𝒲 = WaveletOp(ComplexF32, wavelet(WT.db4), (nx, ny))

v = Variable(copy(x_adj))                      # warm start from the direct reconstruction
λ = 2.0f-3

x̂, iterations = @minimize ls(𝒜 * v - y) + λ * norm(𝒲 * v, 1) with FISTA(maxit = 60, verbose = false)
println("converged in ", iterations, " iterations")

nrmse(x) = norm(abs.(x) - abs.(x_true)) / norm(abs.(x_true))
println("hand-written NRMSE:  ", round(nrmse(~x̂), digits = 4))

x_api = reconstruct(data, IterativeReconstruction(L1Wavelet2D(λ); maxit = 60); verbosity = Silent())
println("`reconstruct` NRMSE: ", round(nrmse(x_api), digits = 4))

# The two are the same problem but not the same run: `reconstruct` also scales the data, hands
# FISTA a Lipschitz-constant hint from ‖𝒜‖ and applies its own relative stopping rule, which is
# worth a visible amount of accuracy at a fixed iteration count.

jim(
    jim(~x̂; title = "hand-written problem"),
    jim(x_api; title = "reconstruct(...)");
    layout = (1, 2), size = (800, 350)
)

# %%
# Two variables, two priors: a sparse part and a part that is low rank after a wavelet transform.
# This is the L+S model of notebook 7, written directly.
a = Variable(zeros(ComplexF32, nx, ny))
b = Variable(copy(x_adj))

(â, b̂), it2 = @minimize ls(𝒜 * (a + b) - y) + 5.0f-3 * norm(a, 1) + 1.0f-3 * norm(𝒲 * b, 1) with FISTA(maxit = 40, verbose = false)
println("iterations: ", it2, ", NRMSE of the sum: ", round(nrmse(~â + ~b̂), digits = 4))

jim(
    jim(~â; title = "sparse part"),
    jim(~b̂; title = "wavelet-sparse part"),
    jim(~â + ~b̂; title = "sum");
    layout = (1, 3), size = (1100, 330)
)

# %%
# `problem` + `solve` is the non-macro form, and lets you inspect what a given solver expects.
p = problem(ls(𝒜 * v - y), λ * norm(𝒲 * v, 1))
alg, kwargs, variables = StructuredOptimization.parse_problem(p, FISTA())
println("keys prepared for FISTA: ", keys(kwargs))

# %% [markdown]
# ## 5. Proximal operators directly
#
# Every regularizer is ultimately a proximal map. They can be evaluated on their own, which is the
# quickest way to understand what a term does — and to test a new one.

# %%
z = randn(ComplexF32, 8, 8)
γ = 0.5

f = NormL1(0.3)
p_l1 = similar(z)
value = prox!(p_l1, f, z, γ)
println("NormL1: value at the prox point = ", round(value, digits = 4))
println("soft thresholding by γλ = ", γ * 0.3, ":")
println("  |z|     ", round.(abs.(z[1:4, 1]), digits = 3))
println("  |prox|  ", round.(abs.(p_l1[1:4, 1]), digits = 3))

# %%
# The nuclear norm shrinks singular values instead of entries.
M = randn(ComplexF32, 16, 6)
p_nuc = similar(M)
prox!(p_nuc, NuclearNorm(0.8), M, 1.0)
println("singular values before: ", round.(svdvals(M)[1:6], digits = 3))
println("singular values after:  ", round.(svdvals(p_nuc)[1:6], digits = 3))

# %%
# MRT's regularizers expose the same thing through `calculate` (the value) and `get_operator`
# (the transform), which is what the extension interface is built on.
reg = L1Wavelet2D(2.0f-3)
println("value of the term at x_true: ", round(calculate(reg, x_true), digits = 4))
println("its operator: ", typeof(get_operator(reg, x_true)).name.name)
println("dimensions it couples: ", get_affected_dims(reg, nothing, (:x, :y, :slice)))

# %% [markdown]
# ## 6. Adding a regularizer of your own
#
# A regularizer is a `struct <: Regularization` plus three methods:
#
# - `get_operator(reg, x; threaded)` — the linear transform it penalizes.
# - `materialize(reg, x::Variable; threaded)` — the `StructuredOptimization.Term`.
# - `get_affected_dims(reg, dimspec, image_dims)` — which image dimensions it couples (this is
#   what decides whether the problem still decomposes over slices).
#
# Optionally `scale_regularization` (if the term is homogeneous, so that data scaling can adjust
# λ), `bind_dimensions` (if it is parameterized by a dimension name) and
# `materialize_with_auxiliaries` (if it introduces extra optimization variables).
#
# Here is a spatially weighted ℓ₁ penalty: sparsity enforced only outside a region of interest,
# which is a crude way of saying "I know where the object is".

# %%
struct MaskedL1{T, W} <: Regularization
    λ::T
    weights::W
end

# The penalty is element-wise, so the operator is the identity and the spatial weights ride along
# in the proximal function (`NormL1` accepts an array of weights).
MriReconstructionToolbox.get_operator(reg::MaskedL1, x::AbstractArray; threaded::Bool = true) =
    Eye(eltype(x), size(x))

# The `::Nothing` slot is the dimension specification a dimension-parameterized regularizer would
# use; an element-wise penalty couples nothing, so it returns an empty tuple.
MriReconstructionToolbox.get_affected_dims(::MaskedL1, ::Nothing, image_dims) = ()

# ℓ₁ is homogeneous of degree one, so λ scales linearly with the data scaling.
MriReconstructionToolbox.scale_regularization(reg::MaskedL1, factor::Real) =
    MaskedL1(reg.λ * factor, reg.weights)

function MriReconstructionToolbox.materialize(reg::MaskedL1, x::Variable{T}; threaded::Bool) where {T}
    op = get_operator(reg, ~x; threaded)
    Γ = real(T).(reg.λ .* reg.weights)
    return StructuredOptimization.Term(1, NormL1(Γ), op * x, "‖Γ .* x‖₁")
end

# %%
# Weight the background 10× more heavily than the object.
radius = [sqrt((i - nx / 2)^2 + (j - ny / 2)^2) for i in 1:nx, j in 1:ny]
weights = Float32.(ifelse.(radius .< 0.42nx, 0.1, 1.0))

x_masked = reconstruct(data, IterativeReconstruction(MaskedL1(5.0f-3, weights); maxit = 60); verbosity = Silent())
println("MaskedL1 NRMSE: ", round(nrmse(x_masked), digits = 4))
println("plain L1Image:  ", round(nrmse(reconstruct(data, IterativeReconstruction(L1Image(5.0f-3); maxit = 60); verbosity = Silent())), digits = 4))

jim(
    jim(weights; title = "penalty weights"),
    jim(x_masked; title = "MaskedL1 reconstruction");
    layout = (1, 2), size = (800, 350)
)

# %% [markdown]
# (Down-weighting the object means *less* regularization where the signal is, so this particular
# prior is worse than plain `L1Image` on this phantom. The point is the interface: a fifteen-line
# regularizer drops straight into `reconstruct`, data scaling, algorithm selection and problem
# decomposition.)

# %%
# It works through the whole stack: `calculate` evaluates it, and the problem still decomposes
# over batch dimensions because `get_affected_dims` says it couples nothing.
println("value at x_true: ", round(calculate(MaskedL1(5.0f-3, weights), x_true), digits = 4))
println("affected dims:   ", get_affected_dims(MaskedL1(5.0f-3, weights), nothing, (:x, :y, :slice)))

# %% [markdown]
# ### Where a new proximal function belongs
#
# If the new term is a *proximal function* with no MRI-specific content — a norm, an indicator,
# a projection — it belongs in the `ProximalOperators` fork under `deps/`, not in MRT. Only the
# MRI-facing wrapper (which operator it applies, which dimensions it touches) lives here.
