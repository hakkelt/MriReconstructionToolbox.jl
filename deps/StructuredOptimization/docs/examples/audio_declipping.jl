# # Audio declipping
#
# A clipped recording has lost every sample that exceeded the converter's range. What is
# left is a *constraint*, not data: each surviving sample is known exactly, and each clipped
# one is known only to lie beyond the threshold, with the right sign.
#
# The prior that makes the problem solvable is sparsity in a frequency dictionary — a short
# musical signal is a handful of tones, so its DCT is sparse. Writing
# ``\mathbf{x} = \mathrm{idct}(\mathbf{c})`` and solving for the coefficients ``\mathbf{c}``:
#
# ```math
# \operatorname*{minimize}_{\mathbf{c}} \quad \|\mathbf{c}\|_1
# \quad\text{subject to}\quad
# [\mathrm{idct}(\mathbf{c})]_{\mathcal{R}} = \mathbf{y}_{\mathcal{R}}
# ```
#
# where ``\mathcal{R}`` is the set of unclipped samples. The listening samples on the
# [Demos](@ref) page come from exactly this model, run on a real recording; this page uses a
# synthetic signal so the documentation build stays self-contained.

using StructuredOptimization
using ProximalAlgorithms
using LinearAlgebra, Random, FFTW

Random.seed!(0)

N = 1024
t = range(0, 1; length = N)
clean = sin.(2π * 55 * t) + 0.6 * sin.(2π * 110 * t) + 0.3 * sin.(2π * 165 * t)
clean ./= maximum(abs, clean)

threshold = 0.6
clipped = clamp.(clean, -threshold, threshold)
reliable = findall(abs.(clipped) .< threshold - 1.0e-9)      # samples that survived
length(reliable) / N                                          # fraction kept

# The model. `idct(c)[reliable]` composes an inverse DCT with a `GetIndex`; both are
# operators, so nothing is materialised. The data-fidelity term is a least-squares penalty on
# the reliable samples rather than a hard constraint, which keeps the problem in the
# composite form a proximal-gradient method wants.

c = Variable(N)
~c .= 0.0

@minimize ls(idct(c)[reliable] - clipped[reliable]) + 1.0e-3 * norm(c, 1) with ProximalAlgorithms.PANOCplus(tol = 1.0e-8, maxit = 2000)

restored = idct(~c)
nothing #hide

# The restored signal should exceed the clipping threshold where the original did — that is
# the whole point, and it is what a plain interpolation cannot do:

(peak_clipped = maximum(abs, clipped), peak_restored = maximum(abs, restored), peak_clean = maximum(abs, clean))

# Error on the clipped samples only, which is where the reconstruction is doing work:

clipped_idx = setdiff(1:N, reliable)
norm(restored[clipped_idx] - clean[clipped_idx]) / norm(clean[clipped_idx])

# ## The declipping constraint proper
#
# The formulation above lets the restored signal fall back below the threshold on a clipped
# sample, which the physics forbids. Adding that knowledge as an inequality constraint gives
# the model in the demo notebook; it needs a solver with a second proximable slot, and
# [`suggest_algorithm`](@ref) will tell you which ones qualify.
