# # Total variation denoising
#
# Total variation asks for an image that is close to the data and *piecewise* smooth: the
# penalty is on the magnitude of the gradient, summed over pixels, so a few large jumps are
# cheaper than many small ones. Edges survive; noise does not.
#
# ```math
# \operatorname*{minimize}_{\mathbf{x}} \quad
# \tfrac{1}{2}\|\mathbf{x} - \mathbf{y}\|^2 + \lambda \, \mathrm{TV}(\mathbf{x})
# ```
#
# The point of interest here is that the gradient operator is never a matrix. `variation`
# builds a finite-difference operator that applies in ``O(N)`` and knows it is linear.

using StructuredOptimization
using ProximalAlgorithms
using LinearAlgebra, Random

Random.seed!(0)

N = 64
truth = zeros(N, N)                       # a few piecewise-constant blocks
truth[10:30, 10:30] .= 1.0
truth[35:55, 20:50] .= 0.6

y = truth + 0.15 * randn(N, N)
nothing #hide

# `variation(x)` stacks the horizontal and vertical differences, so `norm(variation(x), 1)`
# is the anisotropic total variation.
#
# The term is *not* proximable: the operator is neither diagonal nor AAᴴ-diagonal, and the
# prox of ``\|\nabla \cdot\|_1`` has no closed form. A proximal-gradient method therefore
# cannot take it, and asking for one is refused rather than silently mis-solved. What can
# take it is an algorithm with a separate *operator slot*, which keeps the operator outside
# the function and never needs the composed prox:

x = Variable(N, N)
~x .= y                                   # a sensible starting point

λ = 0.12
p = problem(ls(x - y) + λ * norm(variation(x), 1))
[typeof(a).parameters[1] for a in suggest_algorithm(p)]

# Solving with ADMM:

solve(p, ProximalAlgorithms.ADMM(maxit = 400))
nothing #hide

# Denoising quality, as the relative error against the truth. The noisy input is the
# baseline to beat:

(noisy = norm(y - truth) / norm(truth), denoised = norm(~x - truth) / norm(truth))

# The total variation of the result is a fraction of the data's — which is what the
# regularizer was asked for:

tv(z) = sum(abs, StructuredOptimization.operator(variation(Variable(size(z)...))) * z)
(tv_data = tv(y), tv_denoised = tv(~x), tv_truth = tv(truth))

# ## Why the operator matters
#
# `variation` on a 512×512 image is a ``524288 \times 262144`` linear map. As a dense matrix
# that is a terabyte; as a sparse one it is still a million stored entries to build and
# index. As an operator it stores nothing at all and applies by subtracting shifted views.
# See [Matrix-free operators](@ref).
