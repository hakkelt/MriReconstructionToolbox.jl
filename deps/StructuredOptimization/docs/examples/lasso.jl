# # Lasso, and warm starting
#
# The lasso is the smallest interesting composite problem: a smooth data term plus a
# non-smooth regularizer whose proximal operator is soft thresholding.
#
# ```math
# \operatorname*{minimize}_{\mathbf{x}} \quad
# \tfrac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|_1
# ```

using StructuredOptimization
using ProximalAlgorithms
using LinearAlgebra, Random

Random.seed!(0)

n, m, k = 200, 100, 8                     # unknowns, measurements, true non-zeros
A = randn(m, n)
x_true = zeros(n)
x_true[randperm(n)[1:k]] .= randn(k)
b = A * x_true + 0.01 * randn(m)
nothing #hide

# The model is written the way the mathematics is written. `ls` is
# ``\tfrac{1}{2}\|\cdot\|^2``; `norm(x, 1)` is the regularizer.

x = Variable(n)
λ = 0.1 * norm(A' * b, Inf)               # large enough to give a sparse solution

@minimize ls(A * x - b) + λ * norm(x, 1)

count(!iszero, ~x)

# `~x` dereferences the variable. The support is recovered up to the noise level:

norm(~x - x_true) / norm(x_true)

# ## Warm starting
#
# A `Variable` owns its data, so solving a second problem over the same variable starts from
# wherever the first one finished. That is the whole warm-start mechanism — there is no flag
# to set. Following a *regularization path* down to a smaller `λ` costs a fraction of
# solving at the small `λ` from scratch:

iters_cold = Int[]
iters_warm = Int[]
path = λ .* [1.0, 0.5, 0.25, 0.125]

for λi in path
    y = Variable(n)                       # fresh variable: cold start from zero
    _, it = solve(problem(ls(A * y - b) + λi * norm(y, 1)), ProximalAlgorithms.PANOCplus(tol = 1.0e-8))
    push!(iters_cold, it)

    _, it = solve(problem(ls(A * x - b) + λi * norm(x, 1)), ProximalAlgorithms.PANOCplus(tol = 1.0e-8))
    push!(iters_warm, it)                 # reuses the previous solution in `~x`
end

[iters_cold iters_warm]

# To opt out, reset the data explicitly before solving:

~x .= 0.0
nothing #hide
