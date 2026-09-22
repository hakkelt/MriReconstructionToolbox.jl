# # FFT deconvolution, and the normal-operator speed-up
#
# Deconvolution is the archetypal matrix-free problem: the forward operator is a
# convolution, which as a matrix would be dense and enormous, and as an operator is two FFTs.
#
# ```math
# \operatorname*{minimize}_{\mathbf{x}} \quad
# \tfrac{1}{2}\|\mathbf{h} \ast \mathbf{x} - \mathbf{y}\|^2 + \lambda\|\mathbf{x}\|_1
# ```
#
# This page also measures the formulation choice the parser makes for the data term.

using StructuredOptimization
using AbstractOperators, ProximalOperators, ProximalAlgorithms
using LinearAlgebra, Random, FFTW, DSP

Random.seed!(0)

N = 2048
h = exp.(-(0:63) ./ 12) .* cos.(2π * (0:63) ./ 9)      # a decaying oscillatory kernel

x_true = zeros(N)
x_true[randperm(N)[1:20]] .= randn(20)                  # a sparse spike train
y = DSP.conv(x_true, h) + 1.0e-3 * randn(N + length(h) - 1)
nothing #hide

# `conv(x, h)` builds a convolution operator. Its normal operator is a single multiplication
# in the frequency domain — the product *fuses* — which is what makes the formulation below
# worth choosing.

x = Variable(N)
~x .= 0.0

@minimize ls(conv(x, h) - y) + 0.02 * norm(x, 1) with ProximalAlgorithms.PANOCplus(tol = 1.0e-8, maxit = 2000)

(recovered = count(!iszero, ~x), planted = 20, relative_error = norm(~x - x_true) / norm(x_true))

# ## Which formulation was selected
#
# The data term's operator is tall (`N` in, `N + length(h) - 1` out), linear, and its normal
# operator fuses — a convolution composed with its adjoint is one multiplication in the
# frequency domain. That makes `SqrNormL2WithNormalOp` the cheapest candidate once the
# algorithm asks only for a gradient. On a dense operator of the same aspect ratio:

A = MatrixOp(randn(800, 200))                           # tall, fusing
StructuredOptimization.best_formulation(A, SqrNormL2(), 0, 1)

# A wide operator loses: `LᴴL` would act on the larger space.

StructuredOptimization.best_formulation(MatrixOp(randn(200, 800)), SqrNormL2(), 0, 1)

# Ranking a formulation reads only operator metadata — sizes and trait predicates — so it
# costs the same whatever the operator's size, and allocates nothing:

small, big = MatrixOp(randn(10, 8)), MatrixOp(randn(800, 600))
StructuredOptimization.best_formulation(small, SqrNormL2(), 0, 1)                     # warm up
StructuredOptimization.best_formulation(big, SqrNormL2(), 0, 1)
(
    small = @allocated(StructuredOptimization.best_formulation(small, SqrNormL2(), 0, 1)),
    big = @allocated(StructuredOptimization.best_formulation(big, SqrNormL2(), 0, 1)),
)

# ## The speed-up, measured
#
# Per gradient, the fused normal operator against the generic two-pass formulation on a tall
# operator:

Ad = MatrixOp(randn(2000, 500))
xd, gd = randn(500), zeros(500)

normal = StructuredOptimization.SqrNormL2WithNormalOp(Ad, 1)
generic = Precompose(SqrNormL2(), Ad, 1, 0)

StructuredOptimization.gradient!(gd, normal, xd)          # warm up
ProximalOperators.gradient!(gd, generic, xd)

t_normal = minimum(@elapsed(StructuredOptimization.gradient!(gd, normal, xd)) for _ in 1:50)
t_generic = minimum(@elapsed(ProximalOperators.gradient!(gd, generic, xd)) for _ in 1:50)
(t_normal, t_generic, speedup = t_generic / t_normal)

# The catch is that forming ``\mathbf{A}^{\mathsf{H}}\mathbf{A}`` is a one-off cost, and that
# it squares the condition number. `benchmark/benchmarks.jl` measures where the trade turns;
# [Matrix-free operators](@ref) summarises the answer.
