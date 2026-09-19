# # A non-convex problem: Rosenbrock
#
# Rosenbrock's function is the standard test of a method's ability to follow a curved,
# ill-conditioned valley:
#
# ```math
# f(\mathbf{x}) = (1 - x_1)^2 + 100\,(x_2 - x_1^2)^2 .
# ```
#
# Written in this package's vocabulary it is a least-squares term over a *non-linear*
# expression — and that is what makes it non-convex, because composing a convex function with
# a non-linear map does not preserve convexity.

using StructuredOptimization
using ProximalAlgorithms
using LinearAlgebra, Random

x = Variable(2)
~x .= [-1.2, 1.0]                          # the classical starting point

# `pow(x[1:1], 2)` is a non-linear operator that knows its own Jacobian, so no automatic
# differentiation is involved. The two residuals become two terms.
#
# The slices are `1:1` rather than `1`: a scalar index would give an expression whose
# codomain is a scalar rather than a one-element vector, which the gradient machinery does
# not currently handle.

residual(x) = ls(sqrt(2) * (x[1:1] - [1.0])) + ls(sqrt(2) * 10 * (x[2:2] - pow(x[1:1], 2)))
nothing #hide

# The factor ``\sqrt{2}`` cancels the ``\tfrac{1}{2}`` in `ls`, so the objective is exactly
# Rosenbrock's.
#
# ## Which algorithms apply
#
# The term is smooth but not convex, so an algorithm that assumes convexity must refuse it.
# That refusal is structural — the parser never evaluates the function to find out:

p = problem(residual(x))
algs = suggest_algorithm(p)
[typeof(a).name.name for a in algs]

# `FastForwardBackward` is absent, and asking for it by name reports why:

StructuredOptimization.print_diagnostics(p, ProximalAlgorithms.FastForwardBackward())

# `ZeroFPR` and `PANOCplus` both permit a non-convex smooth term:

solve(p, ProximalAlgorithms.PANOCplus(tol = 1.0e-10, maxit = 20000))
~x

# The minimum is at ``(1, 1)``:

norm(~x - [1.0, 1.0])

# ## The formulation behind it
#
# Because the operator is non-linear, the parser reaches the last formulation in its table:
# [`PrecomposeNonlinear`](@ref), which applies the operator and then its Jacobian adjoint. It
# has a gradient and no proximal operator, which is exactly why a purely proximal algorithm
# cannot take this term either.

t = p[2]                                   # the term containing `pow`
g = StructuredOptimization.merge_function_with_operator(
    StructuredOptimization.operator(t), t.f, StructuredOptimization.displacement(t), t.lambda
)
typeof(g).name.name

# The other term is linear, so it takes an ordinary linear formulation — the two terms of one
# problem need not share one:

t1 = p[1]
StructuredOptimization.best_formulation(
    StructuredOptimization.operator(t1), t1.f, StructuredOptimization.displacement(t1), t1.lambda
)
