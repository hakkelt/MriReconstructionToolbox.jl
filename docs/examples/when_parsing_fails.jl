# # When parsing fails
#
# `solve` refuses problems it cannot certify rather than running a solver whose assumptions
# are violated — which would stall, or converge to something that is not a solution. This
# page walks through reading that refusal.

using StructuredOptimization
using ProximalAlgorithms
using LinearAlgebra, Random

Random.seed!(0)

# ## A term that is not proximable
#
# `norm(A*x, 1)` looks like `norm(x, 1)` with an operator attached, but proximability does not
# survive composition: knowing the soft-threshold formula tells you nothing about the prox of
# ``\|\mathbf{A}\cdot\|_1``.

n = 40
A = randn(30, n)
x = Variable(n)
~x .= 0.0

p_bad = problem(norm(A * x, 1))
StructuredOptimization.is_proximable(first(p_bad))

# That does not mean nothing can solve it. Algorithms with a *separate operator slot* take
# the composition apart and never ask for the composed prox, so they accept the term:

length(suggest_algorithm(p_bad))

# A plain proximal-gradient method has no such slot, and refuses it:

ProximalAlgorithms.FastForwardBackward() in suggest_algorithm(p_bad)

# [`print_diagnostics`](@ref StructuredOptimization.print_diagnostics) names what blocked it,
# term by term:

StructuredOptimization.print_diagnostics(p_bad, ProximalAlgorithms.FastForwardBackward())

# ### Three ways to make it proximal-gradient friendly
#
# **Give the operator a structure that absorbs.** `fft` is AAᴴ-diagonal, so
# ``\|\mathrm{fft}(\mathbf{x})\|_1`` *is* proximable — the "prox trick" of
# [How problems are parsed](@ref). This is why sparsity in a transform domain is cheap while
# sparsity under a general dictionary is not:

z = Variable(n)
~z .= 0.0
ProximalAlgorithms.FastForwardBackward() in suggest_algorithm(problem(norm(fft(z), 1) + ls(z)))

# **Smooth it.** [`smooth`](@ref) replaces a term by its Moreau envelope, which is
# differentiable and has the same minimizers in the limit of a small parameter — at the cost
# of solving a slightly different problem:

b = randn(30)
length(suggest_algorithm(problem(ls(A * x - b) + smooth(norm(A * x, 1), 0.1))))

# **Introduce the composition as a variable of its own,** so the ``\ell_1`` penalty applies
# to something the algorithm can prox directly and the operator moves into a data term. That
# is a modelling change rather than a syntax trick, and it is what a splitting method does
# internally anyway.

# ## A term that is not convex
#
# A non-linear composition is smooth but not convex, so an algorithm that assumes convexity
# must refuse it. Here the refusal is about a *different* property, and the diagnostic says
# so:

y = Variable(5)
~y .= 0.1
p_nonconvex = problem(ls(sin(y) - randn(5)))

StructuredOptimization.print_diagnostics(p_nonconvex, ProximalAlgorithms.FastForwardBackward())

# The error thrown by `solve` carries the same information, so a caught exception is as
# useful as the printed report:

try
    solve(p_nonconvex, ProximalAlgorithms.FastForwardBackward())
catch err
    println(err.msg)
end

# Choosing an algorithm that permits a non-convex smooth term parses it immediately:

[typeof(a).name.name for a in suggest_algorithm(p_nonconvex)]

# ## Naming terms for readable diagnostics
#
# By default a term prints as its desugared operator graph, which is accurate but hard to
# read. [`@term`](@ref) records the source text instead:

t = @term norm(A * x, 1)
t.repr

# and that is what the diagnostics and the error message will show.
