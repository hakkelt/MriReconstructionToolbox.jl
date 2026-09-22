# # A multi-variable problem
#
# Nothing stops a term from mentioning several variables. This is a source-separation shape:
# a measurement explained as the sum of two contributions, one sparse and one bounded.
#
# ```math
# \operatorname*{minimize}_{\mathbf{x},\,\mathbf{y}} \quad
# \tfrac{1}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{y} - \mathbf{b}\|^2
# + \lambda\|\mathbf{x}\|_1
# \quad\text{subject to}\quad \|\mathbf{y}\|_2 \le r
# ```

using StructuredOptimization
using ProximalAlgorithms
using LinearAlgebra, Random

Random.seed!(0)

m, n1, n2 = 120, 60, 20
A, B = randn(m, n1), randn(m, n2)

x_true = zeros(n1); x_true[randperm(n1)[1:5]] .= randn(5)
y_true = randn(n2); y_true .*= 0.8 / norm(y_true)
b = A * x_true + B * y_true + 0.01 * randn(m)
nothing #hide

# Each variable carries its own regularizer or constraint, and the data term couples them.

x, y = Variable(n1), Variable(n2)
~x .= 0.0
~y .= 0.0

λ, r = 0.05, 1.0
@minimize ls(A * x + B * y - b) + λ * norm(x, 1) st norm(y, 2) <= r with ProximalAlgorithms.PANOCplus(tol = 1.0e-8, maxit = 5000)

(support = count(!iszero, ~x), radius = norm(~y))

# ## What the parser had to do
#
# The two variables live in different spaces, so before anything can be stacked each term is
# *expanded* to the joint domain: the ``\ell_1`` term, which mentions only `x`, gets a zero
# block for `y`. The data term's operator then becomes an `HCAT` of `A` and `B` over the
# joint `ArrayPartition` domain.

terms = problem(ls(A * x + B * y - b) + λ * norm(x, 1), norm(y, 2) <= r)
vars = StructuredOptimization.extract_variables(terms)
op = StructuredOptimization.extract_operators(vars, terms[1])
typeof(op).name.name

# That expansion is also why a multi-variable least-squares term is usually *wide*: its
# domain is ``n_1 + n_2`` while its codomain is the shared ``m``. Here ``80 \le 120``, so the
# fused block Gram is still worth assembling —

StructuredOptimization.normal_op_worthwhile(op)

# — but adding a third variable would tip it over, and the parser would fall back to applying
# the `HCAT` and its adjoint in turn. See [Matrix-free operators](@ref).
#
# ## Constraints are terms
#
# `norm(y, 2) <= r` is not special syntax: it builds a term whose function is the indicator
# of the ball, and whose prox is the projection onto it. `st` and a `+` are two spellings of
# the same thing — `problem(...)` flattens both into one `TermSet`.

length(terms)
