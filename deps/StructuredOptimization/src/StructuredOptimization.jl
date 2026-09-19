module StructuredOptimization

using LinearAlgebra
using RecursiveArrayTools
using ProximalCore
using AbstractOperators, DSPOperators, FFTWOperators
using ProximalOperators
using ProximalAlgorithms
using Combinatorics: permutations, powerset
using ProximalAlgorithms: IterativeAlgorithm, override_parameters

abstract type AbstractExpression end

include("syntax/variable.jl")
include("syntax/expressions/expression.jl")
include("syntax/terms/term.jl")

const TermOrExpr = Union{Term, AbstractExpression}

include("calculus/precomposeNonlinear.jl") # TODO move to ProximalOperators?
include("calculus/sqrNormL2WithNormalOp.jl")

# problem parsing
include("solvers/terms_extract.jl")
include("solvers/terms_properties.jl")
include("solvers/parse.jl")

# solver calls
include("solvers/build_solve.jl")
include("solvers/minimize.jl")

# Bridge ProximalOperators-style `gradient`/`gradient!` to the `value_and_gradient`
# interface ProximalAlgorithms expects. This must accept *any* smooth function this
# package composes and hands to a solver — including arbitrary ProximalOperators
# building blocks (SqrNormL2, Precompose, Postcompose, MoreauEnvelope, …) and this
# package's own wrappers — so it cannot be narrowed to owned types without dropping
# support for problems built from those. It is therefore a deliberate cross-interface
# adaptation (both functions are dependencies of this package); it is listed in the
# Aqua `treat_as_own` allowlist to mark it as intentional rather than accidental
# piracy. ProximalAlgorithms' own `value_and_gradient(::AutoDifferentiable/::Zero, x)`
# methods are more specific, so they still take precedence for those types.
ProximalAlgorithms.value_and_gradient(f, x) = begin
    y, fy = gradient(f, x)
    return fy, y
end
ProximalAlgorithms.value_and_gradient!(grad_f_x, f, x) = begin
    fy = gradient!(grad_f_x, f, x)
    return fy
end

end
