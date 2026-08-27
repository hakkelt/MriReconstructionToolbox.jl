module StructuredOptimization

using LinearAlgebra
using RecursiveArrayTools
using ProximalCore
using AbstractOperators, DSPOperators, FFTWOperators
using ProximalOperators
using ProximalAlgorithms
using Combinatorics: permutations, powerset
using ProximalAlgorithms: IterativeAlgorithm, override_parameters

ProximalAlgorithms.value_and_gradient(f, x) = begin
  y, fy = gradient(f, x)
  return fy, y
end
ProximalAlgorithms.value_and_gradient!(grad_f_x, f, x) = begin
  fy = gradient!(grad_f_x, f, x)
  return fy
end

abstract type AbstractExpression end

include("syntax/variable.jl")
include("syntax/expressions/expression.jl")
include("syntax/terms/term.jl")

const TermOrExpr =  Union{Term,AbstractExpression}

include("calculus/precomposeNonlinear.jl") # TODO move to ProximalOperators?
include("calculus/sqrNormL2WithNormalOp.jl")

# problem parsing
include("solvers/terms_extract.jl")
include("solvers/terms_properties.jl")
include("solvers/parse.jl")

# solver calls
include("solvers/build_solve.jl")
include("solvers/minimize.jl")

end
