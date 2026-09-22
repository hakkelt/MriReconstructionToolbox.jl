# AirspeedVelocity.jl benchmark suite for StructuredOptimization.jl.
#
# Run locally with
#
#     julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE)'
#
# or, to compare two revisions the way CI does,
#
#     benchpkg StructuredOptimization --rev=master,HEAD
#
# Note the singular directory name: `benchmarks/` (plural) holds the demo scripts that
# reproduce the figures in the documentation and is not a benchmark suite.
#
# What is measured, and why:
#
#   * `formulation/` — the claim that `merge_function_with_operator`'s cost model is right.
#     The gradient of a least-squares term is timed under both formulations it chooses
#     between (the fused normal operator and the generic `Precompose`) across tall, square
#     and wide operators, which is what sets the `normal_op_worthwhile` threshold.
#   * `block_gram/` — the multi-variable case: the assembled block Gram against applying the
#     `HCAT` and its adjoint in turn (N² operator applications against 2N).
#   * `absorption/` — the diagonal and AAᴴ-diagonal "prox trick" absorptions against the
#     naive formulation, guarding the performance claims made in the theory documentation.
#   * `parse/` — the scoring budget: ranking formulations and term subsets has to be
#     negligible next to the optimization pass it selects.

using BenchmarkTools
using StructuredOptimization
using AbstractOperators, DSPOperators, FFTWOperators
using ProximalOperators
using ProximalAlgorithms
using RecursiveArrayTools
using LinearAlgebra, Random, FFTW

const SO = StructuredOptimization

const SUITE = BenchmarkGroup()

# Deterministic inputs: a benchmark that changes its problem between revisions compares
# nothing.
Random.seed!(0)

# ---------------------------------------------------------------------------------------
# formulation/ — normal operator vs Precompose, across aspect ratios
# ---------------------------------------------------------------------------------------

SUITE["formulation"] = BenchmarkGroup()

# `n` is the domain, `m` the codomain. `normal_op_worthwhile` currently accepts `n <= m`;
# the sweep brackets that threshold so the crossover can be read off directly.
const ASPECTS = [
    ("tall", 200, 800),
    ("square", 400, 400),
    ("mildly_wide", 400, 300),
    ("wide", 800, 200),
]

for (name, n, m) in ASPECTS
    A = MatrixOp(randn(m, n))
    b = randn(m)
    x = randn(n)
    grad = similar(x)

    normal = SO.SqrNormL2WithNormalOp(AbstractOperators.AffineAdd(A, -b), 1)
    precomposed = Precompose(SqrNormL2(), A, 1, -b)

    group = BenchmarkGroup()
    group["normal_op"] = @benchmarkable SO.gradient!($grad, $normal, $x)
    group["precompose"] = @benchmarkable ProximalOperators.gradient!($grad, $precomposed, $x)
    # Building the fused normal operator is a one-off cost the formulation has to earn back;
    # it is timed separately so the crossover can account for it.
    group["build_normal_op"] = @benchmarkable SO.fused_normal_op($A)
    SUITE["formulation"][name] = group
end

# A non-fusing operator: `Lᴴ L` stays a `Compose`, so the normal-operator formulation saves
# no pass and must not be selected. Timed to show what selecting it would have cost.
let
    xv = Variable(512)
    nonfusing = SO.operator(fft(MatrixOp(randn(512, 512)) * xv))
    x = randn(512)
    grad = similar(x)
    precomposed = Precompose(SqrNormL2(), nonfusing, 1, 0)
    group = BenchmarkGroup()
    group["precompose"] = @benchmarkable ProximalOperators.gradient!($grad, $precomposed, $x)
    group["fuse_attempt"] = @benchmarkable SO.fused_normal_op($nonfusing)
    SUITE["formulation"]["nonfusing"] = group
end

# ---------------------------------------------------------------------------------------
# block_gram/ — multi-variable terms
# ---------------------------------------------------------------------------------------

SUITE["block_gram"] = BenchmarkGroup()

let
    n1, n2, m = 150, 100, 600
    u, v = Variable(n1), Variable(n2)
    A, B, b = randn(m, n1), randn(m, n2), randn(m)
    t = ls(A * u + B * v - b)
    op = SO.extract_operators((u, v), t)

    x = ArrayPartition(randn(n1), randn(n2))
    grad = similar(x)

    linear_op = AbstractOperators.remove_displacement(op)
    normal = SO.with_normal_op(t.f, linear_op, SO.displacement(t), t.lambda)
    precomposed = Precompose(t.f, linear_op, 1, SO.displacement(t))

    SUITE["block_gram"]["normal_op"] = @benchmarkable SO.gradient!($grad, $normal, $x)
    SUITE["block_gram"]["hcat_two_pass"] = @benchmarkable ProximalOperators.gradient!($grad, $precomposed, $x)
    SUITE["block_gram"]["assemble"] = @benchmarkable SO.fused_normal_op($linear_op)
end

# ---------------------------------------------------------------------------------------
# absorption/ — the prox trick against the naive formulation
# ---------------------------------------------------------------------------------------

SUITE["absorption"] = BenchmarkGroup()

let
    n = 4096
    a = randn(n) .+ 2
    x = randn(n)
    y = similar(x)
    D = DiagOp(a)

    absorbed = SO.merge_function_with_operator(D, NormL1(), 0, 1)
    naive = Precompose(NormL1(), D, a .^ 2, 0)

    SUITE["absorption"]["diagonal_absorbed"] = @benchmarkable prox!($y, $absorbed, $x, 1.0)
    SUITE["absorption"]["diagonal_precompose"] = @benchmarkable prox!($y, $naive, $x, 1.0)
end

let
    n = 4096
    xv = Variable(n)
    dft = SO.operator(fft(xv))
    x = randn(n)
    y = similar(x)

    absorbed = SO.merge_function_with_operator(dft, NormL1(), 0, 1)
    SUITE["absorption"]["aac_diagonal_absorbed"] = @benchmarkable prox!($y, $absorbed, $x, 1.0)
end

# ---------------------------------------------------------------------------------------
# parse/ — the scoring budget
# ---------------------------------------------------------------------------------------

SUITE["parse"] = BenchmarkGroup()

let
    n, m = 200, 300
    A, b = randn(m, n), randn(m)
    x = Variable(n)
    ~x .= 0.0
    p = problem(ls(A * x - b) + 1.0e-2 * norm(x, 1))
    alg = ProximalAlgorithms.PANOCplus(maxit = 5, tol = 0.0)
    assumptions = ProximalAlgorithms.get_assumptions(alg)
    terms = collect(p)

    op = SO.operator(first(p))
    f = first(p).f

    # Ranking one term's formulations: this is what must stay negligible.
    SUITE["parse"]["best_formulation"] = @benchmarkable SO.best_formulation($op, $f, 0, 1)
    SUITE["parse"]["selection_cost"] =
        @benchmarkable sum(SO.selection_cost(a, $terms) for a in $assumptions)
    # The whole parse, which also *builds* the selected formulation.
    SUITE["parse"]["parse_problem"] = @benchmarkable SO.parse_problem($p, $alg)
    # The yardstick: a five-iteration optimization pass on the same problem.
    SUITE["parse"]["solve_5_iterations"] = @benchmarkable solve($p, $alg)
end
