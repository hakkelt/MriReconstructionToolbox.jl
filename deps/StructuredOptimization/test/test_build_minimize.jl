using ProximalAlgorithms: ZeroFPR, PANOC, PANOCplus

x = Variable(10)
A = randn(5, 10)
y = Variable(7)
B = randn(5, 7)
b = randn(5)

println("\nTesting @minimize \n")
~x .= 0.
~y .= 0.
slv, = @minimize ls(A*x - B*y + b) st norm(x, 2) <= 1e4, norm(y, 2) <= 1.0 with PANOCplus()
~x .= 0.
slv, = @minimize ls(A*x - b) st norm(x, 2) <= 1.0 with PANOCplus()
~x .= 0.
slv, = @minimize ls(A*x - b) st norm(x, 2) <= 1.0
~x .= 0.
slv, = @minimize ls(A*x - b) + norm(x, 1) with PANOCplus()
~x .= 0.
slv, = @minimize ls(A*x - b) + norm(x, 1)
~x .= 0.
slv, = @minimize ls(A*x - b)

# suggest_algorithm and print_diagnostics
prob_lasso = problem(ls(A*x - b) + 1e-3*norm(x, 1))
algs = StructuredOptimization.suggest_algorithm(prob_lasso)
@test !isempty(algs)
@test_nowarn StructuredOptimization.print_diagnostics(prob_lasso, PANOCplus())

# multi-solver solve (first solver in list is tried)
let A_ms = randn(5, 10), b_ms = randn(5)
    x_ms = Variable(10)
    sol_ms = solve(problem(ls(A_ms*x_ms - b_ms) + 1e-3*norm(x_ms, 1)), (PANOCplus(maxit=20), ZeroFPR(maxit=20)))
    @test !isnothing(sol_ms)
end

#TODO many many more tests
Random.seed!(12345)
x = Variable(5)
A = randn(10, 5)
b = randn(10)

println("\nTesting @minimize nonlinear \n")
slv, = @minimize ls(sigmoid(A*x,10) - b)+norm(x,1) with PANOCplus(tol = 1e-6)
xpg = copy(~x)
~x .= 0.
slv, = @minimize ls(sigmoid(A*x,10) - b)+norm(x,1) with ZeroFPR(tol = 1e-6)
xzfpr = copy(~x)
~x .= 0.

@test norm(xzfpr-xpg) <= 1e-4

# test nonconvex Rosenbrock function with known minimum
function test_solver(solver)
	x = Variable(1)
	y = Variable(1)
	a, b = 2.0, 100.0

	cf = norm(x - a)^2 + b * norm(pow(x, 2) - y)^2
	@minimize cf + 1e-10 * norm(x, 1) + 1e-10 * norm(y, 1) with solver

	@test norm(~x - [a]) < 1e-4
	@test norm(~y - [a^2]) < 1e-4
end
solvers = [ZeroFPR(; tol=1e-6), PANOCplus(; tol=1e-6)]
for solver in solvers
	test_solver(solver)
end

# build_solve.jl — print_diagnostics(terms), error paths
let A = randn(5, 4), b = randn(5)
    x = Variable(4)
    prob = problem(ls(A*x - b) + norm(x, 1))

    # print_diagnostics with no algorithm argument (auto-finds best)
    @test_nowarn StructuredOptimization.print_diagnostics(prob)

    # solve with a tuple of solvers
    ~x .= 0.0
    sol = solve(prob, (PANOCplus(tol=1e-6),))
    @test !isnothing(sol)

    # solve with no solver (auto-select)
    x2 = Variable(4)
    ~x2 .= 0.0
    prob2 = problem(ls(A*x2 - b) + norm(x2, 1))
    sol2 = solve(prob2)
    @test !isnothing(sol2)
    @test norm(~x2, Inf) <= norm(b) + 1
end

# build_solve.jl — error for unparseable problem (with single solver)
let
    # CGNR only handles purely quadratic+linear problems.
    x_err = Variable(4)
    prob_bad = problem(norm(x_err, 1))
    @test_throws ErrorException solve(prob_bad, ProximalAlgorithms.CGNR())
end

# minimize.jl — @minimize st ... with solver
let A = randn(5, 4), b = randn(5)
    x = Variable(4)
    ~x .= 0.0
    @minimize ls(A*x - b) st norm(x, 2) <= 1.0 with PANOCplus(tol=1e-6)
    @test norm(~x, Inf) <= norm(b) + 1
end

# minimize.jl — @minimize with a Symbol
let A = randn(5, 4), b = randn(5)
    x = Variable(4)
    ~x .= 0.0
    my_term = ls(A*x - b) + norm(x, 1)
    sol = solve(my_term)
    @test !isnothing(sol)
end

# build_solve.jl — multi-solver tuple where all solvers fail
let
    x = Variable(4)
    prob_bad = problem(norm(x, 1))
    @test_throws ErrorException solve(prob_bad, (ProximalAlgorithms.CGNR(), ProximalAlgorithms.CGNR(maxit=5)))
end
