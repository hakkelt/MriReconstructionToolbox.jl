# Phase 2.3 — deterministic assumption matching in `parse_problem`.
# Phase 2.4 — rejecting ruleset: a solver whose convexity/smoothness assumptions the
# term structure cannot certify must fail at solve time with a diagnostic naming the
# unsatisfied property, instead of silently running a solver that stalls or returns
# a wrong answer.

using ProximalAlgorithms: PANOCplus, ZeroFPR, FastForwardBackward

const SO_M = StructuredOptimization

# Capture the stdout of a `print_diagnostics` call as a String. `redirect_stdout`
# needs a real file descriptor, so route through a temp file rather than an IOBuffer.
function capture_diagnostics(f)
    return mktemp() do _path, io
        redirect_stdout(io) do
            f()
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
end

@testset "Phase 2.3 deterministic matching" begin
    Random.seed!(230)
    x = Variable(6)
    A = randn(4, 6)
    b = randn(4)
    # IndBallL2 (norm(x,2) <= c) is genuinely proximable, so PANOCplus can parse it.
    p = problem(ls(A * x - b), norm(x, 2) <= 1.0)

    # Parsing is deterministic: repeated calls select the same terms for the same
    # kwargs (the greedy largest-subset-first rule has no external order dependence).
    r1 = SO_M.parse_problem(p, PANOCplus())
    r2 = SO_M.parse_problem(p, PANOCplus())
    @test r1 !== nothing
    @test r2 !== nothing
    @test Set(keys(r1[2])) == Set(keys(r2[2]))

    # A single term matched against an assumption is found via the shared helper.
    vars = SO_M.extract_variables(p)
    smooth_assumption = first(ProximalAlgorithms.get_assumptions(PANOCplus()))
    match = SO_M.match_assumption(smooth_assumption, p, vars)
    @test match !== nothing
    _, matched_terms = match
    @test length(matched_terms) >= 1

    # suggest_algorithm still returns a non-empty list for a standard lasso problem.
    @test !isempty(SO_M.suggest_algorithm(p))
end

@testset "Phase 2.4 rejecting ruleset" begin
    Random.seed!(240)
    x = Variable(5)
    b = randn(5)
    # `sin(x)` is a nonlinear (hence non-convex) composition; the least-squares term
    # is smooth but not convex.
    p = problem(ls(sin(x) - b))

    # FastForwardBackward requires a convex smooth term -> parsing must reject it.
    @test SO_M.parse_problem(p, FastForwardBackward()) === nothing

    # solve surfaces a clear error instead of silently running.
    @test_throws ErrorException solve(p, FastForwardBackward())

    # The diagnostic names the unsatisfied property.
    diag = capture_diagnostics(() -> SO_M.print_diagnostics(p, FastForwardBackward()))
    @test occursin("is_convex", diag)

    # ZeroFPR permits nonconvex smooth f, so it parses the same problem.
    @test SO_M.parse_problem(p, ZeroFPR()) !== nothing
end
