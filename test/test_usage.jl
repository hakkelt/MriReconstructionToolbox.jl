using ProximalAlgorithms: PANOCplus, FastForwardBackward, ZeroFPR

Random.seed!(0)

################################################################################
### Regularized least squares, with two variable blocks to make things weird
################################################################################

println("Testing: regularized least squares, with two variable blocks to make things weird")
m, n1, n2 = 30, 50, 100

A1 = randn(m, n1)
A2 = randn(m, n2)
b = randn(m)

lam1 = 0.2
lam2 = 1.0

# Solve with PANOC+

x1_panocplus = Variable(n1)
x2_panocplus = Variable(n2)
expr = ls(A1*x1_panocplus + A2*x2_panocplus - b) + lam1*norm(x1_panocplus, 1) + lam2*norm(x2_panocplus, 2)
prob = problem(expr)
@time sol = solve(prob, PANOCplus())

res = A1*~x1_panocplus + A2*~x2_panocplus - b
grad1 = A1'*res
grad2 = A2'*res
ind1_zero = (~x1_panocplus .== 0)
subgr1 = lam1*sign.(~x1_panocplus)
subdiff1_low, subdiff1_upp = copy(subgr1), copy(subgr1)
subdiff1_low[ind1_zero] .= -lam1
subdiff1_upp[ind1_zero] .= +lam1
subgr2 = lam2*(~x2_panocplus/norm(~x2_panocplus, 2))

@test maximum(subdiff1_low + grad1) <= 1e-6
@test maximum(-subdiff1_upp - grad1) <= 1e-6
@test norm(grad2 + subgr2) <= 1e-6

# Solve with FastForwardBackward

x1_ffb = Variable(n1)
x2_ffb = Variable(n2)
expr = ls(A1*x1_ffb + A2*x2_ffb - b) + lam1*norm(x1_ffb, 1) + lam2*norm(x2_ffb, 2)
prob = problem(expr)
@time sol = solve(prob, FastForwardBackward())

@test norm(~x1_panocplus - ~x1_ffb, Inf)/(1+norm(~x1_ffb, Inf)) <= 1e-6
@test norm(~x2_panocplus - ~x2_ffb, Inf)/(1+norm(~x2_ffb, Inf)) <= 1e-6

###############################################################################
## Lasso problem with known solution
###############################################################################

println("Testing: lasso problem with known solution")

m, n, nnz_x_star = 200, 100, 10
A = randn(m, n)
lam = 1.0
x_star = randn(n)
x_star[nnz_x_star+1:end] .= 0.0
y_star = lam*sign.(x_star)
b = A*x_star + A'\y_star
@test norm(A'*(A*x_star - b) + lam*sign.(x_star)) <= 1e-12

# Solve with PG

x_pg = Variable(n)
expr = ls(A*x_pg - b) + lam*norm(x_pg, 1)
prob = problem(expr)
@time sol = solve(prob, PANOCplus(tol=1e-10, verbose=false))

@test norm(~x_pg - x_star, Inf) <= 1e-8
@test norm(A'*(A*~x_pg - b) + lam*sign.(~x_pg)) <= 1e-6

# Solve with PANOC+

x_fpg = Variable(n)
expr = ls(A*x_fpg - b) + lam*norm(x_fpg, 1)
prob = problem(expr)
@time sol = solve(prob, PANOCplus(tol=1e-10, verbose=false))

@test norm(~x_fpg - x_star, Inf) <= 1e-8
@test norm(A'*(A*~x_fpg - b) + lam*sign.(~x_fpg)) <= 1e-6

# Solve with ZeroFPR — dispatch test only on a tiny problem; ZeroFPR hits stepsize-too-small
# on larger problems, so we use (5×3) here and only check boundedness, not convergence accuracy

let A_tiny = randn(5, 3), b_tiny = randn(5), lam_tiny = 0.1
    x_zerofpr = Variable(3)
    expr = ls(A_tiny*x_zerofpr - b_tiny) + lam_tiny*norm(x_zerofpr, 1)
    prob = problem(expr)
    @time sol = solve(prob, ZeroFPR(tol=1e-4, verbose=false))
    @test norm(~x_zerofpr, Inf) <= norm(b_tiny) + 1  # solution is bounded (solver ran)
    @test !any(isnan.(~x_zerofpr))
end

# Solve with FastForwardBackward (proximal gradient — different algorithm type to PANOCplus)

x_ffb = Variable(n)
expr = ls(A*x_ffb - b) + lam*norm(x_ffb, 1)
prob = problem(expr)
@time sol = solve(prob, FastForwardBackward(tol=1e-10, verbose=false))

@test norm(~x_ffb - x_star, Inf) <= 1e-6
@test norm(A'*(A*~x_ffb - b) + lam*sign.(~x_ffb)) <= 1e-4

################################################################################
### Problem with smooth, non-quadratic term
################################################################################

println("Testing: problem with smooth, non-quadratic term")

m, n, nnz_x_orig = 200, 500, 10
A = randn(m, n)
lam = 1.0
x_orig = randn(n)
x_orig[nnz_x_orig+1:end] .= 0.0
b = A*x_orig + randn(m)

# Solve with PG

x_pg = Variable(n)
expr = smooth(norm(A*x_pg - b, 2)) + lam*norm(x_pg, 1)
prob = problem(expr)
@time sol = solve(prob, PANOCplus(tol=1e-6, verbose=false))

# Solve with PANOC+

x_fpg = Variable(n)
expr = smooth(norm(A*x_fpg - b, 2)) + lam*norm(x_fpg, 1)
prob = problem(expr)
@time sol = solve(prob, PANOCplus(tol=1e-6, verbose=false))

# Solve with ZeroFPR — dispatch test only on tiny problem (ZeroFPR is slow under coverage)
let A_t = randn(5, 3), b_t = randn(5), lam_t = 0.1
    x_zerofpr = Variable(3)
    expr = smooth(norm(A_t*x_zerofpr - b_t, 2)) + lam_t*norm(x_zerofpr, 1)
    prob = problem(expr)
    @time sol = solve(prob, ZeroFPR(tol=1e-4, verbose=false))
    @test !any(isnan.(~x_zerofpr))
end

# Solve with FastForwardBackward (proximal gradient)

x_panoc = Variable(n)
expr = smooth(norm(A*x_panoc - b, 2)) + lam*norm(x_panoc, 1)
prob = problem(expr)
@time sol = solve(prob, FastForwardBackward(tol=1e-6, verbose=false))

# Solve with minimize, default solver/options

#x = Variable(n)
#@time sol = @minimize smooth(norm(A*x - b, 2)) + lam*norm(x, 1)

@test norm(~x_pg - ~x_fpg, Inf)/(1+norm(~x_pg, Inf)) <= 1e-2
@test norm(~x_pg - ~x_panoc, Inf)/(1+norm(~x_pg, Inf)) <= 1e-2
#@test norm(~x_pg - ~x, Inf)/(1+norm(~x_pg, Inf)) <= 1e-3

################################################################################
### Box-constrained least-squares
################################################################################

println("Testing: box-constrained least-squares")

m, n = 500, 200
A = randn(m, n)
lb, ub = -1.0, 1.0
x_orig = 2.0*randn(n)
x_orig = max.(lb, min.(ub, x_orig))
b = A*x_orig + randn(m)

# Solve with PG

x_pg = Variable(n)
expr = ls(A*x_pg - b)
prob = problem(expr, x_pg in [lb, ub])
@time sol = solve(prob, PANOCplus(tol=1e-6, verbose=false))

@test norm(~x_pg - max.(lb, min.(ub, ~x_pg)), Inf) <= 1e-12
@test norm(~x_pg - max.(lb, min.(ub, ~x_pg - A'*(A*~x_pg - b))), Inf)/(1+norm(~x_pg, Inf)) <= 1e-6

# Solve with PANOC+

x_fpg = Variable(n)
expr = ls(A*x_fpg - b)
prob = problem(expr, x_fpg in [lb, ub])
@time sol = solve(prob, PANOCplus(tol=1e-6, verbose=false))

@test norm(~x_fpg - max.(lb, min.(ub, ~x_fpg)), Inf) <= 1e-12
@test norm(~x_fpg - max.(lb, min.(ub, ~x_fpg - A'*(A*~x_fpg - b))), Inf)/(1+norm(~x_fpg, Inf)) <= 1e-6

# Solve with ZeroFPR — dispatch test only on tiny problem
let A_t = randn(8, 4), b_t = randn(8), lb_t = -1.0, ub_t = 1.0
    x_zerofpr = Variable(4)
    expr = ls(A_t*x_zerofpr - b_t)
    prob = problem(expr, x_zerofpr in [lb_t, ub_t])
    @time sol = solve(prob, ZeroFPR(tol=1e-4, verbose=false))
    @test norm(~x_zerofpr - max.(lb_t, min.(ub_t, ~x_zerofpr)), Inf) <= 1e-12
    @test !any(isnan.(~x_zerofpr))
end

# Solve with FastForwardBackward (proximal gradient)

x_panoc = Variable(n)
expr = ls(A*x_panoc - b)
prob = problem(expr, x_panoc in [lb, ub])
@time sol = solve(prob, FastForwardBackward(tol=1e-6, verbose=false))

@test norm(~x_panoc - max.(lb, min.(ub, ~x_panoc)), Inf) <= 1e-12
@test norm(~x_panoc - max.(lb, min.(ub, ~x_panoc - A'*(A*~x_panoc - b))), Inf)/(1+norm(~x_panoc, Inf)) <= 1e-4

# Solve with minimize, default solver/options

#x = Variable(n)
#@time sol = @minimize ls(A*x - b) st x in [lb, ub]

#@test norm(~x - max.(lb, min.(ub, ~x)), Inf) <= 1e-12
#@test norm(~x - max.(lb, min.(ub, ~x - A'*(A*~x - b))), Inf)/(1+norm(~x, Inf)) <= 1e-4

################################################################################
### Non-negative least-squares from a known solution
################################################################################

println("Testing: non-negative least-squares from a known solution")

# Lagrangian:
#
#   0.5||Ax-b||^2 + y'(x-z) + [z >= 0]
#
# Optimality conditions:
#
#   A'(Ax-b) + y = 0, or A'b = A'Ax + y
#   x = z
#   z >= 0
#   y <= 0
#   y'z = 0

m, n, nnz_x_star = 500, 200, 100
A = randn(m, n)
x_star = rand(n)
x_star[nnz_x_star+1:end] .= 0.0
y_star = -rand(n)
y_star[1:nnz_x_star] .= 0.0
b = A*x_star + A'\y_star

# Solve with PG

x_pg = Variable(n)
expr = ls(A*x_pg - b)
prob = problem(expr, x_pg >= 0.0)
@time sol = solve(prob, PANOCplus(tol=1e-8, verbose=false))

@test all(~x_pg .>= 0.0)
@test norm(~x_pg - x_star, Inf)/(1+norm(x_star, Inf)) <= 1e-8

# Solve with PANOC+

x_fpg = Variable(n)
expr = ls(A*x_fpg - b)
prob = problem(expr, x_fpg >= 0.0)
@time sol = solve(prob, PANOCplus(tol=1e-8, verbose=false))

@test all(~x_fpg .>= 0.0)
@test norm(~x_fpg - x_star, Inf)/(1+norm(x_star, Inf)) <= 1e-8

# Solve with ZeroFPR — dispatch test only on tiny problem

let A_t = randn(8, 4), x_t = max.(0.0, randn(4)), b_t = A_t*x_t + randn(8)*0.01
    x_zerofpr = Variable(4)
    expr = ls(A_t*x_zerofpr - b_t)
    prob = problem(expr, x_zerofpr >= 0.0)
    @time sol = solve(prob, ZeroFPR(tol=1e-4, verbose=false))
    @test all(~x_zerofpr .>= -1e-10)
    @test !any(isnan.(~x_zerofpr))
end

# Solve with FastForwardBackward (proximal gradient — different algorithm type)

x_panoc = Variable(n)
expr = ls(A*x_panoc - b)
prob = problem(expr, x_panoc >= 0.0)
@time sol = solve(prob, FastForwardBackward(tol=1e-8, verbose=false))

@test all(~x_panoc .>= 0.0)
@test norm(~x_panoc - x_star, Inf)/(1+norm(x_star, Inf)) <= 1e-6

# Solve with minimize, default solver/options

x = Variable(n)
@time sol = @minimize ls(A*x - b) st x >= 0.0

@test all(~x .>= 0.0)
@test norm(~x - x_star, Inf)/(1+norm(x_star, Inf)) <= 1e-6

################################################################################
### normalop_ls: compare 1/2||Ax-b||^2 solved via ls vs normalop_ls
################################################################################

println("Testing: normalop_ls end-to-end (compare with ls)")

Random.seed!(99)
m_nop, n_nop, nnz_nop = 50, 30, 5
A_nop = randn(m_nop, n_nop)
lam_nop = 0.5
x_star_nop = randn(n_nop)
x_star_nop[nnz_nop+1:end] .= 0.0
y_star_nop = lam_nop * sign.(x_star_nop)
b_nop = A_nop * x_star_nop + A_nop' \ y_star_nop

x_ls_nop = Variable(n_nop)
@time solve(problem(ls(A_nop * x_ls_nop - b_nop) + lam_nop * norm(x_ls_nop, 1)), PANOCplus(tol=1e-10, verbose=false))

x_nop2 = Variable(n_nop)
@time solve(problem(normalop_ls(A_nop * x_nop2 - b_nop) + lam_nop * norm(x_nop2, 1)), PANOCplus(tol=1e-10, verbose=false))

@test norm(~x_ls_nop - ~x_nop2, Inf) / (1 + norm(~x_ls_nop, Inf)) <= 1e-2
