using ProximalAlgorithms: ZeroFPR, PANOC, PANOCplus, ADMM, CGNR

A = randn(3,5)
b = randn(3)

x_zfpr = Variable(5)
prob_zfpr = problem(ls(A*x_zfpr - b) + 1e-3*norm(x_zfpr, 1))
sol_zfpr = solve(prob_zfpr, ZeroFPR(maxit=10))
@test norm(A*(~x_zfpr) - b) <= norm(b)  # residual bounded after starting from zero

x_pnc = Variable(5)
prob_pnc = problem(ls(A*x_pnc - b) + 1e-3*norm(x_pnc, 1))
sol_pnc = solve(prob_pnc, PANOC(maxit=10))
@test !isnothing(sol_pnc)

x_pncp = Variable(5)
prob_pncp = problem(ls(A*x_pncp - b) + 1e-3*norm(x_pncp, 1))
sol_pncp = solve(prob_pncp, PANOCplus(maxit=10))
@test !isnothing(sol_pncp)

x_admm = Variable(5)
prob_admm = problem(ls(A*x_admm - b) + 1e-3*norm(x_admm, 1))
sol_admm = solve(prob_admm, ADMM(maxit=10))
@test !isnothing(sol_admm)

x_cg = Variable(5)
prob_cg = problem(ls(A*x_cg - b) + 1e-3*norm(x_cg, 2)^2)
sol_cg = solve(prob_cg, CGNR(maxit=10))
@test !isnothing(sol_cg)

# Signed correctness of the least-squares term for the solvers that consume it through the
# `LeastSquaresTerm` assumption (ADMM, CG, CGNR): `ls(A*x - b)` has displacement `-b`, while those
# solvers minimize `‖Ax - b‖²`, so the displacement must reach them negated. Getting this wrong
# returns `-x`, which has the same norm and residual magnitude as the solution and is therefore
# invisible to the existence and residual assertions above.
x_ridge = (A'A + 2e-3 * I) \ (A'b)  # minimizer of ½‖Ax - b‖² + 1e-3‖x‖²

x_cg_exact = Variable(5)
solve(problem(ls(A*x_cg_exact - b) + 1e-3*norm(x_cg_exact, 2)^2), CGNR(maxit=200))
@test ~x_cg_exact ≈ x_ridge rtol=1e-4

# ADMM is checked on an ℓ1 problem against a solver that does not go through `LeastSquaresTerm`
# (PANOCplus), on an overdetermined system so that the minimizer is unique. A fixed `rho` is used
# because the default adaptive penalty sequence stalls on this problem.
A_tall, b_tall = randn(6, 4), randn(6)

x_ref_l1 = Variable(4)
solve(problem(ls(A_tall*x_ref_l1 - b_tall) + 1e-2*norm(x_ref_l1, 1)), PANOCplus(maxit=2000))

x_admm_l1 = Variable(4)
solve(problem(ls(A_tall*x_admm_l1 - b_tall) + 1e-2*norm(x_admm_l1, 1)), ADMM(maxit=5000, rho=1.0))
@test ~x_admm_l1 ≈ ~x_ref_l1 rtol=1e-4
