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
