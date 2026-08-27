println("\nTesting cost terms\n")

# Simple Terms

x = Variable(randn(10))
X = Variable(randn(3,4))
A = randn(4,10)
b = randn(4)

cf = norm(x, 0)
@test cf.lambda == 1
@test cf.f(~x) == norm(~x,0)

cf = 3*norm(x, 0)
@test cf.lambda == 3
@test cf.f(~x) == norm(~x,0)

cf = norm(x, 0) <= 3
@test cf.lambda == 1
@test cf.f(~x) == (IndBallL0(3))(~x)

cf = norm(x, 1)
@test cf.lambda == 1
@test cf.f(~x) == (NormL1())(~x)

cf = norm(x, 1) <= 1.5
@test cf.lambda == 1
@test cf.f(~x) == (IndBallL1(1.5))(~x)

cf = 10*norm(x, 1) <= 1.5
@test cf.lambda == 1
@test cf.f(~x) == (IndBallL1(1.5/10))(~x)

cf = norm(x)
@test cf.lambda == 1
@test cf.f(~x) == norm(~x)

cf = pi*norm(x,2)
@test cf.lambda - pi == 0
@test cf.f(~x) == norm(~x)

cf = 3*norm(X,2,1)
@test cf.lambda - 3 == 0
@test cf.f(~X) == sum(  sqrt.(sum((~X).^2, dims=1 )) ) 

cf = 4*norm(X,2,1; dim=2)
@test cf.lambda - 4 == 0
@test cf.f(~X) == sum(  sqrt.(sum((~X).^2, dims=2 )) ) 

@test_throws ErrorException 4*norm(X,1,2)

cf = norm(x, 2) <= 2.3
@test cf.lambda == 1
@test cf.f(~x) == (IndBallL2(2.3))(~x)

cf = norm(x, 2) == 2.3
@test cf.lambda == 1
@test cf.f(~x) == (IndSphereL2(2.3))(~x)

cf = norm(x, Inf)
@test cf.lambda == 1
@test cf.f(~x) == norm(~x,Inf)

cf = norm(x, Inf) <= 5.0
@test cf.lambda == 1
@test cf.f(~x) == (IndBallLinf(5.0))(~x)

cf = x <= 3.0
@test cf.lambda == 1
@test cf.f(~x) == (IndBox(-Inf, 3.0))(~x)

cf = 3.0 <= x
@test cf.lambda == 1
@test cf.f(~x) == (IndBox(3.0, Inf))(~x)

cf = x >= 1.0
@test cf.lambda == 1
@test cf.f(~x) == (IndBox(1.0, Inf))(~x)

cf = 1.0 >= x
@test cf.lambda == 1
@test cf.f(~x) == (IndBox(-Inf,1.0))(~x)

cf = x in [-5.0, 5.0]
@test cf.lambda == 1
@test cf.f(~x) == (IndBox(-5.0, 5.0))(~x)

cf = norm(x, 2)^2
@test cf.lambda == 1
@test cf.f(~x) == norm(~x)^2

cf = 0.5*norm(x, 2)^2
@test cf.lambda == 0.5
@test cf.f(~x) == norm(~x)^2

cf = 7*(0.5*norm(x, 2))^2
@test cf.lambda == 7*0.25
@test cf.f(~x) == norm(~x)^2

cf = 2*rank(X) <= 6
@test cf.lambda == 1
@test cf.f(~X) == (IndBallRank(3))(~X)

cf = rank(X)
@test_throws MethodError cf.f(~X)

cf = norm(X,*)
U, S, V = svd(~X)
@test cf.lambda == 1
@test cf.f(~X) == sum(S)

y = randn(size(~x))
cf = hingeloss(x,y)
@test cf.lambda == 1
@test cf.f(~x) == (HingeLoss(y))(~x)

y = randn(size(~x))
cf = sqrhingeloss(x,y)
@test cf.lambda == 1
@test cf.f(~x) == (SqrHingeLoss(y))(~x)

y = randn(size(~x))
cf = logisticloss(x,y)
@test cf.lambda == 1
@test cf.f(~x) == (LogisticLoss(y))(~x)

xp = Variable(rand(10)) 
bp = rand(Float64, size(~xp))
cf = crossentropy(xp,bp)
@test cf.lambda == 1
@test cf.f(~xp) == (CrossEntropy(bp))(~xp)

cf = logbarrier(x)
@test cf.lambda == 1
@test cf.f(~x) == (LogBarrier(1.0))(~x)

cf = maximum(x)
@test cf.lambda == 1
@test cf.f(~x) == (Maximum(1.0))(~x)

cf = sumpositive(x)
@test cf.lambda == 1
@test cf.f(~x) == (SumPositive())(~x)

a = 1.
cf = huberloss(x,a)
@test cf.lambda == 1
@test cf.f(~x) == (HuberLoss(a))(~x)

a = randn(size(x))
cf = dot(a,x)
@test cf.lambda == 1
@test cf.f(~x) == (Linear(a))(~x)

#IndBinary
lu = (-1.0,randn(length(~x)))
cf = x == lu
@test cf.lambda == 1
@test cf.f(~x) == (IndBinary(lu...))(~x)

# IndAffine
cf = A*x-b == 0
@test cf.lambda == 1
@test cf.f(~x) == (IndAffine(A,b))(~x)

cf = (A*x == b)
@test cf.lambda == 1
@test cf.f(~x) == (IndAffine(A,-b))(~x)

cf = 2*norm(x,1)
ccf = conj(cf)
@test ccf.A == cf.A
@test ccf.f == Conjugate(Postcompose(NormL1(),2.0))
@test_throws ErrorException conj(norm(randn(2,10)*x,1))

cf = 2*norm(x,1)
ccf = smooth(cf,2.0)
@test ccf.A == cf.A
@test ccf.f(~x) == MoreauEnvelope(Postcompose(NormL1(),2),2.0)(~x)

# Summing terms

x = Variable(10)
cf = ls(x) + 10*norm(x, 1)
@test cf[1].lambda == 1
@test cf[1].f(~x) == 0.5*norm(~x)^2
@test cf[2].lambda == 10
@test cf[2].f(~x) == norm(~x,1)

# More complex situations

x = Variable(10)
A = randn(5, 10)
y = Variable(7)
B = randn(5, 7)
b = randn(5)

cf = ls(A*x - b) + norm(x, 1)
@test cf[1].lambda == 1
@test cf[1].f(~x) == 0.5*norm(~x)^2
@test norm(affine(cf[1])*(~x) - (A*(~x)-b)) < 1e-12
@test cf[2].lambda == 1
@test cf[2].f(~x) == norm(~x,1)

cf = ls(A*x - B*y + b) + norm(y, 1) + 5*norm(y, 2)
@test cf[1].lambda == 1
@test cf[1].f(~x) == 0.5*norm(~x)^2
@test cf[2].lambda == 1
@test cf[2].f(~x) == norm(~x,1)
@test cf[3].lambda == 5
@test cf[3].f(~x) == norm(~x,2)

cf = 10*(ls(A*x - B*y + b) + norm(y, 1) + 5*norm(y, 2))
@test cf[1].lambda == 10
@test cf[1].f(~x) == 0.5*norm(~x)^2
@test cf[2].lambda == 10
@test cf[2].f(~x) == norm(~x,1)
@test cf[3].lambda == 50
@test cf[3].f(~x) == norm(~x,2)

cf = 0.5*norm(A*x - B*y + b, 2)^2 + norm(x, 1) + norm(y, 2)
@test cf[1].lambda == 0.5
@test cf[1].f(~x) == norm(~x)^2
@test cf[2].lambda == 1
@test cf[2].f(~x) == norm(~x,1)
@test cf[3].lambda == 1
@test cf[3].f(~x) == norm(~x,2)

# Properties
A = randn(5, 10)
u = Variable(5)
w = Variable(5)
z = Variable(5)

cf = norm(A*x + z)
@test StructuredOptimization.is_smooth(cf) == false
@test StructuredOptimization.is_smooth(cf^2) == true

cf = norm(w + z)^2
@test StructuredOptimization.is_smooth(cf) == true
@test StructuredOptimization.is_AcA_diagonal(cf) == false

cf = norm(x, 1) + norm(y, 2)
@test StructuredOptimization.is_smooth.(cf.terms) == (false,false)
@test StructuredOptimization.is_smooth(cf) == false
@test StructuredOptimization.is_AcA_diagonal.(cf.terms) == (true,true)
@test StructuredOptimization.is_AcA_diagonal(cf) == true

# normalop_ls
A2 = randn(5, 10)
x2 = Variable(10)
ex = A2 * x2
t_nls = normalop_ls(ex)
@test t_nls.f isa StructuredOptimization.SqrNormL2WithNormalOp
@test_throws ErrorException normalop_ls(x2)

# normalop_ls with a multi-variable expression: previously a confusing
# `MethodError: no method matching length(::AffineAdd{HCAT{...}})`.
let y2 = Variable(10)
    t_nls_multi = normalop_ls(A2 * x2 + A2 * y2)
    @test t_nls_multi.f isa StructuredOptimization.SqrNormL2WithNormalOp
    @test StructuredOptimization.is_strongly_convex(t_nls_multi) == false

    # gradient matches the plain-ls formulation exactly
    op_multi = StructuredOptimization.extract_operators((x2, y2), t_nls_multi)
    @test AbstractOperators.is_eye(op_multi)
    xv, yv = randn(10), randn(10)
    gy = ArrayPartition(zeros(10), zeros(10))
    StructuredOptimization.gradient!(gy, t_nls_multi.f, ArrayPartition(xv, yv))
    expected = A2' * (A2 * (xv + yv))
    @test gy.x[1] ≈ expected
    @test gy.x[2] ≈ expected

    # end-to-end: solving with normalop_ls reaches the same minimizer as ls
    nrmA2 = opnorm(A2)
    b2 = randn(5)
    x2a, y2a = Variable(10), Variable(10)
    p_nop = problem(normalop_ls(A2 * x2a + A2 * y2a - b2), 0.05 * norm(x2a, 1), 0.05 * norm(y2a, 2))
    solve(p_nop, ProximalAlgorithms.FastForwardBackward(Lf = 2 * nrmA2^2, maxit = 2000, tol = 1.0e-10))
    x2b, y2b = Variable(10), Variable(10)
    p_ls2 = problem(ls(A2 * x2b + A2 * y2b - b2), 0.05 * norm(x2b, 1), 0.05 * norm(y2b, 2))
    solve(p_ls2, ProximalAlgorithms.FastForwardBackward(Lf = 2 * nrmA2^2, maxit = 2000, tol = 1.0e-10))
    @test ~x2a ≈ ~x2b atol = 1.0e-4
    @test ~y2a ≈ ~y2b atol = 1.0e-4
end

# HCAT normal-op fusion: when every block is the *same* operator (the shared-
# encoding-operator multi-component case, e.g. 𝒜*(x+y)), normalop_ls must
# reuse 𝒜's own fast normal operator instead of applying 𝒜 once per block.
let Ashared = MatrixOp(randn(8, 6)), xs = Variable(6), ys = Variable(6)
    ex_shared = Ashared * xs + Ashared * ys
    @test AbstractOperators.has_optimized_normalop(ex_shared.L)
    nop = AbstractOperators.get_normal_op(ex_shared.L)
    av, bv2 = randn(6), randn(6)
    lhs = nop * ArrayPartition(av, bv2)
    rhs = ex_shared.L' * (ex_shared.L * ArrayPartition(av, bv2))
    @test lhs.x[1] ≈ rhs.x[1]
    @test lhs.x[2] ≈ rhs.x[2]

    bsh = randn(8)
    nrmAsh = opnorm(Ashared)
    xs2, ys2 = Variable(6), Variable(6)
    p_shared = problem(normalop_ls(Ashared * xs2 + Ashared * ys2 - bsh), 0.05 * norm(xs2, 1), 0.05 * norm(ys2, 2))
    solve(p_shared, ProximalAlgorithms.FastForwardBackward(Lf = 2 * nrmAsh^2, maxit = 2000, tol = 1.0e-10))
    xs3, ys3 = Variable(6), Variable(6)
    p_ls_shared = problem(ls(Ashared * xs3 + Ashared * ys3 - bsh), 0.05 * norm(xs3, 1), 0.05 * norm(ys3, 2))
    solve(p_ls_shared, ProximalAlgorithms.FastForwardBackward(Lf = 2 * nrmAsh^2, maxit = 2000, tol = 1.0e-10))
    @test ~xs2 ≈ ~xs3 atol = 1.0e-4
    @test ~ys2 ≈ ~ys3 atol = 1.0e-4
end

# A distinct-operator HCAT has no such fusion: no false positive.
let A_a = randn(5, 10), A_b = randn(5, 10)
    H_distinct = HCAT(MatrixOp(A_a), MatrixOp(A_b))
    @test !AbstractOperators.has_optimized_normalop(H_distinct)
end

# IndBallL2 must be marked proximable (needed for multi-variable parsing)
@test StructuredOptimization.is_proximable(IndBallL2)
@test StructuredOptimization.is_proximable(IndBallL2{Float64})
@test StructuredOptimization.is_proximable(norm(x, 2) <= 1.0)

# Properties: separable iff diagonal operator
@test StructuredOptimization.is_separable(norm(x, 1))
@test !StructuredOptimization.is_separable(norm(A*x, 1))

# Properties: strongly convex iff full column rank operator
A_tall = randn(15, 10)
@test StructuredOptimization.is_strongly_convex(ls(A_tall * x2))  # tall matrix → full col rank
@test !StructuredOptimization.is_strongly_convex(ls(A2 * x2))     # fat matrix → not full col rank

# Properties: generalized quadratic
@test StructuredOptimization.is_generalized_quadratic(ls(x2))

# Term + TermSet combinator
let A = randn(5, 4), b = randn(5), c = randn(4)
    x = Variable(4)
    t1 = ls(A*x - b)
    t2 = norm(x, 1)
    ts = t1 + t2
    t3 = dot(c, x)
    ts2 = t3 + ts
    @test ts2 isa StructuredOptimization.TermSet
    @test length(ts2) == 3
end

# proximalOperators_bind.jl — error branches
let x = Variable(4)
    @test_throws ErrorException norm(x, 3)
    @test_throws ErrorException (x in [1.0, 2.0, 3.0])
    x_c = Variable(zeros(ComplexF64, 4))
    ex = fft(x_c)
    @test_throws ErrorException (ex == 0.0)
end

# proximalOperators_bind.jl — normalop_ls with single-variable expression
let A = randn(8, 4), b = randn(8)
    x = Variable(4)
    ~x .= 0.0
    ex = A*x - b
    t = normalop_ls(ex)
    @test t isa StructuredOptimization.Term
    prob = problem(t)
    algs = StructuredOptimization.suggest_algorithm(prob)
    @test !isempty(algs)
    sol = solve(prob, ProximalAlgorithms.PANOCplus(tol=1e-6))
    @test !isnothing(sol)
    x_true = A'*A\(A'*b)
    @test norm(~x - x_true, Inf) / (1 + norm(x_true, Inf)) <= 5e-4
end

# is_proximable returning false (overlapping variables between two terms)
let
    x = Variable(4)
    t1 = ls(x)
    t2 = norm(x, 1)
    ts = problem(t1 + t2)
    @test ts isa StructuredOptimization.TermSet
    @test all(StructuredOptimization.is_proximable.(ts))
    @test !StructuredOptimization.is_separable_sum(ts)
    @test !StructuredOptimization.is_proximable(ts)
end

# is_separable_sum — sliced non-overlapping terms
let
    x = Variable(6)
    t1 = norm(x[1:3], 1)
    t2 = norm(x[4:6], 1)
    ts = problem(t1 + t2)
    @test ts isa StructuredOptimization.TermSet
    @test all(StructuredOptimization.is_proximable.(ts))
    @test StructuredOptimization.is_separable_sum(ts)
    @test StructuredOptimization.is_proximable(ts)
end

let
    x = Variable(6)
    t1 = norm(x[1:4], 1)
    t2 = norm(x[3:6], 1)
    ts = problem(t1 + t2)
    @test ts isa StructuredOptimization.TermSet
    @test all(StructuredOptimization.is_proximable.(ts))
    @test !StructuredOptimization.is_separable_sum(ts)
end

