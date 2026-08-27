println("\nTesting extraction from Terms\n")

# testing extracting stuff from terms

m,n1 = 5,3
x1 = Variable(n1)
A = randn(m,n1)
# single term, single variable
cf = ls(A*x1)
xAll = StructuredOptimization.extract_variables(cf)
@test xAll[1] == x1
L = StructuredOptimization.extract_operators(xAll,cf)
@test typeof(L) <: MatrixOp
La = StructuredOptimization.extract_affines(xAll,cf)
@test typeof(La) <: MatrixOp
f = StructuredOptimization.extract_functions(cf)
@test typeof(f) <: SqrNormL2

# multiple terms, single variable
b1 = randn(n1)
cf = ls(A*x1) + 2.5*norm(x1+b1,1)
xAll = StructuredOptimization.extract_variables(cf)
@test xAll[1] == x1
V = StructuredOptimization.extract_operators(xAll,cf)
@test typeof(V) <: VCAT
@test typeof(V[1]) <: MatrixOp
@test typeof(V[2]) <: Eye
V2 = StructuredOptimization.extract_affines(xAll,cf)
@test typeof(V2) <: VCAT
@test typeof(V2[1]) <: MatrixOp
@test typeof(V2[2]) <: AffineAdd{T} where {T <: Eye}
f = StructuredOptimization.extract_functions(cf)
@test typeof(f) <: SeparableSum
@test typeof(f.fs[1]) <: SqrNormL2
@test typeof(f.fs[2]) <: Postcompose
x = randn(n1)
@test norm(f.fs[2](x) - 2.5*norm(x+b1,1)) < 1e-12

# single term, multiple variables
x2 = Variable(m)
cf = ls(A*x1+x2+20)
xAll = StructuredOptimization.extract_variables(cf)
xAll = (x2,x1) # change the order on pourpose
H = StructuredOptimization.extract_operators(xAll,cf)
@test typeof(H) <: HCAT
@test typeof(H[1]) <: Eye
@test typeof(H[2]) <: MatrixOp
H2 = StructuredOptimization.extract_affines(xAll,cf)
@test typeof(H2[1]) <: AffineAdd{T} where {T <: Eye}
@test typeof(H2[2]) <: AffineAdd{T} where {T <: MatrixOp}
f = StructuredOptimization.extract_functions(cf)
@test typeof(f) <: PrecomposeDiagonal

### multiple terms, multiple variables
n1,n2,n3,n4,n5 = 3,3,4,4,7
A = randn(n5,n1)
x1,x2,x3,x4,x5 = Variable(randn(n1)),Variable(randn(n2)),Variable(randn(n3)),Variable(randn(n4)),Variable(randn(n5))

cf = ls(x1+x2)
xAll = StructuredOptimization.extract_variables(cf)
@test xAll == (x1,x2)

cf = ls(x1+x2)+ls(x1)
xAll = StructuredOptimization.extract_variables(cf)
@test xAll == (x1,x2)

cf = ls(x1+x2)+ls(x3+x4)+ls(x5)+ls(x5+A*x2)+ls(x1)+ls(x5)
xAll = StructuredOptimization.extract_variables(cf)
@test xAll == (x1,x2,x3,x4,x5)

V = StructuredOptimization.extract_operators(xAll,cf)

@test typeof(V[1][1]) <: Eye
@test typeof(V[1][2]) <: Eye
@test typeof(V[1][3]) <: Zeros
@test typeof(V[1][4]) <: Zeros
@test typeof(V[1][5]) <: Zeros

@test typeof(V[2][1]) <: Zeros
@test typeof(V[2][2]) <: Zeros
@test typeof(V[2][3]) <: Eye
@test typeof(V[2][4]) <: Eye
@test typeof(V[2][5]) <: Zeros

@test typeof(V[3][1]) <: Zeros
@test typeof(V[3][2]) <: Zeros
@test typeof(V[3][3]) <: Zeros
@test typeof(V[3][4]) <: Zeros
@test typeof(V[3][5]) <: Eye

@test typeof(V[4][1]) <: Zeros
@test typeof(V[4][2]) <: MatrixOp
@test typeof(V[4][3]) <: Zeros
@test typeof(V[4][4]) <: Zeros
@test typeof(V[4][5]) <: Eye

@test typeof(V[5][1]) <: Eye
@test typeof(V[5][2]) <: Zeros
@test typeof(V[5][3]) <: Zeros
@test typeof(V[5][4]) <: Zeros
@test typeof(V[5][5]) <: Zeros

@test typeof(V[6][1]) <: Zeros
@test typeof(V[6][2]) <: Zeros
@test typeof(V[6][3]) <: Zeros
@test typeof(V[6][4]) <: Zeros
@test typeof(V[6][5]) <: Eye

# minimize.jl — error in problem()
@test_throws ErrorException problem(42)

# parse.jl — diagonal operator with SqrNormL2 and other functions
let n = 4
    d = abs.(randn(n)) .+ 0.1   # positive diagonal
    x = Variable(n)
    ~x .= 0.0

    t_sq = ls(d.*x)
    prob_sq = problem(t_sq)
    algs = StructuredOptimization.suggest_algorithm(prob_sq)
    @test !isempty(algs)
    sol_sq = solve(prob_sq, ProximalAlgorithms.PANOCplus(tol=1e-6))
    @test !isnothing(sol_sq)

    x2 = Variable(n)
    ~x2 .= 0.0
    t_n1 = norm(d.*x2, 1)
    t_smooth = ls(x2)
    prob_n1 = problem(t_n1 + t_smooth)
    algs_n1 = StructuredOptimization.suggest_algorithm(prob_n1)
    @test !isempty(algs_n1)
    sol_n1 = solve(prob_n1, ProximalAlgorithms.PANOCplus(tol=1e-6))
    @test !isnothing(sol_n1)
end
