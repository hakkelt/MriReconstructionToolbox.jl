# Testing precomposition by nonlinear operator

b = randn(10)
g = SqrNormL2(3.0)
G = AffineAdd(AbstractOperators.Sigmoid((10,), 1.0),b,false)
f = StructuredOptimization.PrecomposeNonlinear(g, G)

x = randn(10)

grad_f_x, f_x = gradient(f, x)

@test size(grad_f_x) == size(x)
@test abs(f_x - 3.0/2 * norm(1.0 ./ (1.0 .+ exp.(-x)) - b)^2) <= 1e-10
expx = exp.(x)
expmx = 1.0./expx
grad_f_x_ref = 3.0 * ( expx ./ (1 .+ expx).^2 ) .* (1.0 ./ (1.0 .+ expmx) - b)
@test norm(grad_f_x - grad_f_x_ref) <= 1e-10

## with compose
#with vectors
l,m1,m2,n1,n2 = 2,3,4,5,6
x = ArrayPartition(randn(m1,m2),randn(n1,n2))
A = MatrixOp(randn(l,m1),m2)
B = MatrixOp(randn(m2,n1),n2)
r = randn(l,n2)

b = randn(l,n2)
G = AffineAdd(Ax_mul_Bx( 
              HCAT(A,Zeros(codomain_type(B), size(B,2), size(A,1) )), 
              HCAT(Zeros(codomain_type(A), size(A,2), size(B,1) ),B)
                       ), 
              b,false)

g = SqrNormL2(3.0)
f = StructuredOptimization.PrecomposeNonlinear(g, G)

x = ArrayPartition(randn(m1,m2),randn(n1,n2))

grad_f_x, f_x = gradient(f, x)

r = G*x 
grad_f_x2, f_x2 = gradient(g, r)
grad_f_x2 = jacobian(G,x)'*grad_f_x2

@test norm(f_x-f_x2) < 1e-8
@test norm(grad_f_x2.-grad_f_x2) < 1e-8

## SqrNormL2WithNormalOp
L_mat = randn(8, 5)
L = MatrixOp(L_mat)
xv = randn(5)
f_nop = StructuredOptimization.SqrNormL2WithNormalOp(L)
@test abs(f_nop(xv) - 0.5 * norm(L_mat * xv)^2) < 1e-10
yv = zero(xv)
fy = gradient!(yv, f_nop, xv)
@test norm(yv - L_mat' * (L_mat * xv)) < 1e-10
@test StructuredOptimization.is_convex(typeof(f_nop))
@test StructuredOptimization.is_smooth(typeof(f_nop))
@test StructuredOptimization.is_generalized_quadratic(typeof(f_nop))

# sqrNormL2WithNormalOp.jl — negative lambda error
let A = randn(5, 4)
    op = MatrixOp(A)
    @test_throws ErrorException StructuredOptimization.SqrNormL2WithNormalOp(op, -1.0)
end

