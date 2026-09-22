# Testing precomposition by nonlinear operator

b = randn(10)
g = SqrNormL2(3.0)
G = AffineAdd(AbstractOperators.Sigmoid((10,), 1.0), b, false)
f = StructuredOptimization.PrecomposeNonlinear(g, G)

x = randn(10)

grad_f_x, f_x = gradient(f, x)

@test size(grad_f_x) == size(x)
@test abs(f_x - 3.0 / 2 * norm(1.0 ./ (1.0 .+ exp.(-x)) - b)^2) <= 1.0e-10
expx = exp.(x)
expmx = 1.0 ./ expx
grad_f_x_ref = 3.0 * (expx ./ (1 .+ expx) .^ 2) .* (1.0 ./ (1.0 .+ expmx) - b)
@test norm(grad_f_x - grad_f_x_ref) <= 1.0e-10

## with compose
#with vectors
l, m1, m2, n1, n2 = 2, 3, 4, 5, 6
x = ArrayPartition(randn(m1, m2), randn(n1, n2))
A = MatrixOp(randn(l, m1), m2)
B = MatrixOp(randn(m2, n1), n2)
r = randn(l, n2)

b = randn(l, n2)
G = AffineAdd(
    Ax_mul_Bx(
        HCAT(A, Zeros(codomain_type(B), size(B, 2), size(A, 1))),
        HCAT(Zeros(codomain_type(A), size(A, 2), size(B, 1)), B)
    ),
    b, false
)

g = SqrNormL2(3.0)
f = StructuredOptimization.PrecomposeNonlinear(g, G)

x = ArrayPartition(randn(m1, m2), randn(n1, n2))

grad_f_x, f_x = gradient(f, x)

r = G * x
grad_f_x2, f_x2 = gradient(g, r)
grad_f_x2 = jacobian(G, x)' * grad_f_x2

@test norm(f_x - f_x2) < 1.0e-8
@test norm(grad_f_x2 .- grad_f_x2) < 1.0e-8

## SqrNormL2WithNormalOp
L_mat = randn(8, 5)
L = MatrixOp(L_mat)
xv = randn(5)
f_nop = StructuredOptimization.SqrNormL2WithNormalOp(L)
@test abs(f_nop(xv) - 0.5 * norm(L_mat * xv)^2) < 1.0e-10
yv = zero(xv)
fy = gradient!(yv, f_nop, xv)
@test norm(yv - L_mat' * (L_mat * xv)) < 1.0e-10
# `gradient!` returns the function value, as `ProximalCore.value_and_gradient!` requires
@test abs(fy - 0.5 * norm(L_mat * xv)^2) < 1.0e-10
@test StructuredOptimization.is_convex(typeof(f_nop))
@test StructuredOptimization.is_smooth(typeof(f_nop))
@test StructuredOptimization.is_generalized_quadratic(typeof(f_nop))

# sqrNormL2WithNormalOp.jl — negative lambda error
let A = randn(5, 4)
    op = MatrixOp(A)
    @test_throws ErrorException StructuredOptimization.SqrNormL2WithNormalOp(op, -1.0)
end

# SqrNormL2WithNormalOp: the value `gradient!` returns must stay the potential of the
# gradient it actually computes, for a scalar λ, an array (weighted) λ, and an affine
# operator (where the normal operator carries a displacement). Checked against the
# closed form and against a finite-difference gradient.
@testset "SqrNormL2WithNormalOp value, λ=$lambda, T=$T, affine=$affine" for
    lambda in (1, 0.75, :array), T in (Float64, ComplexF64), affine in (false, true)
    A = randn(T, 7, 4)
    bvec = randn(T, 7)
    xv = randn(T, 4)
    op = affine ? AffineAdd(MatrixOp(A), bvec, false) : MatrixOp(A)
    lam = lambda === :array ? rand(7) .+ 0.1 : lambda
    f = StructuredOptimization.SqrNormL2WithNormalOp(op, lam)

    resid = affine ? A * xv - bvec : A * xv
    weighted_sqnorm = lam isa AbstractArray ? sum(lam[k] * abs2(resid[k]) for k in eachindex(resid)) : lam * norm(resid)^2
    fval = weighted_sqnorm / 2
    grad = lam isa AbstractArray ? A' * (lam .* resid) : lam * (A' * resid)

    # the callable and `gradient!` must agree with each other and with the closed form
    @test abs(f(xv) - fval) < 1.0e-9
    yv = zero(xv)
    @test abs(gradient!(yv, f, xv) - fval) < 1.0e-9
    @test norm(yv - grad) < 1.0e-9

    # finite differences on the real parametrization (the gradient is the Wirtinger
    # gradient w.r.t. conj(x), so a real perturbation probes 2*Re⟨grad, δ⟩ correctly)
    h = 1.0e-6
    for k in eachindex(xv)
        δ = zero(xv)
        δ[k] = h
        fd = (f(xv + δ) - f(xv - δ)) / (2h)
        @test abs(fd - real(grad[k])) < 1.0e-5 * max(1, abs(grad[k]))
        if T <: Complex
            δ[k] = h * im
            fd_im = (f(xv + δ) - f(xv - δ)) / (2h)
            @test abs(fd_im - imag(grad[k])) < 1.0e-5 * max(1, abs(grad[k]))
        end
    end
end

# SqrNormL2WithNormalOp with an operator whose `'` is not the true adjoint (a
# BACKWARD-normalized DFT: A' == A⁻¹ == Aᴴ/N). The value `gradient!` returns must
# still be the potential of the (rescaled) gradient it actually produces.
let n = 8
    op = FFTWOperators.DFT(Float64, (n,); normalization = FFTWOperators.BACKWARD)
    f = StructuredOptimization.SqrNormL2WithNormalOp(op)
    xv = randn(n)
    yv = zero(xv)
    fy = gradient!(yv, f, xv)
    @test abs(fy - f(xv)) < 1.0e-9
    h = 1.0e-6
    for k in eachindex(xv)
        δ = zero(xv)
        δ[k] = h
        fd = (f(xv + δ) - f(xv - δ)) / (2h)
        @test abs(fd - yv[k]) < 1.0e-4 * max(1, abs(yv[k]))
    end
end
