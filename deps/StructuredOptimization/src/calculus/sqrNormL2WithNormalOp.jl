# squared L2 norm (times a constant, or weighted) precomposed with an operator

"""
    SqrNormL2WithNormalOp(L::AbstractOperator, λ=1)

With a nonnegative scalar `λ`, return the squared Euclidean norm
```math
f(x) = \\tfrac{λ}{2σ}\\|L * x\\|^2,
```
where `σ` is the adjoint scaling of `L` described below (`σ = 1`, and the factor disappears,
whenever `L'` is the true adjoint of `L`).
With a nonnegative array `λ`, return the weighted squared Euclidean norm
```math
f(x) = \\tfrac{1}{2σ}∑_i λ_i y_i^2 where y = L * x.
```

This is a special case of the more general `Precompose(SqrNormL2(), L, 1, 0)` operator,
where `L` is a linear operator, and only the gradient is needed, not the proximal operator.
The gradient of the precomposed squared norm is
```math
\\nabla f(x) = Lᴴ * L * x,
```
and in many cases, there is an optimized implementation of the normal operator `Lᴴ * L`
that makes the computation of the gradient much faster than the naive implementation.

`L` may be affine (an `AffineAdd`, as produced by `ls(A*x - b)`): writing `L*x = A*x + d`,
the normal operator carries the displacement `Aᴴd` automatically (`Lᴴ*L*x = AᴴA*x + Aᴴd`
when `L*0 = d`), so `gradient!` computes the correct gradient in a single pass.

`gradient!` returns the function value `f(x)`, as `ProximalCore.value_and_gradient!`
requires, recovered from the gradient without a second application of `L`.

# Adjoint scaling

`L'` is not always the true adjoint of `L`. A `BACKWARD`-normalized DFT, for instance, has
`L' = L⁻¹ = Lᴴ/N`: the pair is off by a positive scalar `σ` defined by
```math
\\mathrm{Re}⟨L u, L u⟩ = σ \\, \\mathrm{Re}⟨u, (L'L) u⟩ .
```
Since `Lᴴ*L*x` (as actually computed from `L'*L`) is then `1/σ` times the true gradient of
`f`, the value returned alongside it must be scaled the same way for the two to be
consistent — otherwise anything that reads both (a backtracking line search, a printed
objective) is meaningless. `σ` is measured once, at construction, with a single probe
through `L` and `L'L`; with a genuine adjoint it is `1` and every formula above reduces to
the usual one.
"""
struct SqrNormL2WithNormalOp{T, SC, L <: AbstractOperator, L2 <: AbstractOperator, D, R <: Real}
    A::L
    # Normal operator used for the gradient. For scalar λ it is AᴴA (the weight is
    # applied afterwards); for array λ it is the *weighted* normal operator
    # Aᴴ·diag(λ)·A, so the gradient Aᴴ·diag(λ)·A·x is computed in one mul!.
    AᴴA::L2
    lambda::T
    # `Aᴴd`: the normal operator's displacement (`AᴴA * 0`), taken through the same,
    # possibly weighted, operator `gradient!` uses, or `nothing` when `A` is purely
    # linear (the overwhelmingly common case), so the per-gradient correction is
    # skipped entirely rather than paying a dot product with zeros.
    Aᴴd::D
    # The constant term of the quadratic, `‖d‖²/(2σ)` (weighted by λ when λ is an array).
    half_sqnorm_d::R
    # `1/σ`, the adjoint scaling of `A` (see the docstring); `1` for a true adjoint pair.
    inv_scaling::R
    function SqrNormL2WithNormalOp(A, lambda; pureAᴴA = nothing)
        @assert A isa AbstractOperator
        @assert is_linear(A)
        if any(lambda .< 0)
            error("coefficients in λ must be nonnegative")
        end
        # Strong convexity of x ↦ ½‖diag(√λ)·A·x‖² needs a positive weight *and* an
        # injective operator (full column rank), otherwise the null space of A is flat.
        strongly_convex = all(lambda .> 0) && is_full_column_rank(A)
        # Built unweighted, purely to measure the adjoint scaling below: that scaling is a
        # property of the (A, A') pair alone and is unaffected by inserting a Hermitian,
        # positive weight between them. A caller that already holds an operator equal to
        # `A' * A` — because it had to build one to decide whether folding `A` into the
        # function is worthwhile at all, see `fused_normal_op` — passes it in rather than
        # paying for the product twice.
        pureAᴴA = pureAᴴA === nothing ? A' * A : pureAᴴA
        if lambda isa AbstractArray
            W = AbstractOperators.DiagOp(AbstractOperators.codomain_type(A), size(A, 1), lambda)
            AᴴA = A' * W * A
        else
            AᴴA = lambda == 1 ? pureAᴴA : lambda * pureAᴴA
        end
        # `A * 0` is the displacement `d` of an affine `A` (zero for a purely linear one);
        # `AᴴA * 0` is `Aᴴd` taken through the very operator `gradient!` uses, so the
        # constants cannot drift from it.
        z = AbstractOperators.allocate_in_domain(A)
        fill!(z, 0)
        d = A * z
        has_displacement = !iszero(d)
        Aᴴd = has_displacement ? AᴴA * z : nothing
        inv_scaling = _inv_adjoint_scaling(A, pureAᴴA, z, d, has_displacement ? pureAᴴA * z : nothing)
        R_ = typeof(inv_scaling)
        half_sqnorm_d = has_displacement ? R_(_weighted_sqnorm(lambda, d) * inv_scaling / 2) : zero(R_)
        return new{typeof(lambda), strongly_convex, typeof(A), typeof(AᴴA), typeof(Aᴴd), R_}(
            A, AᴴA, lambda, Aᴴd, half_sqnorm_d, inv_scaling
        )
    end
end

_weighted_sqnorm(lambda::Real, d) = lambda * real(dot(d, d))
function _weighted_sqnorm(lambda::AbstractArray, d)
    R = real(eltype(d))
    return R(sum(real.(lambda .* abs2.(d))))
end

# `σ` from the docstring, as `1/σ`: `Re⟨A u, A u⟩ / Re⟨u, (A'A) u⟩` for a probe `u`, with the
# displacement of an affine `A` subtracted so that only the linear parts are compared.
#
# The probe is the constant vector, which is deterministic (no RNG dependency, so the value a
# solver prints does not move between runs) and is annihilated by no operator this is used
# with. Should it nevertheless land in the null space, `Aᴴd` — nonzero exactly when there is a
# displacement to correct — is tried next; if that fails too the scaling is left at 1, which is
# the behaviour of a true adjoint pair.
function _inv_adjoint_scaling(A, AᴴA, z, d, Aᴴd)
    R = real(eltype(z))
    u = similar(z)
    for probe in 1:2
        if probe == 1
            fill!(u, one(eltype(z)))
        elseif Aᴴd !== nothing
            copyto!(u, Aᴴd)
        else
            break
        end
        Au = A * u
        w = AᴴA * u
        # strip the affine displacement: `A u = A_lin u + d` and `(A'A) u = (A'A)_lin u + Aᴴd`
        if Aᴴd !== nothing
            Au = Au .- d
            w = w .- Aᴴd
        end
        num = real(dot(Au, Au))
        den = real(dot(u, w))
        isfinite(num) && isfinite(den) && den > 0 && return R(den / num)
    end
    return one(R)
end

is_convex(::Type{<:SqrNormL2WithNormalOp}) = true
is_smooth(::Type{<:SqrNormL2WithNormalOp}) = true
# Only the gradient is implemented. The default would infer proximability from convexity
# and let a solver that needs a prox be selected, which would then fail at the first
# iteration; the whole point of this function is to be the *smooth* formulation.
is_proximable(::Type{<:SqrNormL2WithNormalOp}) = false
is_separable(::Type{<:SqrNormL2WithNormalOp}) = true
is_generalized_quadratic(::Type{<:SqrNormL2WithNormalOp}) = true
is_strongly_convex(::Type{<:SqrNormL2WithNormalOp{T, SC}}) where {T, SC} = SC

SqrNormL2WithNormalOp(A) = SqrNormL2WithNormalOp(A, 1)

function (f::SqrNormL2WithNormalOp)(x)
    y = f.A * x
    return _weighted_sqnorm(f.lambda, y) * f.inv_scaling / 2
end

function gradient!(y, f::SqrNormL2WithNormalOp, x)
    mul!(y, f.AᴴA, x)
    v = real(dot(x, y)) / 2
    if f.Aᴴd !== nothing
        v += real(dot(x, f.Aᴴd)) / 2 + f.half_sqnorm_d
    end
    return v
end

"""
    fused_normal_op(L::AbstractOperator)

Return `Lᴴ * L` for a *linear* `L` when that product is genuinely cheaper than applying `L`
and then `Lᴴ`, and `nothing` when it is not.

This is the applicability test for `SqrNormL2WithNormalOp`. Two things make the product
cheaper. Either it collapses into a single operator — a `MatrixOp` into its Gram matrix, a
`DiagOp` into the squared diagonal, an FFT-based convolution into a single multiplication in
the frequency domain, or whatever specialised product a downstream package defines for its own
operator type — or `AbstractOperators` says so outright through `has_optimized_normalop`, in
which case `L' * L` *is* the optimized form it advertises. The second case need not collapse
to a single operator: `get_normal_op(::Compose)` fuses only the innermost adjoint pair and
keeps the outer factors, so an MRI encoding operator `S`-then-`F` becomes `Sᴴ·(FᴴF)·S` — one
transform where the naive form needs two, but still a `Compose`. A `Compose` with no such
advertisement means no specialised product exists, and the fold would add the value-recovery
bookkeeping without saving a pass.

Collapsing is not on its own enough to make the normal operator the cheaper of the two, so a
`L` that only collapses must also map into a codomain at least as large as its domain (see
[`normal_op_worthwhile`](@ref)). That size test is a dense-matrix estimate, and it is *not*
applied to an operator that advertises an optimized normal operator: there the operator itself
has answered the question, and the estimate would veto exactly the structured cases it cannot
model (a subsampled Fourier encoding maps into a smaller codomain than its domain, and its
normal operator is still the cheaper of the two).

`L` must carry no displacement; [`with_normal_op`](@ref) re-attaches it to the result.
"""
function fused_normal_op(L::AbstractOperator)
    (is_linear(L) && !is_eye(L)) || return nothing
    AbstractOperators.has_optimized_normalop(L) && return L' * L
    normal_op_worthwhile(L) || return nothing
    LᴴL = L' * L
    return LᴴL isa AbstractOperators.Compose ? nothing : LᴴL
end

"""
    normal_op_worthwhile(L::AbstractOperator)

Whether it is worth even *trying* to replace `L` by its normal operator: `L` has to be
linear, not already the identity, and map into a codomain at least as large as its domain.

The last condition is what rules out an underdetermined `L`. `LᴴL` acts on the domain, so
applying it costs on the order of `prod(size(L, 2))^2` against the `2·prod(size(L, 1))·
prod(size(L, 2))` of applying `L` and then `Lᴴ` — the normal operator only wins once the
domain is the smaller of the two spaces. Forming it also squares the condition number, and
on a wide `L` that is paid for nothing. A least-squares term over several variables is the
usual way to end up wide, since its domain is the sum of the blocks' domains.
"""
normal_op_worthwhile(L::AbstractOperator) =
    is_linear(L) && !is_eye(L) && _total_length(size(L, 2)) <= _total_length(size(L, 1))

# `size(op, i)` is a plain size tuple for a single-block operator and a tuple of such
# tuples for a block operator (`HCAT`, `VCAT`), so count the elements of either shape.
_total_length(size_::Tuple{Vararg{Int}}) = prod(size_)
_total_length(size_::Tuple) = sum(_total_length, size_)

# The normal operator of an `HCAT` is the block Gram `[Lᵢᴴ Lⱼ]`, assembled as a `VCAT` of
# `HCAT` rows so that it maps the joint `ArrayPartition` domain onto itself. `Lᴴ * L` does
# not fuse this on its own, which is why multi-variable terms would otherwise never qualify
# — their operator is always an `HCAT`, one block per variable.
#
# Only worth it when *every* one of the N² block products fuses: the block form costs N²
# applications against the 2N of applying the `HCAT` and its adjoint in turn, so a single
# block left as a `Compose` already makes it the more expensive of the two.
function fused_normal_op(L::AbstractOperators.HCAT)
    normal_op_worthwhile(L) || return nothing
    rows = ()
    for Li in L.A
        row = ()
        for Lj in L.A
            Nij = Li' * Lj
            Nij isa AbstractOperators.Compose && return nothing
            row = (row..., Nij)
        end
        rows = (rows..., AbstractOperators.HCAT(row...))
    end
    return AbstractOperators.VCAT(rows...)
end

"""
    with_normal_op(f, op, disp, λ)

Return the `SqrNormL2WithNormalOp` equivalent of `λ * f(op * x + disp)`, or `nothing` when
that rewrite does not apply.

It applies when `f` is a squared ``\\ell_2`` norm with a scalar weight and the linear `op`
has a fused normal operator (see [`fused_normal_op`](@ref)). `op` and `disp` are absorbed
into the returned function, whose domain is then `op`'s domain, so the caller must drop the
operator it passed in rather than composing with it again.
"""
with_normal_op(f, op, disp, λ) = nothing
function with_normal_op(f::SqrNormL2, op::AbstractOperator, disp, λ)
    (λ isa Real && f.lambda isa Real) || return nothing
    has_disp = !(disp isa Number && iszero(disp))
    # A scalar displacement has no array to push through `opᴴ`, and is not something the
    # expression layer produces for a least-squares term anyway.
    (has_disp && !(disp isa AbstractArray)) && return nothing
    LᴴL = fused_normal_op(op)
    LᴴL === nothing && return nothing
    # `op*x + disp` has normal operator `x ↦ opᴴ(op*x + disp) = (opᴴop)x + opᴴdisp`; the
    # constructor reads the displacement back out of it, so it must be attached here.
    A = has_disp ? AbstractOperators.AffineAdd(op, disp) : op
    AᴴA = has_disp ? _tilt_normal_op(LᴴL, op' * disp) : LᴴL
    return SqrNormL2WithNormalOp(A, λ * f.lambda; pureAᴴA = AᴴA)
end

# Attach the displacement `Aᴴd` to a normal operator. A block Gram is tilted row by row:
# its codomain is an `ArrayPartition`, and `AffineAdd` compares `size(d)` — a flat length
# for an `ArrayPartition` — against the operator's codomain size, which for a `VCAT` is a
# tuple of block sizes, so wrapping the whole thing would be rejected. Each row has an
# ordinary array codomain and takes the matching block of `d`.
_tilt_normal_op(N::AbstractOperator, d) = AbstractOperators.AffineAdd(N, d)
_tilt_normal_op(N::AbstractOperators.VCAT, d::ArrayPartition) =
    AbstractOperators.VCAT(map(AbstractOperators.AffineAdd, N.A, d.x)...)
