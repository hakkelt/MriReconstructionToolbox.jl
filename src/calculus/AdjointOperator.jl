export AdjointOperator

"""
	AdjointOperator(A::AbstractOperator)

Shorthand constructor:

	'(A::AbstractOperator)

Returns the adjoint operator of `A`.

```jldoctest
julia> AdjointOperator(ZeroPad((2,2),(0,2)))
[I;0]ᵃ  ℝ^(2, 4) -> ℝ^(2, 2)

julia> [Eye(10); FiniteDiff((10,))]'
[I;δx]ᵃ  ℝ^10  ℝ^9 -> ℝ^10
	
```
"""
struct AdjointOperator{T <: AbstractOperator} <: AbstractOperator
    A::T
    function AdjointOperator(A::T) where {T <: AbstractOperator}
        # The adjoint of an affine operator is the adjoint of its linear part.
        is_affine(A) == false &&
            error("Cannot transpose a nonlinear operator. You might use `jacobian`")
        return new{T}(A)
    end
end

_ndoms_from_type(::Type{<:AdjointOperator{T}}, dim::Int) where {T} = _ndoms_from_type(T, dim == 1 ? 2 : 1)

# Constructors

AdjointOperator(L::AdjointOperator) = L.A

# Properties

has_fast_opnorm(L::AdjointOperator) = has_fast_opnorm(L.A)
LinearAlgebra.opnorm(L::AdjointOperator) = opnorm(L.A)
opnorm_bound(L::AdjointOperator) = opnorm_bound(L.A)
# The keywords are forwarded, not swallowed: `‖A'‖ = ‖A‖` exactly, so whatever promise the inner
# call keeps about its own margin and side is the promise this call makes.
estimate_opnorm(L::AdjointOperator; kwargs...) = estimate_opnorm(L.A; kwargs...)

Base.:(==)(L1::AdjointOperator{T}, L2::AdjointOperator{T}) where {T} = L1.A == L2.A
size(L::AdjointOperator) = size(L.A, 2), size(L.A, 1)

domain_type(L::AdjointOperator) = codomain_type(L.A)
codomain_type(L::AdjointOperator) = domain_type(L.A)
domain_array_type(L::AdjointOperator) = codomain_array_type(L.A)
codomain_array_type(L::AdjointOperator) = domain_array_type(L.A)
is_thread_safe(L::AdjointOperator) = is_thread_safe(L.A)

fun_name(L::AdjointOperator) = fun_name(L.A) * "ᵃ"

is_linear(L::AdjointOperator) = is_affine(L.A)
is_null(L::AdjointOperator) = is_null(L.A)
is_eye(L::AdjointOperator) = is_eye(L.A)
is_diagonal(L::AdjointOperator) = is_diagonal(L.A)
is_AcA_diagonal(L::AdjointOperator) = is_AAc_diagonal(L.A)
is_AAc_diagonal(L::AdjointOperator) = is_AcA_diagonal(L.A)
is_orthogonal(L::AdjointOperator) = is_orthogonal(L.A)
is_invertible(L::AdjointOperator) = is_invertible(L.A)
is_full_row_rank(L::AdjointOperator) = is_full_column_rank(L.A)
is_full_column_rank(L::AdjointOperator) = is_full_row_rank(L.A)

diag(L::AdjointOperator) = diag(L.A)
diag_AcA(L::AdjointOperator) = diag_AAc(L.A)
diag_AAc(L::AdjointOperator) = diag_AcA(L.A)

_children(L::AdjointOperator) = (L.A,)
is_threaded(L::AdjointOperator) = _is_threaded_from_children(L)
supports_threading(L::AdjointOperator) = _supports_threading_from_children(L)

function _copy_operator_impl(L::AdjointOperator; storage_type = nothing, threaded = nothing)
    return AdjointOperator(copy_operator(L.A; storage_type, threaded))
end

"""
	add_mul!(y, L::AdjointOperator, b, buf)

`y .+= L * b`. The generic path writes `L * b` into `buf` and adds, which costs two full passes
over `y` per call; an operator that can accumulate into `y` directly specializes this method and
ignores `buf` (`GetIndex` does). Internal, not exported, but the specialization point is part of
the operator contract: `VCAT`'s adjoint sums its blocks' adjoints through it, so a stack of `N`
blocks that each touch a small, disjoint part of the domain — a per-frame `GetIndex` stack, say —
is quadratic in `N` without a specialization and linear with one. Measured on a `128×128×T`
domain with one `GetIndex` per frame, adjoint wall time went 0.81 → 0.13 ms at `T = 4` and
162.6 → 2.9 ms at `T = 64`.
"""
function add_mul!(y::AbstractArray, L::AdjointOperator, b, buf::AbstractArray)
    mul!(buf, L, b)
    y .+= buf
    return y
end

# `H.A[i]'` is not always an `AdjointOperator`: a self-adjoint operator short-circuits its own
# `AdjointOperator` constructor back to itself (`Eye`, `NormalGetIndex`) or unwraps a double
# adjoint, so `VCAT`'s adjoint loop can hand `add_mul!` a bare operator of any type. This is the
# same buffer-and-add body as the `AdjointOperator` method above, as a fallback for that case.
function add_mul!(y::AbstractArray, L::AbstractOperator, b, buf::AbstractArray)
    mul!(buf, L, b)
    y .+= buf
    return y
end
