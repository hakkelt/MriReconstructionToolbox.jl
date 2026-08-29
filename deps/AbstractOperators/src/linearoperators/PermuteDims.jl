export PermuteDims

"""
	PermuteDims([domain_type=Float64::Type,] dim_in::Tuple, perm)
	PermuteDims(x::AbstractArray, perm)

Create a `LinearOperator` permuting the dimensions of its input, i.e. `L * x == permutedims(x, perm)`.

`perm` must be a permutation of `1:length(dim_in)`. The operator is orthogonal: its adjoint is the
permutation by `invperm(perm)`, and `L'L == I`, which is what lets it be used freely inside terms that
require a tight frame.

```jldoctest
julia> op = PermuteDims(Float64, (2, 3, 4), (3, 1, 2))
P  ℝ^(2, 3, 4) -> ℝ^(4, 2, 3)

julia> x = reshape(collect(1.0:24.0), 2, 3, 4);

julia> op * x == permutedims(x, (3, 1, 2))
true

julia> op' * (op * x) == x
true

```
"""
struct PermuteDims{T, N, S <: AbstractArray{T}} <: LinearOperator
    dim_in::NTuple{N, Int}
    perm::NTuple{N, Int}
    iperm::NTuple{N, Int}
end

# Constructors

function PermuteDims(
        domain_type::Type{T}, dim_in::NTuple{N, Int}, perm;
        array_type::Type = Array{T}
    ) where {T, N}
    permutation = NTuple{N, Int}(perm)
    isperm(permutation) || throw(
        ArgumentError("perm must be a permutation of 1:$N, got $permutation")
    )
    S = _normalize_array_type(array_type, domain_type)
    return PermuteDims{domain_type, N, S}(dim_in, permutation, NTuple{N, Int}(invperm(permutation)))
end

function PermuteDims(dim_in::NTuple{N, Int}, perm; array_type::Type = Array{Float64}) where {N}
    return PermuteDims(Float64, dim_in, perm; array_type)
end

function PermuteDims(x::AbstractArray, perm)
    return PermuteDims(
        eltype(x), size(x), perm; array_type = _array_wrapper(x){eltype(x)}
    )
end

# Mappings

function LinearAlgebra.mul!(y::AbstractArray, L::PermuteDims, b::AbstractArray)
    check(y, L, b)
    permutedims!(y, b, L.perm)
    return y
end

function LinearAlgebra.mul!(y::AbstractArray, A::AdjointOperator{<:PermuteDims}, b::AbstractArray)
    check(y, A, b)
    # A permutation matrix is orthogonal, so the adjoint is the inverse permutation.
    permutedims!(y, b, A.A.iperm)
    return y
end

# Properties

domain_type(::PermuteDims{T}) where {T} = T
codomain_type(::PermuteDims{T}) where {T} = T
domain_array_type(::PermuteDims{T, N, S}) where {T, N, S} = S
codomain_array_type(::PermuteDims{T, N, S}) where {T, N, S} = S
is_thread_safe(::PermuteDims) = true

is_full_row_rank(::PermuteDims) = true
is_full_column_rank(::PermuteDims) = true
is_orthogonal(::PermuteDims) = true
is_invertible(::PermuteDims) = true
diag_AcA(::PermuteDims) = 1.0
diag_AAc(::PermuteDims) = 1.0

function _copy_operator_impl(op::PermuteDims{T, N, S}; storage_type = nothing, threaded = nothing) where {T, N, S}
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return PermuteDims(T, op.dim_in, op.perm; array_type = new_at)
end

size(L::PermuteDims) = (ntuple(i -> L.dim_in[L.perm[i]], length(L.perm)), L.dim_in)

fun_name(::PermuteDims) = "P"
