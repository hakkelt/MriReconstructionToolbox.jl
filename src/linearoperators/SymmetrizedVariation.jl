export SymmetrizedVariation

"""
	SymmetrizedVariation([domain_type=Float64::Type,] dim_in::Tuple)
	SymmetrizedVariation(dims...)
	SymmetrizedVariation(x::AbstractArray)

Create a `LinearOperator` computing the symmetrized gradient (also called the symmetrized derivative, or the
linearized strain tensor) of a vector field.

The input is a vector field `w` on an `N`-dimensional grid `dim_in`, laid out exactly as [`Variation`](@ref)
lays out its output: a `(prod(dim_in), N)` matrix whose `j`-th column is the `j`-th component of the field.
The output is the symmetric tensor field

```math
ℰ(w)_{ij} = \\tfrac{1}{2}\\left(∂_i w_j + ∂_j w_i\\right),
```

stored as a `(prod(dim_in), N(N+1)/2)` matrix holding only the upper triangle: first the `N` diagonal
entries `ℰ(w)_{ii}` in order, then the off-diagonal entries `(i, j)` with `i < j` in lexicographic order.
The off-diagonal entries are scaled by `√2`, so that the Euclidean norm of a row equals the Frobenius norm
of the symmetric tensor it represents — the norm that appears in total generalized variation.

The derivatives are the same forward differences with mirrored boundary conditions that [`Variation`](@ref)
uses, so `SymmetrizedVariation` and `Variation` are consistent discretizations.

```jldoctest
julia> op = SymmetrizedVariation(Float64, (4, 4))
Ɛ  ℝ^(16, 2) -> ℝ^(16, 3)

julia> size(op)
((16, 3), (16, 2))

julia> all(iszero, op * ones(16, 2))
true

```
"""
struct SymmetrizedVariation{T, N, Th, S <: AbstractArray{T}, V <: Variation} <: LinearOperator
    dim_in::NTuple{N, Int}
    # The component-wise gradient is delegated to `Variation`, which keeps the two operators consistent by
    # construction: any change to the boundary convention there applies here without a second edit.
    grad::V
end

# Constructors

function SymmetrizedVariation(
        domain_type::Type{T}, dim_in::NTuple{N, Int};
        threaded::Bool = true, array_type::Type = Array{T}
    ) where {T, N}
    N == 1 && error("use FiniteDiff instead!")
    any(<(2), dim_in) && throw(
        ArgumentError(
            "SymmetrizedVariation requires every dimension to be at least 2, got $dim_in; " *
                "drop the singleton dimension(s) first"
        )
    )
    grad = Variation(domain_type, dim_in; threaded, array_type)
    S = _normalize_array_type(array_type, domain_type)
    return SymmetrizedVariation{domain_type, N, is_threaded(grad), S, typeof(grad)}(dim_in, grad)
end

function SymmetrizedVariation(
        domain_type::Type{T}, dim_in::Vararg{Int}; threaded::Bool = true, array_type::Type = Array{T}
    ) where {T}
    return SymmetrizedVariation(domain_type, dim_in; threaded, array_type)
end
function SymmetrizedVariation(
        dim_in::NTuple{N, Int}; threaded::Bool = true, array_type::Type = Array{Float64}
    ) where {N}
    return SymmetrizedVariation(Float64, dim_in; threaded, array_type)
end
function SymmetrizedVariation(dim_in::Vararg{Int}; threaded::Bool = true, array_type::Type = Array{Float64})
    return SymmetrizedVariation(dim_in; threaded, array_type)
end
function SymmetrizedVariation(x::AbstractArray; threaded::Bool = true)
    return SymmetrizedVariation(
        eltype(x), size(x); threaded, array_type = _array_wrapper(x){eltype(x)}
    )
end

"""
	_symmetrized_channels(N)

Return the tuple of `(i, j)` index pairs of the stored upper triangle, in the order they occupy the columns
of the codomain: the `N` diagonal pairs first, then the off-diagonal ones lexicographically.
"""
function _symmetrized_channels(N::Int)
    diagonal = [(i, i) for i in 1:N]
    off_diagonal = [(i, j) for i in 1:(N - 1) for j in (i + 1):N]
    return vcat(diagonal, off_diagonal)
end

_symmetrized_codomain_size(N::Int) = N * (N + 1) ÷ 2

# Mappings

function LinearAlgebra.mul!(
        y::AbstractArray, A::SymmetrizedVariation{T, N}, w::AbstractArray
    ) where {T, N}
    check(y, A, w)
    R = real(T)
    M = prod(A.dim_in)
    # G[:, d, j] is ∂_d of the j-th component of the field. Allocated here rather than cached in the struct
    # so that one operator can be used concurrently from several threads (`is_thread_safe`).
    G = similar(w, (M, N, N))
    for j in 1:N
        @views mul!(G[:, :, j], A.grad, reshape(w[:, j], A.dim_in))
    end
    for (k, (i, j)) in enumerate(_symmetrized_channels(N))
        if i == j
            @views y[:, k] .= G[:, i, i]
        else
            # ½(∂ᵢwⱼ + ∂ⱼwᵢ), times the √2 that makes the row norm the Frobenius norm of the tensor.
            @views y[:, k] .= (sqrt(R(2)) / 2) .* (G[:, i, j] .+ G[:, j, i])
        end
    end
    return y
end

function LinearAlgebra.mul!(
        w::AbstractArray, A::AdjointOperator{<:SymmetrizedVariation{T, N}}, y::AbstractArray
    ) where {T, N}
    L = A.A
    check(w, A, y)
    R = real(T)
    M = prod(L.dim_in)
    # C[:, d, j] is the coefficient multiplying ∂_d w_j in ⟨ℰw, y⟩, so that ℰᵀy = ∑_d ∂_dᵀ C[:, d, ·].
    C = similar(y, (M, N, N))
    for (k, (i, j)) in enumerate(_symmetrized_channels(N))
        if i == j
            @views C[:, i, i] .= y[:, k]
        else
            @views C[:, i, j] .= (sqrt(R(2)) / 2) .* y[:, k]
            @views C[:, j, i] .= C[:, i, j]
        end
    end
    for j in 1:N
        @views mul!(reshape(w[:, j], L.dim_in), L.grad', C[:, :, j])
    end
    return w
end

# Properties

domain_type(::SymmetrizedVariation{T}) where {T} = T
codomain_type(::SymmetrizedVariation{T}) where {T} = T
domain_array_type(::SymmetrizedVariation{T, N, Th, S}) where {T, N, Th, S} = S
codomain_array_type(::SymmetrizedVariation{T, N, Th, S}) where {T, N, Th, S} = S
is_thread_safe(::SymmetrizedVariation) = true
is_threaded(::SymmetrizedVariation{T, N, Th}) where {T, N, Th} = Th
supports_threading(::SymmetrizedVariation) = true

# The work is done by the inner `Variation`, so the crossover is the same one measured for it.
threading_threshold(::Type{<:SymmetrizedVariation}) = threading_threshold(Variation)

function _copy_operator_impl(
        op::SymmetrizedVariation{T, N, Th, S}; storage_type = nothing, threaded = nothing
    ) where {T, N, Th, S}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return SymmetrizedVariation(T, op.dim_in; threaded = new_threaded, array_type = new_at)
end

function size(L::SymmetrizedVariation{T, N}) where {T, N}
    M = prod(L.dim_in)
    return (M, _symmetrized_codomain_size(N)), (M, N)
end

fun_name(::SymmetrizedVariation) = "Ɛ"
