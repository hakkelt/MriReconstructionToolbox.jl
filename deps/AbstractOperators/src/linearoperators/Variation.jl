export Variation

"""
Variation([domain_type=Float64::Type,] dim_in::Tuple)
Variation(dims...)
Variation(x::AbstractArray)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns a matrix with its `i`th column consisting of the vectorized discretized gradient over the `i`th `direction obtained using forward finite differences.

```jldoctest
julia> Variation(Float64,(10,2))
Ʋ  ℝ^(10, 2) -> ℝ^(20, 2)

julia> Variation(2,2,2)
Ʋ  ℝ^(2, 2, 2) -> ℝ^(8, 3)

julia> Variation(ones(2,2))*[1. 2.; 1. 2.]
4×2 Matrix{Float64}:
 0.0  1.0
 0.0  1.0
 0.0  1.0
 0.0  1.0

```
"""
struct Variation{T, N, Th, S <: AbstractArray{T}} <: LinearOperator
    dim_in::NTuple{N, Int}
end

# Constructors
#default constructor
function Variation(
        domain_type::Type{T}, dim_in::NTuple{N, Int};
        threaded::Bool = true, array_type::Type = Array{T}
    ) where {T, N}
    N == 1 && error("use FiniteDiff instead!")
    # A singleton dimension has no finite difference to take: the forward kernel's
    # `next_slicing` runs off the end of `b` (a `BoundsError` today, so nothing can be
    # relying on it), and the adjoint would have to invent a value. Rejected at construction
    # so the failure names the cause instead of surfacing as an index error inside `mul!`.
    any(<(2), dim_in) && throw(
        ArgumentError(
            "Variation requires every dimension to be at least 2, got $dim_in; " *
                "drop the singleton dimension(s) first"
        )
    )
    S = _normalize_array_type(array_type, domain_type)
    th = _elementwise_threaded(Variation, threaded, domain_type, dim_in, S)
    return Variation{domain_type, N, th, S}(dim_in)
end

function Variation(
        domain_type::Type{T}, dim_in::Vararg{Int}; threaded::Bool = true, array_type::Type = Array{T}
    ) where {T}
    return Variation(domain_type, dim_in; threaded, array_type)
end
function Variation(
        dim_in::NTuple{N, Int}; threaded::Bool = true, array_type::Type = Array{Float64}
    ) where {N}
    return Variation(Float64, dim_in; threaded, array_type)
end
function Variation(dim_in::Vararg{Int}; threaded::Bool = true, array_type::Type = Array{Float64})
    return Variation(dim_in; threaded, array_type)
end
function Variation(x::AbstractArray; threaded::Bool = true)
    # Delegates to the dimension-tuple constructor rather than building the struct directly,
    # so the threading policy and the dimension validation are stated in exactly one place --
    # otherwise `Variation(zeros(100,100))` and `Variation(Float64,(100,100))` could end up
    # thresholding threading differently (element count vs. byte size) for identical inputs.
    return Variation(eltype(x), size(x); threaded, array_type = _array_wrapper(x){eltype(x)})
end

# Mappings
# Non-threaded forward
@inbounds function LinearAlgebra.mul!(
        y::AbstractArray, A::Variation{T, N, false, <:Any}, b::AbstractArray
    ) where {T, N}
    check(y, A, b)
    @assert firstindex(b) == 1 "Only support 1-based arrays"
    @assert firstindex(y) == 1 "Only support 1-based arrays"

    # First dimension -- special case
    batch_length = size(b, 1)
    @.. y[2:end, 1] = b[2:end] - b[1:(end - 1)] # finite difference along the first dimension, but incorrect for boundaries
    @.. y[1:batch_length:end, 1] = b[2:batch_length:end] - b[1:batch_length:end] # correct boundaries with mirrored boundary conditions

    # Other dimensions
    batch_count = length(b) ÷ batch_length
    for d in 2:N
        for k in 0:(batch_count - 1)
            slice_start = k * batch_length + 1
            slice_end = (k + 1) * batch_length
            slicing = slice_start:slice_end
            if k % size(b, d) == 0
                next_slice_start = (k + 1) * batch_length + 1
                next_slice_end = (k + 2) * batch_length
                next_slicing = next_slice_start:next_slice_end
                @views y[slicing, d] .= b[next_slicing] .- b[slicing]
            else
                prev_slice_start = (k - 1) * batch_length + 1
                prev_slice_end = k * batch_length
                prev_slicing = prev_slice_start:prev_slice_end
                @views y[slicing, d] .= b[slicing] .- b[prev_slicing]
            end
        end
        batch_count ÷= size(b, d)
        batch_length *= size(b, d)
    end
    return y
end

@inbounds function LinearAlgebra.mul!(
        y::AbstractArray, A::Variation{T, N, true, <:Any}, b::AbstractArray
    ) where {T, N}
    check(y, A, b)
    @assert firstindex(b) == 1 "Only support 1-based arrays"
    @assert firstindex(y) == 1 "Only support 1-based arrays"

    # First dimension -- special case
    batch_length = size(b, 1)
    @.. thread = true y[2:end, 1] = b[2:end] - b[1:(end - 1)] # finite difference along the first dimension, but incorrect for boundaries
    @.. thread = true y[1:batch_length:end, 1] = b[2:batch_length:end] - b[1:batch_length:end] # correct boundaries with mirrored boundary conditions

    # Other dimensions
    batch_count = length(b) ÷ batch_length
    for d in 2:N
        @batch for k in 0:(batch_count - 1)
            slice_start = k * batch_length + 1
            slice_end = (k + 1) * batch_length
            slicing = slice_start:slice_end
            if k % size(b, d) == 0
                next_slice_start = (k + 1) * batch_length + 1
                next_slice_end = (k + 2) * batch_length
                next_slicing = next_slice_start:next_slice_end
                @views y[slicing, d] .= b[next_slicing] .- b[slicing]
            else
                prev_slice_start = (k - 1) * batch_length + 1
                prev_slice_end = k * batch_length
                prev_slicing = prev_slice_start:prev_slice_end
                @views y[slicing, d] .= b[slicing] .- b[prev_slicing]
            end
        end
        batch_count ÷= size(b, d)
        batch_length *= size(b, d)
    end
    return y
end

"""
    _variation_adjoint_term(b, cnt, d, i, n, stride)

Contribution of dimension `d` to the adjoint at linear index `cnt`, where `i` is the index
of `cnt` along that dimension and `n = size(y, d)`.

The forward difference along a dimension of length `n` is `(Dx)_1 = x_2 - x_1` (the mirrored
boundary) and `(Dx)_i = x_i - x_{i-1}` for `i >= 2`, so column `j` of the transpose collects
three separable contributions:

  * from row `i = j`: `-b_1` when `j == 1`, otherwise `+b_j`;
  * from row `i = 1`: `+b_1`, but only when `j == 2` (the mirrored boundary's second entry);
  * from row `i = j + 1`: `-b_{j+1}`, whenever such a row exists, i.e. `j < n`.

Written as three independent terms rather than a branch chain over `i` on purpose. The chain
tested `i == 2` *before* `i == n`, so a dimension of length exactly 2 took the interior
formula for its last column and read `b[cnt + stride]` one element past the end -- a
`BoundsError` for `Variation(Float64, (8, 2))'` and friends, and it applied to *any*
dimension of size 2, not only the trailing one. Here `i == 2` and `i == n` are separate
`+=`/`-=` steps, so the `n == 2` case correctly takes both.
"""
@inline function _variation_adjoint_term(
        b::AbstractArray, cnt::Int, d::Int, i::Int, n::Int, stride::Int
    )
    v = i == 1 ? -b[cnt, d] : b[cnt, d]
    i == 2 && (v += b[cnt - stride, d])
    i < n && (v -= b[cnt + stride, d])
    return v
end

# Body of the adjoint for a single linear index, shared by the threaded and non-threaded
# methods below so the two cannot drift apart.
#
# `N` is a plain `Int`, deliberately not a `Val`. Inside Polyester's `@batch` the loop body
# becomes a closure, which does not carry the enclosing method's static parameters, so a
# `Val(N)` argument is built from a runtime value: every call then dispatches dynamically and
# allocates. Measured at n = 2^22 that cost 293 MB and turned a 4.5x speedup into 0.25x.
@inline function _variation_adjoint_at!(
        y::AbstractArray, b::AbstractArray, cnt::Int, N::Int
    )
    acc = _variation_adjoint_term(b, cnt, 1, (cnt - 1) % size(y, 1) + 1, size(y, 1), 1)
    stride = size(y, 1)
    for d in 2:N
        i_d = ((cnt - 1) ÷ stride) % size(y, d) + 1
        acc += _variation_adjoint_term(b, cnt, d, i_d, size(y, d), stride)
        stride *= size(y, d)
    end
    y[cnt] = acc
    return nothing
end

# Adjoint, in the forward's own idiom: flat, strided slab passes over `y` and column `d` of
# `b`, no scalar-per-element indexing arithmetic. `y` accumulates the three separable
# contributions from `_variation_adjoint_term`'s docstring across all `N` dimensions, so it is
# zeroed first:
#   - interior, `j = 2:n`         -> `+b_j`               (row `i = j`)
#   - boundary, `j = 1`           -> `-b_1`                (row `i = j`, `j == 1`)
#   - `j = 1:n-1`                 -> `-b_{j+1}`            (row `i = j+1`, whenever `j < n`)
#   - `j = 2`                     -> `+b_1`                (row `i = 1`, mirrored boundary)
#
# Dimension 1 mirrors the forward's own trick: a single whole-array shifted pass gets the
# interior terms (1 and 2) right everywhere except at each block's own start/end, which are
# then corrected in place with two additional strided passes (`_variation_adjoint_dim1!`).
# Reshaping a `view` to recover per-dimension slabs directly (the more obvious rewrite) boxes
# on this codebase's Julia/FastBroadcast combination, so this stays index arithmetic on the
# flat arrays instead -- verified zero-allocating by the "adjoint allocates nothing" test.
# Dimensions `2:N` mirror the forward's `k`-loop over `batch_length`-sized slabs directly
# (`_variation_adjoint_dim!`), the boundary slab (`k % size(y, d) == 0`) folding in both the
# `j == 1` and `j == 2` terms, exactly as the forward's boundary branch folds in its own two
# slices.
@inline function _variation_adjoint_dim1!(y::AbstractArray, bcol, n::Int, ::Val{false})
    len = length(y)
    @inbounds @simd for i in 2:len
        y[i] += bcol[i]
    end
    @inbounds @simd for i in 1:(len - 1)
        y[i] -= bcol[i + 1]
    end
    @inbounds @simd for i in 1:n:len
        y[i] -= bcol[i]
        y[i + 1] += bcol[i]
    end
    if len > n
        @inbounds @simd for i in (n + 1):n:len
            y[i] -= bcol[i]
            y[i - 1] += bcol[i]
        end
    end
    return
end
@inline function _variation_adjoint_dim1!(y::AbstractArray, bcol, n::Int, ::Val{true})
    len = length(y)
    @batch for i in 2:len
        @inbounds y[i] += bcol[i]
    end
    @batch for i in 1:(len - 1)
        @inbounds y[i] -= bcol[i + 1]
    end
    # The remaining two ranges have only `len ÷ n` elements each -- one per block, not one per
    # array element -- so they are never worth Polyester's task-spawn cost; run them serially
    # regardless of `threaded`, exactly as `S4` skips a no-op threading scope elsewhere.
    @inbounds @simd for i in 1:n:len
        y[i] -= bcol[i]
        y[i + 1] += bcol[i]
    end
    if len > n
        @inbounds @simd for i in (n + 1):n:len
            y[i] -= bcol[i]
            y[i - 1] += bcol[i]
        end
    end
    return
end

@inbounds function _variation_adjoint!(y::AbstractArray{T, N}, b::AbstractArray, thread::Val) where {T, N}
    fill!(y, zero(T))

    # Dimension 1
    n1 = size(y, 1)
    _variation_adjoint_dim1!(y, view(b, :, 1), n1, thread)

    # Dimensions 2:N, in the forward's own `batch_length`/`k` slab layout.
    batch_length = n1
    batch_count = length(y) ÷ batch_length
    for d in 2:N
        bcol = view(b, :, d)
        n = size(y, d)
        _variation_adjoint_dim!(y, bcol, batch_length, batch_count, n, thread)
        batch_count ÷= n
        batch_length *= n
    end
    return y
end

@inline function _variation_adjoint_dim!(y, bcol, batch_length::Int, batch_count::Int, n::Int, ::Val{false})
    for k in 0:(batch_count - 1)
        _variation_adjoint_slab!(y, bcol, batch_length, k, n)
    end
    return
end
@inline function _variation_adjoint_dim!(y, bcol, batch_length::Int, batch_count::Int, n::Int, ::Val{true})
    @batch for k in 0:(batch_count - 1)
        _variation_adjoint_slab!(y, bcol, batch_length, k, n)
    end
    return
end

# One slab of dimension `d`, writing **only its own** `batch_length` elements of `y` and
# gathering the three terms of `_variation_adjoint_term` from `bcol` instead.
#
# It used to scatter: each slab added its own contribution and then reached into the
# neighbouring slab (`slice_start ± batch_length`) to deposit the term that slab owed. That is
# correct serially, but `_variation_adjoint_dim!`'s threaded method runs the `k` loop under
# `@batch`, so two threads holding adjacent `k` did concurrent read-modify-write on the same
# elements and lost updates. Measured before this change, threaded vs serial on the same input:
# different in 500/500 runs, with relative errors up to 2.3e-1 on `(100, 50)` and 1.8e-1 on
# `(64, 64, 16)` — wrong results, not merely non-reproducible ones.
#
# Gathering keeps the write set of every `k` disjoint, which is what makes the `@batch` legal.
# `j` is the 0-based index along dimension `d`, so the terms are: `-b_1` at `j == 0` and `+b_j`
# otherwise (this slab); `+b_1` at `j == 1` (the mirrored boundary); and `-b_{j+1}` whenever a
# next slab exists. The per-slab branches are loop-invariant and hoist out of the `@simd` runs.
@inline function _variation_adjoint_slab!(y, bcol, batch_length::Int, k::Int, n::Int)
    slice_start = k * batch_length + 1
    slice_end = (k + 1) * batch_length
    j = k % n
    if j == 0
        @inbounds @simd for i in slice_start:slice_end
            y[i] -= bcol[i]
        end
    else
        @inbounds @simd for i in slice_start:slice_end
            y[i] += bcol[i]
        end
    end
    if j == 1
        @inbounds @simd for i in slice_start:slice_end
            y[i] += bcol[i - batch_length]
        end
    end
    if j != n - 1
        @inbounds @simd for i in slice_start:slice_end
            y[i] -= bcol[i + batch_length]
        end
    end
    return
end

# Non-threaded adjoint
function LinearAlgebra.mul!(
        y::AbstractArray, A::AdjointOperator{<:Variation{T, N, false}}, b::AbstractArray
    ) where {T, N}
    check(y, A, b)
    return _variation_adjoint!(y, b, Val(false))
end

# Threaded adjoint
function LinearAlgebra.mul!(
        y::AbstractArray, A::AdjointOperator{<:Variation{T, N, true}}, b::AbstractArray
    ) where {T, N}
    check(y, A, b)
    return _variation_adjoint!(y, b, Val(true))
end

# Properties

domain_type(::Variation{T}) where {T} = T
codomain_type(::Variation{T}) where {T} = T
domain_array_type(::Variation{T, N, Th, S}) where {T, N, Th, S} = S
codomain_array_type(::Variation{T, N, Th, S}) where {T, N, Th, S} = S
is_thread_safe(::Variation) = true

function _copy_operator_impl(
        op::Variation{T, N, Th, S}; storage_type = nothing, threaded = nothing
    ) where {T, N, Th, S}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Variation(T, op.dim_in; threaded = new_threaded, array_type = new_at)
end

size(L::Variation{T, N}) where {T, N} = ((prod(L.dim_in), N), L.dim_in)

fun_name(L::Variation) = "Ʋ"

is_threaded(::Variation{T, N, Th, S}) where {T, N, Th, S} = Th
# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl.
# Crossover of this operator's real `mul!` (forward + adjoint): Float64 2^10, Float32 2^10.
# Lower than FiniteDiff despite both being "arithmetic" because Variation makes one strided
# pass per dimension, so each element carries several times more work.
threading_threshold(::Type{<:Variation}) = 2^10
supports_threading(::Variation) = true
