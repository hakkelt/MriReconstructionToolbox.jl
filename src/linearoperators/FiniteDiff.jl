export FiniteDiff

#TODO add boundary conditions

"""
	FiniteDiff([domain_type=Float64::Type,] dim_in::Tuple, direction = 1)
	FiniteDiff(x::AbstractArray, direction = 1)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the discretized gradient over the specified `direction` obtained using forward finite differences.

```jldoctest
julia> FiniteDiff(Float64,(3,))
δx  ℝ^3 -> ℝ^2

julia> FiniteDiff((3,4),2)
δy  ℝ^(3, 4) -> ℝ^(3, 3)

julia> all(FiniteDiff(ones(2,2,2,3),1)*ones(2,2,2,3) .== 0)
true
	
```
"""
struct FiniteDiff{N, D, T, S <: AbstractArray{T}, Th} <: LinearOperator
    dim_in::NTuple{N, Int}
    function FiniteDiff{N, D, T, S, Th}(dim_in) where {N, D, T, S <: AbstractArray{T}, Th}
        D > N && error("direction is bigger the number of dimension $N")
        Th isa Bool || throw(ArgumentError("FiniteDiff threading parameter must be a Bool"))
        return new{N, D, T, S, Th}(dim_in)
    end
end

# Thin alias for the shared resolver, kept because the constructors below read better with
# a short name. `threaded = false` vetoes; anything else defers to the policy.
function _finitediff_threaded(threaded::Bool, ::Type{T}, dim_in, ::Type{S}) where {T, S <: AbstractArray}
    return _elementwise_threaded(FiniteDiff, threaded, T, dim_in, S)
end

# Constructors
# Val-dispatch constructor — fully type-stable (D is known at compile time)
function FiniteDiff(
        ::Type{T}, dim_in::NTuple{N, Int}, ::Val{D};
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N, D}
    S = _normalize_array_type(array_type, T)
    return _finitediff_threaded(threaded, T, dim_in, S) ?
        FiniteDiff{N, D, T, S, true}(dim_in) :
        FiniteDiff{N, D, T, S, false}(dim_in)
end

# Specialized no-direction constructor: D=1 is a compile-time literal — fully type-stable
function FiniteDiff(
        dim_in::NTuple{N, Int}; array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N}
    S = _normalize_array_type(array_type, Float64)
    return _finitediff_threaded(threaded, Float64, dim_in, S) ?
        FiniteDiff{N, 1, Float64, S, true}(dim_in) :
        FiniteDiff{N, 1, Float64, S, false}(dim_in)
end

# Specialized no-direction constructor: D=1 is a compile-time literal, so this stays fully
# type-stable. Without it the two-argument call would fall through to the `dir::Int` method
# below and pay a runtime dispatch on `Val(dir)` — which JET's `@test_opt` flags.
function FiniteDiff(
        domain_type::Type{T}, dim_in::NTuple{N, Int};
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return _finitediff_threaded(threaded, T, dim_in, S) ?
        FiniteDiff{N, 1, T, S, true}(dim_in) :
        FiniteDiff{N, 1, T, S, false}(dim_in)
end

# Direction as a runtime Int — necessarily delegates through `Val`, so this path is
# dynamically dispatched by construction. Call the `Val{D}` method directly from
# performance-sensitive code.
function FiniteDiff(
        domain_type::Type{T}, dim_in::NTuple{N, Int}, dir::Int;
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N}
    return FiniteDiff(domain_type, dim_in, Val(dir); array_type, threaded)
end

function FiniteDiff(
        dim_in::NTuple{N, Int}, dir::Int; array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N}
    return FiniteDiff(Float64, dim_in, Val(dir); array_type, threaded)
end

function FiniteDiff(x::AbstractArray{T, N}, dir::Int = 1; threaded::Bool = true) where {T, N}
    S = _normalize_array_type(_array_wrapper(x), T)
    return FiniteDiff{N, dir, T, S, _finitediff_threaded(threaded, T, size(x), S)}(size(x))
end

# Mappings

# An array of size `dim_in` is a `(pre * n, post)` matrix, with `n = dim_in[D]`, `pre` the
# product of the dimensions before `D` and `post` of those after it. A step of one along `D`
# is a shift of `pre` rows, so the difference of each column is the difference of two
# contiguous row ranges, and the innermost loop runs over `pre * (n - 1)` elements. Indexing
# the N-dimensional array along `D` instead runs the innermost loop over `dim_in[1]` alone,
# which is short when the leading dimension is (a 2- or 3-vector of coordinates, say).
@inline function _finitediff_slabs(dim_in::NTuple{N, Int}, ::Val{D}) where {N, D}
    pre = post = 1
    for i in 1:(D - 1)
        pre *= dim_in[i]
    end
    for i in (D + 1):N
        post *= dim_in[i]
    end
    return pre, dim_in[D], post
end

# `reshape` of an `Array` allocates a new array header on every call; a `ReshapedArray` over
# an `IndexLinear` parent needs no index arithmetic beyond the column offset and allocates
# nothing. Other arrays, device arrays among them, keep their own `reshape`.
_slab_view(a::Array, dims::Dims) = Base.ReshapedArray(a, dims, ())
_slab_view(a::AbstractArray, dims::Dims) = reshape(a, dims)

# `@views` keeps the whole forward difference allocation-free -- plain indexing would
# materialise a temporary for each side of the subtraction -- which is also what makes it
# worth threading. The index ranges are computed before the broadcast: inside it `@.` would
# dot their arithmetic too, and spelled as `end` and `:` inference widens them to a union of
# index types.
function mul!(y::AbstractArray, L::FiniteDiff{N, D, T, S, false}, b::AbstractArray) where {N, D, T, S}
    check(y, L, b)
    pre, n, post = _finitediff_slabs(L.dim_in, Val(D))
    m = pre * (n - 1)
    ahead, behind, cols = (pre + 1):(pre + m), 1:m, 1:post
    B, Y = _slab_view(b, (pre * n, post)), _slab_view(y, (m, post))
    @views @. Y = B[ahead, cols] - B[behind, cols]
    return y
end

# A threaded broadcast splits its last axis, so with a single column (`D` the last dimension)
# the difference is written over vectors instead.
function mul!(y::AbstractArray, L::FiniteDiff{N, D, T, S, true}, b::AbstractArray) where {N, D, T, S}
    check(y, L, b)
    pre, n, post = _finitediff_slabs(L.dim_in, Val(D))
    m = pre * (n - 1)
    ahead, behind, cols = (pre + 1):(pre + m), 1:m, 1:post
    if post == 1
        Bv, Yv = _slab_view(b, (pre * n,)), _slab_view(y, (m,))
        @views @.. thread = true Yv = Bv[ahead] - Bv[behind]
    else
        B, Y = _slab_view(b, (pre * n, post)), _slab_view(y, (m, post))
        @views @.. thread = true Y = B[ahead, cols] - B[behind, cols]
    end
    return y
end

function mul!(
        y::AbstractArray, L::AdjointOperator{<:FiniteDiff{N, D, T, S, Th}}, b::AbstractArray
    ) where {N, D, T, S, Th}
    check(y, L, b)
    pre, n, post = _finitediff_slabs(L.A.dim_in, Val(D))
    m = pre * (n - 1)
    # Rows of `y` along `D`: the first, the middle ones, the last; and the rows of `b` they read.
    top, middle, bottom = 1:pre, (pre + 1):m, (m + 1):(m + pre)
    middle_behind, bottom_in, cols = 1:(m - pre), (m - pre + 1):m, 1:post
    Y, B = _slab_view(y, (pre * n, post)), _slab_view(b, (m, post))
    # Same story as the forward pass: `@views` removes the temporaries, and the middle
    # block -- the only one whose size grows with `dim_in` -- is the part worth threading.
    @views @. Y[top, cols] = -B[top, cols]
    # With two samples along `D` there are no middle rows, and a threaded broadcast would
    # still split the empty block's columns across the threads.
    if n > 2
        if Th && post == 1
            Yv, Bv = _slab_view(y, (pre * n,)), _slab_view(b, (m,))
            @views @.. thread = true Yv[middle] = Bv[middle_behind] - Bv[middle]
        elseif Th
            @views @.. thread = true Y[middle, cols] = B[middle_behind, cols] - B[middle, cols]
        else
            @views @. Y[middle, cols] = B[middle_behind, cols] - B[middle, cols]
        end
    end
    @views @. Y[bottom, cols] = B[bottom_in, cols]
    return y
end

# The same slabs as the 3-argument kernels, each written as `α * difference + β * y`.
function mul!(
        y::AbstractArray, L::FiniteDiff{N, D, T, S, Th}, b::AbstractArray, α::Number, β::Number
    ) where {N, D, T, S, Th}
    check(y, L, b)
    pre, n, post = _finitediff_slabs(L.dim_in, Val(D))
    m = pre * (n - 1)
    ahead, behind, cols = (pre + 1):(pre + m), 1:m, 1:post
    if Th && post == 1
        Bv, Yv = _slab_view(b, (pre * n,)), _slab_view(y, (m,))
        _store!(Yv, Broadcast.broadcasted(-, view(Bv, ahead), view(Bv, behind)), α, β, Val(true))
    else
        B, Y = _slab_view(b, (pre * n, post)), _slab_view(y, (m, post))
        _store!(Y, Broadcast.broadcasted(-, view(B, ahead, cols), view(B, behind, cols)), α, β, Val(Th))
    end
    return y
end

function mul!(
        y::AbstractArray, L::AdjointOperator{<:FiniteDiff{N, D, T, S, Th}}, b::AbstractArray, α::Number, β::Number
    ) where {N, D, T, S, Th}
    check(y, L, b)
    pre, n, post = _finitediff_slabs(L.A.dim_in, Val(D))
    m = pre * (n - 1)
    top, middle, bottom = 1:pre, (pre + 1):m, (m + 1):(m + pre)
    middle_behind, bottom_in, cols = 1:(m - pre), (m - pre + 1):m, 1:post
    Y, B = _slab_view(y, (pre * n, post)), _slab_view(b, (m, post))
    _store!(view(Y, top, cols), Broadcast.broadcasted(-, view(B, top, cols)), α, β)
    if n > 2
        if Th && post == 1
            Yv, Bv = _slab_view(y, (pre * n,)), _slab_view(b, (m,))
            rhs = Broadcast.broadcasted(-, view(Bv, middle_behind), view(Bv, middle))
            _store!(view(Yv, middle), rhs, α, β, Val(true))
        else
            rhs = Broadcast.broadcasted(-, view(B, middle_behind, cols), view(B, middle, cols))
            _store!(view(Y, middle, cols), rhs, α, β, Val(Th))
        end
    end
    _store!(view(Y, bottom, cols), view(B, bottom_in, cols), α, β)
    return y
end

# Properties

domain_type(::FiniteDiff{<:Any, <:Any, T}) where {T} = T
codomain_type(::FiniteDiff{<:Any, <:Any, T}) where {T} = T
domain_array_type(::FiniteDiff{N, D, T, S}) where {N, D, T, S} = S
codomain_array_type(::FiniteDiff{N, D, T, S}) where {N, D, T, S} = S
is_thread_safe(::FiniteDiff) = true
is_threaded(::FiniteDiff{N, D, T, S, Th}) where {N, D, T, S, Th} = Th

# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl.
# Crossover of this operator's real `mul!`: Float64 2^15, Float32 2^16.
threading_threshold(::Type{<:FiniteDiff}) = 2^16

function _copy_operator_impl(
        op::FiniteDiff{N, D, T, S, Th}; storage_type = nothing, threaded = nothing
    ) where {N, D, T, S, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return FiniteDiff(T, op.dim_in, Val(D); array_type = new_at, threaded = new_threaded)
end

function size(L::FiniteDiff{N, D}) where {N, D}
    dim_out = ntuple(i -> i == D ? L.dim_in[i] - 1 : L.dim_in[i], Val(N))
    return dim_out, L.dim_in
end

fun_name(::FiniteDiff{<:Any, 1}) = "δx"
fun_name(::FiniteDiff{<:Any, 2}) = "δy"
fun_name(::FiniteDiff{<:Any, 3}) = "δz"
fun_name(::FiniteDiff{<:Any, D}) where {D} = "δx$D"

is_full_row_rank(::FiniteDiff) = true
supports_threading(::FiniteDiff) = true
