export HigherOrderDiff

"""
	HigherOrderDiff([domain_type=Float64::Type,] dim_in::Tuple, direction, order)
	HigherOrderDiff(x::AbstractArray, direction, order)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns
the forward finite difference of order `order` along `direction`:

```math
y_i = \\sum_{j=0}^{K} (-1)^{K-j} \\binom{K}{j} x_{i+j}, \\qquad i = 1, \\dots, n - K,
```

with `K = order` and `n = size(x, direction)`. It equals `order` chained [`FiniteDiff`](@ref)s
along the same direction, computed in a single pass over the array and without the
intermediate arrays, and it is what multiplying such a chain builds: `FiniteDiff`s and
`HigherOrderDiff`s along one direction, with the same element and storage type, combine into
one, and so do their adjoints. Order 1 is `FiniteDiff` itself, bit for bit.

The single pass sums the stencil in another order than the chain does, so a combined chain
differs from the chain applied step by step by rounding: at most a few `K * eps` relative to
the input's largest entry.

```jldoctest
julia> HigherOrderDiff(Float64, (10,), 1, 2)
δx²  ℝ^10 -> ℝ^8

julia> HigherOrderDiff((3, 6), 2, 3)
δy³  ℝ^(3, 6) -> ℝ^(3, 3)

julia> FiniteDiff((10,)) * FiniteDiff((11,))
δx²  ℝ^11 -> ℝ^9

julia> HigherOrderDiff(Float64, (5,), 1, 2) * collect(1.0:5.0) .^ 2
3-element Vector{Float64}:
 2.0
 2.0
 2.0
```
"""
struct HigherOrderDiff{N, D, K, T, S <: AbstractArray{T}, Th} <: LinearOperator
    dim_in::NTuple{N, Int}
    function HigherOrderDiff{N, D, K, T, S, Th}(dim_in) where {N, D, K, T, S <: AbstractArray{T}, Th}
        D > N && error("direction is bigger the number of dimension $N")
        K isa Int && K >= 1 || throw(ArgumentError("HigherOrderDiff order must be a positive Int, got $K"))
        dim_in[D] > K ||
            throw(ArgumentError("HigherOrderDiff of order $K needs more than $K samples along direction $D, got $(dim_in[D])"))
        Th isa Bool || throw(ArgumentError("HigherOrderDiff threading parameter must be a Bool"))
        return new{N, D, K, T, S, Th}(dim_in)
    end
end

# Constructors
# Val-dispatch constructor -- fully type-stable (D and K are known at compile time)
function HigherOrderDiff(
        ::Type{T}, dim_in::NTuple{N, Int}, ::Val{D}, ::Val{K};
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N, D, K}
    S = _normalize_array_type(array_type, T)
    return _elementwise_threaded(HigherOrderDiff, threaded, T, dim_in, S) ?
        HigherOrderDiff{N, D, K, T, S, true}(dim_in) :
        HigherOrderDiff{N, D, K, T, S, false}(dim_in)
end

# Direction and order as runtime Ints -- necessarily delegates through `Val`. Call the `Val`
# method directly from performance-sensitive code.
function HigherOrderDiff(
        domain_type::Type{T}, dim_in::NTuple{N, Int}, dir::Int, order::Int;
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N}
    return HigherOrderDiff(domain_type, dim_in, Val(dir), Val(order); array_type, threaded)
end

function HigherOrderDiff(
        dim_in::NTuple{N, Int}, dir::Int, order::Int; array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N}
    return HigherOrderDiff(Float64, dim_in, Val(dir), Val(order); array_type, threaded)
end

function HigherOrderDiff(x::AbstractArray{T, N}, dir::Int, order::Int; threaded::Bool = true) where {T, N}
    return HigherOrderDiff(T, size(x), Val(dir), Val(order); array_type = _array_wrapper(x), threaded)
end

# Mappings

# Coefficient of tap `j` of the order-`K` forward difference.
_hod_coef(K, j) = (isodd(K - j) ? -1 : 1) * binomial(K, j)

# `dest = Σ_j c_j src[rows[j + 1], cols]` with the taps unrolled into one broadcast, `cols`
# being `nothing` for vectors. The sum runs from `j = K` down, so that order 1 reads as
# `src[rows[2]] - src[rows[1]]`, which is `FiniteDiff`'s own expression. The ranges are bound
# to locals before the broadcast, since `@views` and `@.` would rewrite them inside it.
@generated function _hod_stencil!(
        dest, src, rows::NTuple{M, UnitRange{Int}}, cols, ::Val{Th}
    ) where {M, Th}
    K = M - 1
    names = [Symbol(:r, j) for j in 0:K]
    tap(j) = cols === Nothing ? :(src[$(names[j + 1])]) : :(src[$(names[j + 1]), cols])
    term(j) = abs(_hod_coef(K, j)) == 1 ? tap(j) : :($(abs(_hod_coef(K, j))) * $(tap(j)))
    rhs = term(K)
    for j in (K - 1):-1:0
        rhs = _hod_coef(K, j) > 0 ? :($rhs + $(term(j))) : :($rhs - $(term(j)))
    end
    bind = [:($(names[j + 1]) = rows[$(j + 1)]) for j in 0:K]
    kernel = Th ? :(@views @.. thread = true dest = $rhs) : :(@views @. dest = $rhs)
    return quote
        $(bind...)
        $kernel
        return dest
    end
end

function mul!(
        y::AbstractArray, L::HigherOrderDiff{N, D, K, T, S, Th}, b::AbstractArray
    ) where {N, D, K, T, S, Th}
    check(y, L, b)
    pre, n, post = _finitediff_slabs(L.dim_in, Val(D))
    m = pre * (n - K)
    rows = ntuple(j -> ((j - 1) * pre + 1):((j - 1) * pre + m), Val(K + 1))
    if Th && post == 1
        _hod_stencil!(_slab_view(y, (m,)), _slab_view(b, (pre * n,)), rows, nothing, Val(true))
    else
        cols = 1:post
        _hod_stencil!(_slab_view(y, (m, post)), _slab_view(b, (pre * n, post)), rows, cols, Val(Th))
    end
    return y
end

# The adjoint is `x_r = Σ_j c_j y_{r - j}` over the taps with `0 <= r - j < n - K`. Rows
# `K:(n - K - 1)` along `D` (zero-based) have every tap and are one stencil broadcast; the `K`
# rows at either end have fewer and are summed tap by tap. With `D` the last dimension those
# rows are as large as the interior ones, so they thread too.
function mul!(
        y::AbstractArray, L::AdjointOperator{<:HigherOrderDiff{N, D, K, T, S, Th}}, b::AbstractArray
    ) where {N, D, K, T, S, Th}
    check(y, L, b)
    pre, n, post = _finitediff_slabs(L.A.dim_in, Val(D))
    nout = n - K
    cols = 1:post
    vectors = Th && post == 1
    Y, B = _slab_view(y, (pre * n, post)), _slab_view(b, (pre * nout, post))
    Yv, Bv = _slab_view(y, (length(y),)), _slab_view(b, (length(b),))
    if nout > K
        rows = ntuple(j -> ((K - j + 1) * pre + 1):((nout - j + 1) * pre), Val(K + 1))
        interior = (K * pre + 1):(nout * pre)
        if vectors
            _hod_stencil!(view(Yv, interior), Bv, rows, nothing, Val(true))
        else
            _hod_stencil!(view(Y, interior, cols), B, rows, cols, Val(Th))
        end
    end
    for r in Iterators.flatten((0:(K - 1), max(K, nout):(n - 1)))
        out = (r * pre + 1):((r + 1) * pre)
        jlo, jhi = max(0, r - nout + 1), min(K, r)
        for j in jhi:-1:jlo
            tap = ((r - j) * pre + 1):((r - j + 1) * pre)
            if vectors
                _hod_tap!(view(Yv, out), view(Bv, tap), _hod_coef(K, j), j < jhi, Val(true))
            else
                _hod_tap!(view(Y, out, cols), view(B, tap, cols), _hod_coef(K, j), j < jhi, Val(Th))
            end
        end
    end
    return y
end

# `dest = c * src`, or `dest += c * src` when `accumulate`.
function _hod_tap!(dest, src, c, accumulate::Bool, ::Val{Th}) where {Th}
    if Th && accumulate
        @.. thread = true dest = dest + c * src
    elseif Th
        @.. thread = true dest = c * src
    elseif accumulate
        @. dest = dest + c * src
    else
        @. dest = c * src
    end
    return dest
end

# Properties

domain_type(::HigherOrderDiff{N, D, K, T}) where {N, D, K, T} = T
codomain_type(::HigherOrderDiff{N, D, K, T}) where {N, D, K, T} = T
domain_array_type(::HigherOrderDiff{N, D, K, T, S}) where {N, D, K, T, S} = S
codomain_array_type(::HigherOrderDiff{N, D, K, T, S}) where {N, D, K, T, S} = S
is_thread_safe(::HigherOrderDiff) = true
is_threaded(::HigherOrderDiff{N, D, K, T, S, Th}) where {N, D, K, T, S, Th} = Th

# The same elementwise pass as `FiniteDiff`'s, with `K + 1` reads per output instead of two;
# its crossover is used.
threading_threshold(::Type{<:HigherOrderDiff}) = threading_threshold(FiniteDiff)

function _copy_operator_impl(
        op::HigherOrderDiff{N, D, K, T, S, Th}; storage_type = nothing, threaded = nothing
    ) where {N, D, K, T, S, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return HigherOrderDiff(T, op.dim_in, Val(D), Val(K); array_type = new_at, threaded = new_threaded)
end

function size(L::HigherOrderDiff{N, D, K}) where {N, D, K}
    dim_out = ntuple(i -> i == D ? L.dim_in[i] - K : L.dim_in[i], Val(N))
    return dim_out, L.dim_in
end

const _SUPERSCRIPTS = ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')
_superscript(k::Int) = join(_SUPERSCRIPTS[d + 1] for d in reverse(digits(k)))
fun_name(::HigherOrderDiff{N, D, K}) where {N, D, K} =
    (D == 1 ? "δx" : D == 2 ? "δy" : D == 3 ? "δz" : "δx$D") * _superscript(K)

is_full_row_rank(::HigherOrderDiff) = true
supports_threading(::HigherOrderDiff) = true
