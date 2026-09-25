export BroadCast

abstract type AbstractBroadCast{T, N, M, Threaded} <: AbstractOperator end

struct NoOperatorBroadCast{T, N, M, Threaded, S, Flat} <: AbstractBroadCast{T, N, M, Threaded}
    dim_in::NTuple{N, Int}
    reshaped_dim_in::NTuple{M, Int}
    dim_out::NTuple{M, Int}
    function NoOperatorBroadCast(
            T::Type, S, dim_in::NTuple{N, Int}, reshaped_dim_in::NTuple{M, Int},
            dim_out::NTuple{M, Int}; threaded::Bool = true
        ) where {N, M}
        Base.Broadcast.check_broadcast_shape(dim_out, reshaped_dim_in)
        compact = all(reshaped_dim_in[d] == dim_out[d] for d in 1:N)
        # Two hard prerequisites for the flat kernels below, neither of them a size heuristic.
        # `compact`: they are only correct when the broadcast dimensions are trailing, which
        # is also what lets them be `memcpy`s rather than a Cartesian broadcast. CPU storage:
        # they index elements, and a device array must keep the `.=`/`sum!` forms that run as
        # kernels there. It is a type parameter because the serial path dispatches on it too,
        # not only the threaded one. The size question then goes through the shared policy,
        # which is expressed in elements.
        flat = compact && _is_cpu_storage(S)
        th = flat && _elementwise_threaded(NoOperatorBroadCast, threaded, T, dim_out, S)
        return new{T, N, M, th, S, flat}(dim_in, reshaped_dim_in, dim_out)
    end
end

struct OperatorBroadCast{T, N, M, Threaded, Compact, Imask, L, C, D, K} <: AbstractBroadCast{T, N, M, Threaded}
    A::L
    dim_out::NTuple{M, Int}
    idxs::CartesianIndices{K}
    bufC::C
    bufD::D
    function OperatorBroadCast(
            A, dim_out::NTuple{M, Int}; threaded::Bool = true
        ) where {M}
        Base.Broadcast.check_broadcast_shape(dim_out, size(A, 1))
        threaded = _elementwise_threaded(
            OperatorBroadCast, threaded, codomain_type(A), dim_out,
            _policy_storage(codomain_array_type(A)),
        )
        N = ndims(A, 1)
        T = codomain_type(A)
        dim_in = size(A, 1)
        Imask = Tuple(d ≤ N && (dim_out[d] == dim_in[d]) for d in 1:M)
        broadcast_dims = Tuple(Imask[d] ? 1 : dim_out[d] for d in eachindex(dim_out))
        idxs = CartesianIndices(broadcast_dims)
        compact = all(Imask[1:N])
        bufC = allocate_in_codomain(A)
        if threaded
            bufD = [allocate_in_domain(A) for _ in 1:Threads.nthreads()]
            # Nesting safety: the adjoint loop below threads over `idxs` and calls the
            # wrapped operator inside it, so every per-thread instance -- including the
            # first -- must be non-threaded.
            A = _per_thread_operators(A, Threads.nthreads())
        else
            bufD = allocate_in_domain(A)
        end
        L = typeof(A)
        C = typeof(bufC)
        D = typeof(bufD)
        K = length(broadcast_dims)
        return new{T, N, M, threaded, compact, Imask, L, C, D, K}(A, dim_out, idxs, bufC, bufD)
    end
end

# Constructors

"""
	BroadCast(A::AbstractOperator, dim_out...)

BroadCast the codomain dimensions of an `AbstractOperator`.

```jldoctest
julia> A = Eye(2)
I  ℝ^2 -> ℝ^2

julia> B = BroadCast(A,(2,3))
.I  ℝ^2 -> ℝ^(2, 3)

julia> B*[1.;2.]
2×3 Matrix{Float64}:
 1.0  1.0  1.0
 2.0  2.0  2.0
	
```
"""
function BroadCast(
        A::L, dim_out::NTuple{N, Int}; threaded::Bool = true
    ) where {N, L <: AbstractOperator}
    if length(dim_out) < ndims(A, 1)
        error("dim_out must have at least as many dimensions as the codomain of A")
    end
    if dim_out == size(A, 1)
        return A
    elseif is_eye(A)
        dim_in = size(A, 2)
        reshaped_dim_in = ntuple(d -> d <= ndims(A, 1) ? size(A, 1)[d] : 1, length(dim_out))
        return NoOperatorBroadCast(domain_type(A), domain_array_type(A), dim_in, reshaped_dim_in, dim_out; threaded)
    else
        return OperatorBroadCast(A, dim_out; threaded)
    end
end

# Mappings

# A compact broadcast -- every broadcast axis trailing -- makes the codomain `ncopies` contiguous
# copies of the domain laid out one after the other, so both directions reduce to linear work on
# a `(length(domain), ncopies)` reshape. Going through that reshape matters on its own, before
# any threading: `y .= reshape(b, reshaped_dim_in)` broadcasts against a shape with singleton
# axes, which is the generic Cartesian path, while the copies below are `memcpy`s. Measured on a
# 128x128 image broadcast over 8 coils (2^17 ComplexF32), AMD EPYC 7352, 2026-09-21:
# 151 us for the `.=` form against 24 us here, and 183 us for `sum!` on the 3-D shape against
# 70 us for the flattened reduction.
#
# `Threads.@threads`, not the `Polyester.@batch` this kernel used before, and each threaded
# kernel calls `quiesce_foreign_pools()` first. Two reasons, in that order:
#
#   * A caller that has narrowed the thread budget switches Polyester off for the duration
#     (that is what a `GuardedPool` does), and a `@batch` kernel then runs *serially* without
#     saying so. MRT's solver does exactly this around a whole solve, which is how the 2^18
#     threshold below came to be measured on a path that never ran. `Threads.@threads` is not
#     guarded, so what the policy says about this operator is what happens.
#   * Polyester's workers keep spinning on Julia's threads for a while after a `@batch` region
#     ends, so a `Threads.@threads` region opened next waits for them rather than running on
#     them -- 5.7 us against 219.9 us for an empty loop on 8 threads, AMD EPYC 7352,
#     Julia 1.13.0, 2026-09-21. `quiesce_foreign_pools` parks them for about 0.1 us and the
#     penalty is gone; the batch and block loops get the same treatment from
#     `@budgeted_threads`, which calls it for them.
#
# The `@batch` form is still marginally the faster of the two in a process where nothing parks
# it (13.9 us against 18.9 us at 2^17), which is the price paid here for a kernel that behaves
# the same in both settings. See `threading_policy.jl` for the policy note.
function _copy_flat!(_y, _x)
    @inbounds for k in axes(_y, 2)
        copyto!(_y, (k - 1) * length(_x) + 1, _x, 1, length(_x))
    end
    return _y
end

function _copy_flat_threaded!(_y, _x)
    ncopies = size(_y, 2)
    nchunks, per_copy = _broadcast_chunking(length(_x), ncopies)
    quiesce_foreign_pools()
    Threads.@threads for t in 1:nchunks
        k, j = fldmod1(t, per_copy)
        lo, hi = _chunk_range(length(_x), per_copy, j)
        lo > hi && continue
        @inbounds copyto!(_y, (k - 1) * length(_x) + lo, _x, lo, hi - lo + 1)
    end
    return _y
end

function _sum_flat!(_y, _b)
    @inbounds for i in eachindex(_y)
        acc = zero(eltype(_y))
        for k in axes(_b, 2)
            acc += _b[i, k]
        end
        _y[i] = acc
    end
    return _y
end

function _sum_flat_threaded!(_y, _b)
    nchunks = min(Threads.nthreads(), length(_y))
    quiesce_foreign_pools()
    Threads.@threads for t in 1:nchunks
        lo, hi = _chunk_range(length(_y), nchunks, t)
        @inbounds for i in lo:hi
            acc = zero(eltype(_y))
            for k in axes(_b, 2)
                acc += _b[i, k]
            end
            _y[i] = acc
        end
    end
    return _y
end

# One copy per worker when there are at least as many copies as threads; otherwise split each
# copy further, so that broadcasting one large image over two coils still occupies the machine.
function _broadcast_chunking(len::Int, ncopies::Int)
    nt = Threads.nthreads()
    per_copy = ncopies >= nt ? 1 : min(cld(nt, ncopies), len)
    return ncopies * per_copy, per_copy
end

function _chunk_range(len::Int, nchunks::Int, i::Int)
    size = cld(len, nchunks)
    return (i - 1) * size + 1, min(len, i * size)
end

# The copy count is spelled out rather than left to `:`, which JET's `@test_opt` infers as `Any`;
# the copies above index the flat layout directly for the same reason, instead of going through
# a `view` of one column.
_flat_pair(y, x) = (reshape(y, length(x), length(y) ÷ length(x)), vec(x))

# Kept for callers outside this file (`OperatorBroadCast`), and as the single place the compact
# layout assumption is written down.
function tbroadcast!(y, x)
    _y, _x = _flat_pair(y, x)
    return _copy_flat_threaded!(_y, _x)
end

# A broadcast is an expansion into copies, and its adjoint the sum of those copies, so both join
# a pointwise run of a `Compose` when the broadcast axes are adjacent.
_pw_kind(::Type{<:NoOperatorBroadCast{T, N, M, Th, S}}) where {T, N, M, Th, S} =
    S <: Array ? PwExpandKind() : PwNoneKind()
_pw_kind(::Type{<:AdjointOperator{<:NoOperatorBroadCast{T, N, M, Th, S}}}) where {T, N, M, Th, S} =
    S <: Array ? PwReduceKind() : PwNoneKind()
_pw_layout(A::AdjointOperator{<:NoOperatorBroadCast}) = _pw_layout(A.A)
function _pw_layout(A::NoOperatorBroadCast)
    r, o = A.reshaped_dim_in, A.dim_out
    first_axis, last_axis = 0, 0
    for d in eachindex(o)
        if r[d] != o[d]
            first_axis == 0 && (first_axis = d)
            last_axis = d
        end
    end
    first_axis == 0 && return (prod(o), 1)
    inner, K = 1, 1
    for d in eachindex(o)
        if d < first_axis
            inner *= o[d]
        elseif d <= last_axis
            r[d] == 1 || return nothing
            K *= o[d]
        end
    end
    return inner, K
end

# NoOperatorBroadCast
function mul!(y, A::NoOperatorBroadCast{T, N, M, false, S, false}, b) where {T, N, M, S}
    check(y, A, b)
    b = reshape(b, A.reshaped_dim_in)
    return y .= b # not compact: the broadcast axes are interleaved, so only `.=` is correct
end

function mul!(y, A::NoOperatorBroadCast{T, N, M, false, S, true}, b) where {T, N, M, S}
    check(y, A, b)
    _y, _x = _flat_pair(y, b)
    _copy_flat!(_y, _x)
    return y
end

function mul!(y, A::NoOperatorBroadCast{T, N, M, true}, b) where {T, N, M}
    check(y, A, b)
    _y, _x = _flat_pair(y, b)
    _copy_flat_threaded!(_y, _x)
    return y
end

function mul!(
        y, A::AdjointOperator{<:NoOperatorBroadCast{T, N, M, Th, S, false}}, b
    ) where {T, N, M, Th, S}
    check(y, A, b)
    y = reshape(y, A.A.reshaped_dim_in)
    return sum!(y, b)
end

function mul!(
        y, A::AdjointOperator{<:NoOperatorBroadCast{T, N, M, false, S, true}}, b
    ) where {T, N, M, S}
    check(y, A, b)
    _b, _y = _flat_pair(b, y)
    _sum_flat!(_y, _b)
    return y
end

# The adjoint reduction crosses over earlier than the forward copy -- 2^14 against 2^16 on the
# sweep in `threading_threshold(::Type{<:AbstractBroadCast})` -- but both directions share the
# operator's single `Threaded` flag, which the forward one sets. Between 2^14 and 2^16 the
# adjoint therefore runs serial although threading it would pay 1.4x-2.1x.
function mul!(
        y, A::AdjointOperator{<:NoOperatorBroadCast{T, N, M, true, S, true}}, b
    ) where {T, N, M, S}
    check(y, A, b)
    _b, _y = _flat_pair(b, y)
    _sum_flat_threaded!(_y, _b)
    return y
end

# OperatorBroadCast

function mul!(y, R::OperatorBroadCast, b) # Non-threaded
    check(y, R, b)
    mul!(R.bufC, R.A, b)
    return y .= R.bufC # non-threaded broadcasting
end

function mul!(y, R::OperatorBroadCast{T, N, M, true, Compact}, b) where {T, N, M, Compact} # Threaded
    check(y, R, b)
    mul!(R.bufC, R.A[1], b)
    if Compact
        return tbroadcast!(y, R.bufC) # threaded broadcasting
    else
        return y .= R.bufC # non-threaded broadcasting
    end
end

function mul!(y, A::AdjointOperator{<:OperatorBroadCast{T, N, M, false}}, b) where {T, N, M} # Non-threaded
    check(y, A, b)
    R = A.A
    for idx in R.idxs
        b_slice = get_input_slice(R, idx, b)
        if size(b_slice) != size(R.A, 1)
            b_slice = reshape(b_slice, size(R.A, 1))
        end
        mul!(R.bufD, R.A', b_slice)
        if idx == first(R.idxs)
            y .= R.bufD
        else
            y .+= R.bufD
        end
    end
    return y
end

function mul!(y, A::AdjointOperator{<:OperatorBroadCast{T, N, M, true}}, b) where {T, N, M} # Threaded
    check(y, A, b)
    R = A.A
    fill!(y, 0)
    lock = ReentrantLock()
    thread_count = min(Threads.nthreads(), length(R.idxs))
    batch_size = length(R.idxs) / thread_count
    # Budgeted: the body calls `mul!` on arbitrary sub-operators, which may themselves use
    # BLAS/FFTW, so without budgeting this loop is a genuine oversubscription source.
    @budgeted_threads for t in 1:thread_count
        idx_start = max(1, floor(Int, (t - 1) * batch_size + 1))
        idx_end = min(length(R.idxs), floor(Int, t * batch_size))
        for i in idx_start:idx_end
            b_slice = get_input_slice(R, R.idxs[i], b)
            if size(b_slice) != size(R.A[t], 1)
                b_slice = reshape(b_slice, size(R.A[t], 1))
            end
            mul!(R.bufD[t], R.A[t]', b_slice)
            @lock lock y .+= R.bufD[t]
        end
    end
    return y
end

# Properties
function Base.:(==)(R1::NoOperatorBroadCast{T, N, M}, R2::NoOperatorBroadCast{T, N, M}) where {T, N, M}
    return R1.dim_in == R2.dim_in && R1.dim_out == R2.dim_out
end
function Base.:(==)(
        R1::OperatorBroadCast{T, N, M}, R2::OperatorBroadCast{T, N, M}
    ) where {T, N, M}
    return R1.A == R2.A && R1.dim_out == R2.dim_out
end

size(R::NoOperatorBroadCast) = (R.dim_out, R.dim_in)
size(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = (R.dim_out, size(R.A, 2))
size(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = (R.dim_out, size(R.A[1], 2))

domain_type(::NoOperatorBroadCast{T}) where {T} = T
domain_type(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = domain_type(R.A)
domain_type(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = domain_type(R.A[1])
codomain_type(::NoOperatorBroadCast{T}) where {T} = T
codomain_type(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = codomain_type(R.A)
codomain_type(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = codomain_type(R.A[1])
domain_array_type(::NoOperatorBroadCast{T, N, M, Threaded, S}) where {T, N, M, Threaded, S} = S
domain_array_type(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = domain_array_type(R.A)
domain_array_type(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = domain_array_type(R.A[1])
codomain_array_type(::NoOperatorBroadCast{T, N, M, Threaded, S}) where {T, N, M, Threaded, S} = S
codomain_array_type(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = codomain_array_type(R.A)
codomain_array_type(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = codomain_array_type(R.A[1])

is_thread_safe(::NoOperatorBroadCast) = true
is_thread_safe(::OperatorBroadCast) = false

is_linear(::NoOperatorBroadCast) = true
is_linear(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = is_linear(R.A)
is_linear(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = is_linear(R.A[1])
is_affine(::NoOperatorBroadCast) = true
is_affine(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = is_affine(R.A)
is_affine(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = is_affine(R.A[1])
is_null(R::NoOperatorBroadCast) = false
is_null(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = is_null(R.A)
is_null(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = is_null(R.A[1])

fun_name(::NoOperatorBroadCast) = ".I"
fun_name(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = "." * fun_name(R.A)
fun_name(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = "." * fun_name(R.A[1])
remove_displacement(R::NoOperatorBroadCast) = R
function remove_displacement(R::OperatorBroadCast{T, N, M, false, Imask}) where {T, N, M, Imask}
    new_A = remove_displacement(R.A)
    return OperatorBroadCast(new_A, R.dim_out; threaded = false)
end
function remove_displacement(R::OperatorBroadCast{T, N, M, true, Imask}) where {T, N, M, Imask}
    new_A = remove_displacement(R.A[1])
    return OperatorBroadCast(new_A, R.dim_out; threaded = true)
end

has_fast_opnorm(::NoOperatorBroadCast) = true
has_fast_opnorm(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = has_fast_opnorm(R.A)
has_fast_opnorm(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = has_fast_opnorm(R.A[1])
function LinearAlgebra.opnorm(R::NoOperatorBroadCast{T, N, M}) where {T, N, M}
    return real(T)(sqrt(prod(R.dim_out[d] for d in 1:M if R.dim_out[d] != R.reshaped_dim_in[d])))
end
function LinearAlgebra.opnorm(R::OperatorBroadCast{T, N, M, false}) where {T, N, M}
    return _replication_factor(R) * LinearAlgebra.opnorm(R.A)
end
function LinearAlgebra.opnorm(R::OperatorBroadCast{T, N, M, true}) where {T, N, M}
    return _replication_factor(R) * LinearAlgebra.opnorm(R.A[1])
end

function opnorm_bound(R::OperatorBroadCast{T, N, M, false}) where {T, N, M}
    return _replication_factor(R) * opnorm_bound(R.A)
end
function opnorm_bound(R::OperatorBroadCast{T, N, M, true}) where {T, N, M}
    return _replication_factor(R) * opnorm_bound(R.A[1])
end

"""
	_replication_factor(R::OperatorBroadCast)

How much a broadcast scales the norm of what it replicates: `sqrt` of the number of copies each
entry of the inner operator's output is written to.

Replicating a vector `c` times multiplies its 2-norm by exactly `sqrt(c)`, so this factor is
exact rather than an inequality. Leaving it out — which `opnorm(::OperatorBroadCast)` did, by
forwarding straight to the inner operator — under-reports the norm by `sqrt(c)`, and an
under-reported norm is the direction that breaks a Lipschitz constant.
"""
function _replication_factor(R::OperatorBroadCast{T}) where {T}
    # `idxs` is built by the constructor as the Cartesian range of exactly the broadcast
    # dimensions, so its length is the number of copies with nothing left to re-derive.
    return real(T)(sqrt(length(R.idxs)))
end

"""
	_fused_pair_opnorm(B::NoOperatorBroadCast, D::DiagOp)

Exact `‖D ∘ B‖` for a `DiagOp` applied on top of a replicating `BroadCast`.

`D ∘ B` is block diagonal with one single-column block per input position, so its norm is the
largest block norm [1, §2.1] — that is, `max_r sqrt(sum_c |d[r, c]|²)` over the broadcast
positions `c`. The submultiplicative product gives `maximum(abs, d) * sqrt(c)` instead, which
overshoots by up to the square root of the number of copies.

## References

1. Horn, Johnson, "Topics in Matrix Analysis", Cambridge (1991).
"""
function _fused_pair_opnorm(B::NoOperatorBroadCast{T, N, M}, D::DiagOp) where {T, N, M}
    size(D, 2) == B.dim_out || return nothing
    # A `DiagOp` may hold a single number instead of an array, and then every copy is weighted
    # the same, which is the one case the submultiplicative product already gets exactly right.
    D.d isa AbstractArray || return nothing
    bdims = Tuple(d for d in 1:M if B.reshaped_dim_in[d] != B.dim_out[d])
    isempty(bdims) && return nothing
    return float(sqrt(maximum(sum(abs2, D.d; dims = bdims))))
end

# utils

function permute(R::OperatorBroadCast{T, N, M, false}, p::AbstractVector{Int}) where {T, N, M}
    return BroadCast(permute(R.A, p), R.dim_out; threaded = false)
end
function permute(R::OperatorBroadCast{T, N, M, true}, p::AbstractVector{Int}) where {T, N, M}
    return BroadCast(permute(R.A[1], p), R.dim_out; threaded = true)
end

function _copy_operator_impl(
        op::NoOperatorBroadCast{T, N, M, Th, S}; storage_type = nothing, threaded = nothing
    ) where {T, N, M, Th, S}
    new_threaded = threaded === nothing ? Th : threaded
    new_S = storage_type === nothing ? S : storage_type{T}
    return NoOperatorBroadCast(T, new_S, op.dim_in, op.reshaped_dim_in, op.dim_out; threaded = new_threaded)
end

function _copy_operator_impl(
        op::OperatorBroadCast{T, N, M, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, M, Th}
    new_threaded = threaded === nothing ? Th : threaded
    inner_op = Th ? op.A[1] : op.A
    new_op = copy_operator(inner_op; storage_type, threaded)
    return BroadCast(new_op, op.dim_out; threaded = new_threaded)
end

@generated function get_input_slice(
        ::OperatorBroadCast{T, N, M, Threaded, Compact, Imask}, idx::CartesianIndex, b
    ) where {T, N, M, Threaded, Compact, Imask}
    return quote
        @ncall($M, view, b, d -> Imask[d] ? Colon() : idx[d])
    end
end

"""
Broadcasting is pure data movement, so it used to inherit `THRESHOLD_MEMORY_BOUND` (2^18). That
constant was swept for `Polyester.@batch` against FastBroadcast; the kernels here now use
`Threads.@threads` (see the note above `_copy_flat!`), whose crossover is two powers of two
earlier, and the serial side got faster at the same time, so the sweep was re-run for the pair
that actually ships.

PROVENANCE: measured. AMD EPYC 7352 24-Core, 8 Julia threads, OPENBLAS_NUM_THREADS=1,
Julia 1.13.0, 2026-09-21, in a process that never runs a `Polyester.@batch` -- an image
broadcast over 8 copies, serial/threaded ratio (>1 means threading pays):

| elements | forward ComplexF32 | forward ComplexF64 | adjoint ComplexF32 | adjoint ComplexF64 |
|   2^13   |       0.18x        |       0.47x        |       0.79x        |       0.90x        |
|   2^14   |       0.36x        |       0.78x        |       1.42x        |       1.61x        |
|   2^15   |       0.81x        |       1.14x        |       2.12x        |       2.30x        |
|   2^16   |       1.19x        |       1.50x        |       2.95x        |       3.05x        |
|   2^17   |       1.48x        |       1.88x        |       3.31x        |       2.79x        |
|   2^18   |       2.19x        |       2.53x        |       2.98x        |       3.88x        |
|   2^19   |       3.27x        |       2.76x        |       4.12x        |       4.15x        |

2^16 is the first size at which the forward copy pays for both element types, and the forward
is what sets the flag; the adjoint's own crossover at 2^14 is recorded above `mul!` for the
adjoint, which cannot act on it separately.
"""
threading_threshold(::Type{<:AbstractBroadCast}) = 2^16
supports_threading(::AbstractBroadCast) = true

# The `Threaded` type parameter was already there; without these methods the trait fell
# through to the `false` default and contradicted the operator's own dispatch.
is_threaded(::NoOperatorBroadCast{T, N, M, Th}) where {T, N, M, Th} = Th
# For the operator flavour the wrapped operator can thread independently of the broadcast.
is_threaded(R::OperatorBroadCast{T, N, M, Th}) where {T, N, M, Th} =
    Th || any(is_threaded, _broadcast_children(R))
_broadcast_children(R::OperatorBroadCast{T, N, M, false}) where {T, N, M} = (R.A,)
_broadcast_children(R::OperatorBroadCast{T, N, M, true}) where {T, N, M} = R.A
_children(R::OperatorBroadCast) = _broadcast_children(R)
