# Preallocation interface
#
# Some `prox!`/`gradient!` implementations need scratch space. By default they
# allocate it on every call, which is wasteful inside an iterative solver.
# `preallocate(f, x)` returns a copy of `f` carrying buffers sized for inputs
# like `x`, so that subsequent calls allocate nothing.
#
# Operators that need scratch space carry a trailing `buf` field, which is
# either `nothing` (not preallocated) or a concrete `NamedTuple`/struct holding
# the buffers. Since the two cases differ in type, the branch in `get_buffers`
# folds away at compile time, and a single method body serves both APIs.

export preallocate

"""
    preallocate(f, x) -> f′

Return a copy of `f` carrying scratch buffers sized for inputs like `x`. `f′`
computes exactly the same values as `f`, but `prox!` and `gradient!` on `f′`
allocate nothing (or, for factorization-based operators, strictly less).

`x` is only used as a prototype: its `size`, `eltype` and array type are read,
its contents are ignored. Operators that need no scratch space return `f`
itself.

```julia
f = NuclearNorm(1e-3)
fp = preallocate(f, rand(128, 64))
prox!(y, fp, x, gamma)   # no scratch allocation
```

Calling a preallocated operator with an input that does not match the prototype
throws a `DimensionMismatch`.

!!! warning
    The returned operator is *not* thread-safe: all calls share the same
    buffers. Use one preallocated copy per task.

See also [`is_preallocated`](@ref).
"""
preallocate(f, x) = f

"""
    is_preallocated(f)

Return `true` if `f` carries scratch buffers, i.e. if it was obtained from
[`preallocate`](@ref).
"""
is_preallocated(f::T) where T = hasfield(T, :buf) && getfield(f, :buf) !== nothing

# Operators that were already stateful before this interface existed
# (`LeastSquares*`, `Quadratic*`, `IndAffine*`, `IndGraph*`) carry their own
# scratch space and cached factorization, so the generic fallback above is
# already correct for them: `preallocate` returns them unchanged.

# Input signature: what a preallocated operator was sized for.
#
# The size is a value, the number of dimensions, element type and array wrapper
# are type parameters, so the corresponding comparisons fold to constants.

struct InputSignature{N, T, W}
    size::NTuple{N, Int}
end

array_wrapper(x::AbstractArray) = Base.typename(typeof(x)).wrapper

input_signature(x::AbstractArray) =
    InputSignature{ndims(x), eltype(x), array_wrapper(x)}(size(x))

input_signature(xs::Tuple) = map(input_signature, xs)

function Base.show(io::IO, sig::InputSignature{N, T, W}) where {N, T, W}
    print(io, "a ", sig.size, " ", W, "{", T, "}")
end

# The check is always on: no @boundscheck, no @assert (both are elidable).

@inline function check_signature(sig::InputSignature{N, T, W}, x, f) where {N, T, W}
    if ndims(x) != N || size(x) != sig.size || eltype(x) !== T || array_wrapper(x) !== W
        mismatch_error(sig, x, f)
    end
    return nothing
end

@inline function check_signature(sigs::Tuple, xs::Tuple, f)
    if length(sigs) != length(xs)
        throw(DimensionMismatch(
            "$(nameof(typeof(f))) was preallocated for $(length(sigs)) blocks, got $(length(xs))"
        ))
    end
    map((sig, x) -> check_signature(sig, x, f), sigs, xs)
    return nothing
end

@inline check_signature(sigs::Tuple, x, f) = mismatch_error(sigs, x, f)
@inline check_signature(sig::InputSignature, xs::Tuple, f) = mismatch_error(sig, xs, f)

@noinline function mismatch_error(sig, x, f)
    throw(DimensionMismatch(
        "$(nameof(typeof(f))) was preallocated for $(sig) input, got $(summary(x))"
    ))
end

"""
    check_input(f, x)

Verify that `x` matches the prototype `f` was preallocated for, throwing
`DimensionMismatch` otherwise. A no-op for operators that are not preallocated.
"""
@inline check_input(f, x) = _check_input(f.buf, x, f)
@inline _check_input(::Nothing, x, f) = nothing
@inline _check_input(buf, x, f) = check_signature(buf.sig, x, f)

"""
    get_buffers(f, x)

Return the scratch buffers of `f` for input `x`. If `f` is preallocated, the
input is checked against the recorded prototype; otherwise the buffers are
allocated on the spot, which preserves the allocating API.
"""
@inline get_buffers(f, x) = _get_buffers(f, f.buf, x)
@inline _get_buffers(f, ::Nothing, x) = _require_buffers(preallocate(f, x).buf, f, x)
@inline _get_buffers(f, buf, x) = (check_signature(buf.sig, x, f); buf)

@inline _require_buffers(buf, f, x) = buf
@noinline _require_buffers(::Nothing, f, x) = throw(ArgumentError(
    "no `preallocate` method for $(nameof(typeof(f))) with an input of type $(typeof(x))"
))
