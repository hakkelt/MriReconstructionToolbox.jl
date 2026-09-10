"""
	NonNegative(; complex_handling = :error)

Constrain the reconstructed image to be non-negative (element-wise `x ≥ 0`).

# Arguments
- `complex_handling::Symbol = :error`: what to do when the image is complex-valued. `:error`
  (the default) throws an `ArgumentError`, matching plain non-negativity's requirement that `x`
  be ordered. `:real` instead projects onto the real, non-negative numbers — the imaginary part
  is discarded and the real part clamped at `0` — following RegularizedLeastSquares.jl's
  `PositiveRegularization` (`enfReal!` then `enfPos!`).

# Notes
- This is a constraint, not a penalty: it has no `λ` and is enforced exactly by projection, so it never
  trades off against data consistency.
- For a real-valued image, `complex_handling` has no effect: the constraint is always the plain
  non-negative orthant. It only changes what happens when `x` is complex (e.g. magnitude-only or
  real-valued phantom reconstructions carried in a complex array, and parameter maps in
  quantitative MRI).
- Non-negativity is a classic constraint in iterative image reconstruction; in MRI it is most useful for
  quantitative maps and for real-valued reconstructions with a resolved phase
  (see Fessler, *Model-based image reconstruction for MRI*, IEEE Signal Process Mag 2010).
"""
struct NonNegative <: Regularization
    complex_handling::Symbol
    function NonNegative(; complex_handling::Symbol = :error)
        return new(_checked_complex_handling(complex_handling))
    end
end

# The two constraints take the same option, so they validate it in one place.
function _checked_complex_handling(complex_handling::Symbol)
    @argcheck complex_handling in (:error, :real) "complex_handling must be :error or :real, got $complex_handling"
    return complex_handling
end

"""
	BoxConstraint(lower, upper; complex_handling = :error)

Constrain the reconstructed image element-wise to the interval `[lower, upper]`.

# Arguments
- `lower`: Lower bound, a scalar or an array of the same size as `x`.
- `upper`: Upper bound, a scalar or an array of the same size as `x`.
- `complex_handling::Symbol = :error`: what to do when the image is complex-valued, with the
  same meaning as [`NonNegative`](@ref)'s: `:error` throws, `:real` projects onto the real box
  (imaginary part discarded, real part clamped to `[lower, upper]`).

# Notes
- Like [`NonNegative`](@ref), this is an exact constraint enforced by projection.
- Useful for quantitative maps with a physically meaningful range (e.g. proton density in `[0, 1]`,
  relaxation rates bounded from above).
"""
struct BoxConstraint{L, U} <: Regularization
    lower::L
    upper::U
    complex_handling::Symbol
    function BoxConstraint(lower::L, upper::U; complex_handling::Symbol = :error) where {L, U}
        @argcheck all(lower .<= upper) "lower bound must not exceed upper bound"
        return new{L, U}(lower, upper, _checked_complex_handling(complex_handling))
    end
end

const _Constraint = Union{NonNegative, BoxConstraint}

get_operator(::_Constraint, x::AbstractArray; threaded::Bool = true) = identity_operator(x)

# Constraints act element-wise, so no dimension is coupled -- unless the bounds are given as arrays, which
# have the size of the full image and therefore must not be split over batch dimensions.
get_affected_dims(::NonNegative, ::Nothing, image_dims) = ()

function get_affected_dims(reg::BoxConstraint, ::Nothing, image_dims)
    array_bounds = reg.lower isa AbstractArray || reg.upper isa AbstractArray
    return array_bounds ? Tuple(image_dims) : ()
end

# Non-negativity is scale invariant for a positive scaling factor; box bounds live in the same units as the
# image, so they follow the variable scaling (see scale_regularization docstring).
scale_regularization(reg::NonNegative, ::Real) = reg
function scale_regularization(reg::BoxConstraint, factor::Real)
    return BoxConstraint(reg.lower .* factor, reg.upper .* factor; complex_handling = reg.complex_handling)
end

function _check_complex_handling(reg::Regularization, ::Type{T}) where {T}
    T <: Real && return nothing
    reg.complex_handling === :real && return nothing
    return @argcheck false "$(nameof(typeof(reg))) is only defined for real-valued images, got eltype $T (pass complex_handling = :real to project onto the real orthant/box instead)"
end

function materialize(reg::NonNegative, x::Variable{T}; threaded::Bool) where {T}
    _check_complex_handling(reg, T)
    op = get_operator(reg, ~x; threaded)
    f = T <: Real ? IndNonnegative() : IndRealNonnegative()
    return StructuredOptimization.Term(1, f, op * x, "$(get_name(x)) ≥ 0")
end

function materialize(reg::BoxConstraint, x::Variable{T}; threaded::Bool) where {T}
    _check_complex_handling(reg, T)
    if reg.lower isa AbstractArray
        @argcheck size(reg.lower) == size(~x) "Incompatible sizes"
    end
    if reg.upper isa AbstractArray
        @argcheck size(reg.upper) == size(~x) "Incompatible sizes"
    end
    R = real(T)
    lower = reg.lower isa AbstractArray ? R.(reg.lower) : R(reg.lower)
    upper = reg.upper isa AbstractArray ? R.(reg.upper) : R(reg.upper)
    op = get_operator(reg, ~x; threaded)
    repr = if reg.lower isa AbstractArray || reg.upper isa AbstractArray
        "lower ≤ $(get_name(x)) ≤ upper"
    else
        @sprintf "%g ≤ %s ≤ %g" lower get_name(x) upper
    end
    f = T <: Real ? IndBox(lower, upper) : IndRealBox(lower, upper)
    return StructuredOptimization.Term(1, f, op * x, repr)
end
