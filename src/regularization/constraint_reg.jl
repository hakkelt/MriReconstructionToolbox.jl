"""
	NonNegative()

Constrain the reconstructed image to be non-negative (element-wise `x ≥ 0`).

# Notes
- This is a constraint, not a penalty: it has no `λ` and is enforced exactly by projection, so it never
  trades off against data consistency.
- Only applicable to real-valued images (e.g. magnitude-only or real-valued phantom reconstructions, and
  parameter maps in quantitative MRI). Trying to use it on complex data throws an `ArgumentError`.
- Non-negativity is a classic constraint in iterative image reconstruction; in MRI it is most useful for
  quantitative maps and for real-valued reconstructions with a resolved phase
  (see Fessler, *Model-based image reconstruction for MRI*, IEEE Signal Process Mag 2010).
"""
struct NonNegative <: Regularization end

"""
	BoxConstraint(lower, upper)

Constrain the reconstructed image element-wise to the interval `[lower, upper]`.

# Arguments
- `lower`: Lower bound, a scalar or an array of the same size as `x`.
- `upper`: Upper bound, a scalar or an array of the same size as `x`.

# Notes
- Like [`NonNegative`](@ref), this is an exact constraint enforced by projection, and only applicable to
  real-valued images.
- Useful for quantitative maps with a physically meaningful range (e.g. proton density in `[0, 1]`,
  relaxation rates bounded from above).
"""
struct BoxConstraint{L, U} <: Regularization
    lower::L
    upper::U
    function BoxConstraint(lower::L, upper::U) where {L, U}
        @argcheck all(lower .<= upper) "lower bound must not exceed upper bound"
        return new{L, U}(lower, upper)
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
    return BoxConstraint(reg.lower .* factor, reg.upper .* factor)
end

function _check_real(reg::Regularization, ::Type{T}) where {T}
    return @argcheck T <: Real "$(nameof(typeof(reg))) is only defined for real-valued images, got eltype $T"
end

function materialize(reg::NonNegative, x::Variable{T}; threaded::Bool) where {T}
    _check_real(reg, T)
    op = get_operator(reg, ~x; threaded)
    return StructuredOptimization.Term(1, IndNonnegative(), op * x, "$(get_name(x)) ≥ 0")
end

function materialize(reg::BoxConstraint, x::Variable{T}; threaded::Bool) where {T}
    _check_real(reg, T)
    if reg.lower isa AbstractArray
        @argcheck size(reg.lower) == size(~x) "Incompatible sizes"
    end
    if reg.upper isa AbstractArray
        @argcheck size(reg.upper) == size(~x) "Incompatible sizes"
    end
    op = get_operator(reg, ~x; threaded)
    lower = reg.lower isa AbstractArray ? T.(reg.lower) : T(reg.lower)
    upper = reg.upper isa AbstractArray ? T.(reg.upper) : T(reg.upper)
    repr = if reg.lower isa AbstractArray || reg.upper isa AbstractArray
        "lower ≤ $(get_name(x)) ≤ upper"
    else
        @sprintf "%g ≤ %s ≤ %g" lower get_name(x) upper
    end
    return StructuredOptimization.Term(1, IndBox(lower, upper), op * x, repr)
end
