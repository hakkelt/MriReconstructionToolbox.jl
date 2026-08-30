"""
	TemporalTotalVariation(λ; time_dim=nothing)

Create a temporal total variation regularization term with parameter `λ`. The regularization term is given by
`λ‖δₜx‖₁`, where `δₜ` is the forward finite difference operator along the temporal dimension specified by
`time_dim`. If `time_dim` is not provided, it will be inferred as the dimension named `:time` if `x` is a
`NamedDimsArray`.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array matching the size of `δₜx`
  (i.e. the size of `x` with the temporal dimension reduced by one).
- `time_dim`: (optional) Dimension along which to take the differences. Can be an `Integer` (1-based index)
or a `Symbol` (dimension name). If not provided, it will be inferred as the dimension named `:time` if `x` is a
`NamedDimsArray`.

# Notes
- This is the anisotropic (ℓ₁) temporal TV used in the compressed-sensing dynamic MRI literature
  (e.g. Feng et al., *Golden-angle radial sparse parallel MRI*, Magn Reson Med 2014). Unlike
  [`TemporalFourier`](@ref) it does not assume periodic or smooth temporal dynamics, so it copes better with
  irregular motion; unlike [`LowRank`](@ref) it makes no assumption about the number of temporal basis
  functions.
- It is frequently combined with [`LowRank`](@ref) (the "L+S"-style models, see [`Component`](@ref)) or with
  spatial regularizers such as [`TotalVariation2D`](@ref).
"""
struct TemporalTotalVariation{T, D} <: Regularization
    λ::T
    time_dim::D
    function TemporalTotalVariation(λ::T; time_dim::D = nothing) where {T, D}
        _check_dim_spec(time_dim, "time_dim")
        return new{T, D}(λ, time_dim)
    end
end

function get_operator(reg::TemporalTotalVariation, x::AbstractArray; threaded::Bool = true)
    dims = dims_of(x)
    time_dim = get_time_dim(reg.time_dim, dims)
    @argcheck size(x, time_dim) > 1 "TemporalTotalVariation requires at least two samples along the temporal dimension"
    δ = FiniteDiff(unname(x), time_dim; threaded)
    if x isa NamedDimsArray
        # The finite difference keeps the dimension order, only the length along `time_dim` shrinks by one.
        δ = NamedDimsOp{dimnames(x), dimnames(x)}(δ)
    end
    return δ
end

function get_affected_dims(reg::TemporalTotalVariation, ::Nothing, image_dims)
    return (image_dims[get_time_dim(reg.time_dim, image_dims)],)
end

# L1 norm is homogeneous of degree 1, so λ scales linearly (see scale_regularization docstring).
function scale_regularization(reg::TemporalTotalVariation, factor::Real)
    return TemporalTotalVariation(reg.λ .* factor; time_dim = reg.time_dim)
end

bind_dimensions(reg::TemporalTotalVariation, image_dims) = TemporalTotalVariation(reg.λ; time_dim = get_time_dim(reg.time_dim, image_dims))

function materialize(reg::TemporalTotalVariation, x::Variable{T}; threaded::Bool) where {T}
    R = real(T)
    λ = R.(reg.λ)
    op = get_operator(reg, ~x; threaded)
    repr = if reg.λ isa AbstractArray
        "‖Γ .* δₜ$(get_name(x))‖₁"
    else
        @sprintf "%g ⋅ ‖δₜ%s‖₁" λ get_name(x)
    end
    return StructuredOptimization.Term(1, NormL1(λ), op * x, repr)
end
