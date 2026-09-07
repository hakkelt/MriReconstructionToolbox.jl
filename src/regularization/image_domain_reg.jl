"""
	L2Image(λ)
	Tikhonov(λ)

Create an ℓ₂ image-domain regularization term with parameter `λ`.

The regularization term is given by `λ²‖x‖₂²`, or `‖Γ .* x‖₂²` if `λ` is an array `Γ` of the same size as `x`.

`Tikhonov` is an exported alias: Tikhonov regularization is the textbook name for this term, and it
is the name most readers will reach for first. The two names are the same type.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array of the same size as `x`.

# Notes
- This regularization term is also known as ridge regression, or (in BART's `-R Q`) the
  ℓ₂-norm image-domain penalty.
- The squared parameter `λ²` is used in the formulation to align with common conventions in
  the Tikhonov regularization literature.
"""
struct L2Image{T} <: Regularization
    λ::T
end

const Tikhonov = L2Image

get_operator(::L2Image, x::AbstractArray; threaded::Bool = true) = identity_operator(x)

function get_affected_dims(::L2Image, ::Nothing, image_dims)
    return () # L2Image regularization applies element-wise, so no specific dimensions are affected
end

# λ²‖x‖² is homogeneous of degree 2 in x, same as the data-consistency term, so no correction is
# needed (see scale_regularization docstring): falls back to the generic no-op method.

function materialize(reg::L2Image, x::Variable{T}; threaded::Bool) where {T}
    if reg.λ isa AbstractArray
        @argcheck size(reg.λ) == size(x) "Incompatible sizes"
    end
    R = real(T)
    λ = R.(reg.λ)
    op = get_operator(reg, ~x; threaded)
    repr = if reg.λ isa AbstractArray
        "‖Γ .* $(get_name(x))‖₂²"
    else
        @sprintf "‖%g ⋅ %s‖₂²" λ get_name(x)
    end
    return StructuredOptimization.Term(1, SqrNormL2(2 .* λ .^ 2), op * x, repr)
end

"""
	L1Image(λ)

Create a L1 image regularization term with parameter `λ`. The regularization term is given by `λ‖x‖₁`,
or `‖Γ .* x‖₁` if `λ` is an array `Γ` of the same size as `x`.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array of the same size as `x`.
"""
struct L1Image{T} <: Regularization
    λ::T
end

get_operator(::L1Image, x::AbstractArray; threaded::Bool = true) = identity_operator(x)

function get_affected_dims(::L1Image, ::Nothing, image_dims)
    return () # L1-image regularization applies element-wise, so no specific dimensions are affected
end

# L1 norm is homogeneous of degree 1, so λ scales linearly (see scale_regularization docstring).
scale_regularization(reg::L1Image, factor::Real) = L1Image(reg.λ .* factor)

function materialize(reg::L1Image, x::Variable{T}; threaded::Bool) where {T}
    R = real(T)
    λ = R.(reg.λ)
    op = get_operator(reg, ~x; threaded)
    repr = if reg.λ isa AbstractArray
        "‖Γ .* $(get_name(x))‖₁"
    else
        @sprintf "%g ⋅ ‖%s‖₁" λ get_name(x)
    end
    return StructuredOptimization.Term(1, NormL1(λ), op * x, repr)
end
