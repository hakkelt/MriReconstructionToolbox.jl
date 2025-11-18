"""
	Tikhonov(λ)

Create a Tikhonov regularization term with parameter `λ`.

The regularization term is given by `λ²‖x‖₂²`, or `‖Γ .* x‖₂²` if `λ` is an array `Γ` of the same size as `x`.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array of the same size as `x`.

# Notes
- This regularization term is also known as L2 regularization or ridge regression.
- The squared parameter `λ²` is used in the formulation to align with common conventions in
 Tikhonov regularization literature.
"""
struct Tikhonov{T} <: Regularization
	λ::T
end

get_operator(::Tikhonov, x::AbstractArray; threaded::Bool=true) = Eye(x)
get_operator(::Tikhonov, x::NamedDimsArray; threaded::Bool=true) = NamedDimsOp{dimnames(x), dimnames(x)}(Eye(parent(x)))

function get_affected_dims(::Tikhonov, acq_info::AcquisitionInfo, image_dims)
	return () # Tikhonov regularization applies element-wise, so no specific dimensions are affected
end

function materialize(reg::Tikhonov, x::Variable{T}; threaded::Bool) where {T}
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

get_operator(::L1Image, x::AbstractArray; threaded::Bool=true) = Eye(x)
get_operator(::L1Image, x::NamedDimsArray; threaded::Bool=true) = NamedDimsOp{dimnames(x), dimnames(x)}(Eye(parent(x)))

function get_affected_dims(::L1Image, acq_info::AcquisitionInfo, image_dims)
	return () # L1-image regularization applies element-wise, so no specific dimensions are affected
end

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
