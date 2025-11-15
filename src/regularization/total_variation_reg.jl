"""
	TotalVariation2D(λ)

Create a Total Variation regularization term for 2D images with parameter `λ`. The regularization term is given by
`λ‖Δx‖_{2,1}`, where `Δ` is the 2D finite difference operator computing gradients along the two spatial dimensions.

# Arguments
- `λ`: Regularization parameter, must be a scalar.
"""
struct TotalVariation2D{T} <: Regularization
	λ::T
end

function get_operator(::TotalVariation2D, x::AbstractArray; threaded::Bool=true)
	@assert ndims(x) >= 2 "TotalVariation2D requires at least 2 dimensions in the input variable"
	Δ = Variation(view(x,:,:,(ones(Int, ndims(x) - 2)...)); threaded=threaded && ndims(x) == 2)
	if ndims(x) > 2
		input_dims = (:_, :_, fill(:b, ndims(x) - 2)...)
		image_dims = (:_, fill(:b, ndims(x) - 2)..., :_)
		Δ = BatchOp(Δ, size(x)[3:end], input_dims => image_dims; threaded)
	end
	Δ = reshape(Δ, size(x)[1:2]..., size(Δ, 1)[2:end]...) # not necessary but gives an output shape easier to understand
	if x isa NamedDimsArray
		input_dimnames = dimnames(x)
		output_dimnames = (dimnames(x)[1:2]..., :direction, dimnames(x)[3:end]...)
		Δ = NamedDimsOp{input_dimnames, output_dimnames}(Δ)
	end
	return Δ
end

function get_affected_dims(::TotalVariation2D, acq_info::AcquisitionInfo, image_dims)
	return image_dims[1:2]
end

"""
	TotalVariation3D(λ)

Create a Total Variation regularization term for 3D images with parameter `λ`. The regularization term is given by
`λ‖Δx‖_{2,1}`, where `Δ` is the 3D finite difference operator computing gradients along the three spatial dimensions.

# Arguments
- `λ`: Regularization parameter, must be a scalar.
"""
struct TotalVariation3D{T} <: Regularization
	λ::T
end

function get_operator(::TotalVariation3D, x::AbstractArray; threaded::Bool=true)
	@assert ndims(x) >= 3 "TotalVariation3D requires at least 3 dimensions in the input variable"
	Δ = Variation(view(x,:,:,:,(ones(Int, ndims(x) - 3)...)); threaded=threaded && ndims(x) == 3)
	if ndims(x) > 3
		input_dims = (:_, :_, :_, fill(:b, ndims(x) - 3)...)
		image_dims = (:_, fill(:b, ndims(x) - 3)..., :_)
		Δ = BatchOp(Δ, size(x)[4:end], input_dims => image_dims; threaded)
	end
	Δ = reshape(Δ, size(x)[1:3]..., size(Δ, 1)[2:end]...) # not necessary but gives an output shape easier to interpret
	if x isa NamedDimsArray
		input_dimnames = dimnames(x)
		output_dimnames = (dimnames(x)[1:3]..., :direction, dimnames(x)[4:end]...)
		Δ = NamedDimsOp{input_dimnames, output_dimnames}(Δ)
	end
	return Δ
end

function get_affected_dims(::TotalVariation3D, acq_info::AcquisitionInfo, image_dims)
	return image_dims[1:3]
end

function materialize(
	reg::Union{TotalVariation2D,TotalVariation3D}, x::Variable{T}; threaded::Bool
) where {T}
	Δ = get_operator(reg, ~x; threaded)
	# If Δ carries named dimensions, unwrap before reshaping to a 2D operator
	# to avoid creating a Reshape around a NamedDimsOp with mismatched codomain dims.
	if Δ isa NamedDimsOp
		Δ = parent(Δ)
	end
	Δ = reshape(Δ, length(~x), size(Δ, 1)[end]) # new shape: (length(~x), 2) for 2D or (length(~x), 3) for 3D -> required shape for L_{2,1} mixed norm
	λ = real(T)(reg.λ)
	λ_repr = @sprintf "%g" λ
	x_repr = get_name(x)
	repr = if reg isa TotalVariation2D
		"$λ_repr ⋅ (‖Δˣ$(x_repr)‖₂ + ‖Δʸ$(x_repr)‖₂)"
	else
		"$λ_repr ⋅ (‖Δˣ$(x_repr)‖₂ + ‖Δʸ$(x_repr)‖₂ + ‖Δᶻ$(x_repr)‖₂)"
	end
	return StructuredOptimization.Term(1, NormL21(λ, 2), Δ * x, repr)
end
