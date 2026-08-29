"""
	JointSparsity(λ; dim)

Create a joint-sparsity (group-sparsity, ℓ₂,₁ mixed norm) regularization term with parameter `λ`.
The regularization term is given by `λ ∑ᵢ ‖xᵢ‖₂`, where the groups `xᵢ` collect, for every position along
all other dimensions, the entries of `x` along the dimension `dim`.

# Arguments
- `λ`: Regularization parameter, must be a scalar.
- `dim`: Dimension defining the groups. Can be an `Integer` (1-based index) or a `Symbol` (dimension name,
  requires `x` to be a `NamedDimsArray`).

# Notes
- The ℓ₂,₁ norm enforces a *common support*: entries are driven to zero jointly across `dim` instead of
  independently, which is the right prior when the same anatomy is imaged with several contrasts, echoes,
  diffusion directions or velocity encodings. See Majumdar & Ward, *Joint reconstruction of multiecho MR
  images using correlated sparsity*, Magn Reson Imaging 2011, and Huang et al., *Fast multi-contrast MRI
  reconstruction*, Magn Reson Imaging 2014.
- Applied to a sparsifying transform of the image it becomes the multi-contrast analogue of
  [`L1Wavelet2D`](@ref); applied directly in image domain it is the multi-channel analogue of
  [`L1Image`](@ref).
- [`TotalVariation2D`](@ref) and [`TotalVariation3D`](@ref) use the same mixed norm, with the groups formed
  by the gradient directions instead of an image dimension.
"""
struct JointSparsity{T, D} <: Regularization
    λ::T
    dim::D
    function JointSparsity(λ::T; dim::D) where {T, D}
        @argcheck λ isa Real "JointSparsity requires a scalar λ"
        _check_dim_spec(dim, "dim"; allow_nothing = false)
        return new{T, D}(λ, dim)
    end
end

# The group dimension is found the same way as a temporal dimension: by index or by name.
_group_dim(reg::JointSparsity, dims) = get_time_dim(reg.dim, dims)

"""
	_joint_sparsity_shape(x, group_dim)

Collapse `size(x)` into the `(leading, group, trailing)` triple used to expose the group dimension as the
second dimension of a (possibly batched) matrix. Reshaping to this shape is free: it only regroups the
column-major layout.
"""
function _joint_sparsity_shape(x::AbstractArray, group_dim::Int)
    leading = prod(size(x)[1:(group_dim - 1)]; init = 1)
    trailing = prod(size(x)[(group_dim + 1):end]; init = 1)
    return (leading, size(x, group_dim), trailing)
end

function get_operator(reg::JointSparsity, x::AbstractArray; threaded::Bool = true)
    dims = dims_of(x)
    group_dim = _group_dim(reg, dims)
    leading, group, trailing = _joint_sparsity_shape(x, group_dim)
    ℛ = Reshape(Eye(unname(x)), leading, group, trailing)
    if x isa NamedDimsArray
        ℛ = NamedDimsOp{dimnames(x), (:_, dimnames(x, group_dim), :_)}(ℛ)
    end
    return ℛ
end

function get_affected_dims(reg::JointSparsity, ::Nothing, image_dims)
    return (image_dims[_group_dim(reg, image_dims)],)
end

# The ℓ₂,₁ norm is homogeneous of degree 1, so λ scales linearly (see scale_regularization docstring).
scale_regularization(reg::JointSparsity, factor::Real) = JointSparsity(reg.λ * factor; dim = reg.dim)

bind_dimensions(reg::JointSparsity, image_dims) = JointSparsity(reg.λ; dim = _group_dim(reg, image_dims))

function materialize(reg::JointSparsity, x::Variable{T}; threaded::Bool) where {T}
    R = real(T)
    λ = R(reg.λ)
    op = get_operator(reg, ~x; threaded)
    _, _, trailing = _joint_sparsity_shape(~x, _group_dim(reg, dims_of(~x)))
    # `NormL21(λ, 2)` sums the ℓ₂ norms of the *rows* of a matrix, i.e. of the groups along the second
    # dimension. For more than one trailing slice the prox is applied slice by slice, which is exactly the
    # same separable problem.
    f = if trailing == 1
        NormL21(λ, 2)
    else
        SlicedSeparableSum(NormL21(λ, 2), Tuple((Colon(), Colon(), i) for i in 1:trailing))
    end
    repr = @sprintf "%g ⋅ ‖%s‖₂,₁" λ get_name(x)
    return StructuredOptimization.Term(1, f, op * x, repr)
end
