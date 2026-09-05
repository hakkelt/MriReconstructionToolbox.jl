"""
	L1Contourlet(λ; params=ContourletParams(J=3, L_array=parabolic_levels(3)))

Create a L1 contourlet regularization term for images with parameter `λ`. The regularization
term is given by `λ‖𝒩x‖₁`, where `𝒩` is the Nonsubsampled Contourlet Transform (NSCT) operator
from ContourletOperators.jl, applied to the leading 2D image plane.

Despite the name, this wraps [`NSCTOp`](@ref) rather than the (nearly critically sampled)
[`ContourletOp`](@ref): the NSCT is shift-invariant, so every directional subband shares the
input's spatial size, letting the bands stack into a single `(dim_in..., n_bands)` array (see
[`StackedNSCTOp`](@ref)) instead of the ragged, differently-sized bands `ContourletOp` produces.
That uniform stacking is also what lets images with dimensions beyond the leading 2D plane (e.g.
a time or slice axis) be handled the same way as [`L1Wavelet2D`](@ref): applying the transform
independently to every 2D slice via `AbstractOperators.BatchOp`.

!!! note
    With the default biorthogonal filters, the NSCT is not self-adjoint/orthogonal: `𝒩'` is the
    declared **inverse** transform, not the literal linear-algebra transpose (see the `NSCTOp`
    docstring). `λ` must therefore be a scalar, not a per-voxel array -- the coefficient layout is
    `n_bands` times larger than the image, so there is no natural per-coefficient weight to align
    it with.

# Arguments
- `λ`: Regularization parameter (scalar).
- `params`: (optional) `ContourletParams` controlling pyramid levels/directions. Default
  `ContourletParams(J=3, L_array=parabolic_levels(3))`.
"""
struct L1Contourlet{T <: Real, P} <: Regularization
    λ::T
    params::P
    function L1Contourlet(λ::Real; params = ContourletParams(J = 3, L_array = parabolic_levels(3)))
        return new{typeof(λ), typeof(params)}(λ, params)
    end
end

function get_operator(reg::L1Contourlet, x::AbstractArray{T}; threaded::Bool = true) where {T}
    @argcheck ndims(x) >= 2 "L1Contourlet requires at least 2 dimensions in the input variable"
    ximg = x isa NamedDimsArray ? parent(x) : x
    img_2D_size = size(ximg)[1:2]
    Td = T <: Complex ? ComplexF64 : Float64
    nsct = NSCTOp(Td, reg.params, img_2D_size; threaded)
    op = StackedNSCTOp(T, nsct)
    return ndims(ximg) > 2 ? BatchOp(op, size(ximg)[3:end]; threaded) : op
end

function get_affected_dims(::L1Contourlet, ::Nothing, image_dims)
    return image_dims[1:2]
end

# L1 norm is homogeneous of degree 1, so λ scales linearly (see scale_regularization docstring).
scale_regularization(reg::L1Contourlet, factor::Real) =
    L1Contourlet(reg.λ .* factor; params = reg.params)

function materialize(reg::L1Contourlet, x::Variable{T}; threaded::Bool) where {T}
    𝒩 = get_operator(reg, ~x; threaded)
    λ = real(T)(reg.λ)
    repr = @sprintf "%g ⋅ ‖𝒩%s‖₁" λ get_name(x)
    return StructuredOptimization.Term(1, NormL1(λ), 𝒩 * x, repr)
end
