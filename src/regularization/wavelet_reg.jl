"""
	L1Wavelet2D(λ; wavelet=WT.db2, levels=2)

Create a L1 wavelet regularization term for 2D images with parameter `λ`. The regularization term is given by `λ‖𝒲x‖₁`,
where `𝒲` is the 2D wavelet transform operator. The `wavelet` and `levels` parameters control the type of wavelet and the
number of decomposition levels used in the transform.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array of the same size as `x`.
- `wavelet`: (optional) Type of wavelet to use, specified as a `WT.Wavelet`. Default is `WT.db2` (Daubechies 2).
- `levels`: (optional) Number of decomposition levels in the wavelet transform. Default is `2`.
"""
struct L1Wavelet2D{T, W} <: Regularization
    λ::T
    wavelet::W
    levels::Int
    function L1Wavelet2D(λ; wavelet = WT.db2, levels = 2)
        return new{typeof(λ), typeof(wavelet)}(λ, wavelet, levels)
    end
end

function get_operator(reg::L1Wavelet2D, x::AbstractArray{T}; threaded::Bool = true) where {T}
    @argcheck ndims(x) >= 2 "L1Wavelet2D requires at least 2 dimensions in the input variable"
    img_2D_size = size(x)[1:2]
    divisions = 2^reg.levels
    if img_2D_size[1] % divisions != 0 || img_2D_size[2] % divisions != 0
        padded_size = (
            ceil(Int, img_2D_size[1] / divisions) * divisions,
            ceil(Int, img_2D_size[2] / divisions) * divisions,
        )
        𝒲 = WaveletOp(T, wavelet(reg.wavelet), padded_size, reg.levels)
        x_view = view(x, :, :, fill(1, ndims(x) - 2)...)
        𝒵 = ZeroPad(x_view, padded_size .- img_2D_size)
        𝒲 = 𝒲 * 𝒵
    else
        𝒲 = WaveletOp(T, wavelet(reg.wavelet), img_2D_size, reg.levels)
    end
    if ndims(x) > 2
        𝒲 = BatchOp(𝒲, size(x)[3:end]; threaded)
    end
    if x isa NamedDimsArray
        𝒲 = NamedDimsOp{dimnames(x), dimnames(x)}(𝒲)
    end
    return 𝒲
end

function get_affected_dims(::L1Wavelet2D, acq_info::AcquisitionInfo, image_dims)
    return image_dims[1:2]
end

"""
	L1Wavelet3D(λ; wavelet=WT.db2, levels=2)

Create a L1 wavelet regularization term for 3D images with parameter `λ`. The regularization term is given by `λ‖𝒲x‖₁`,
where `𝒲` is the 3D wavelet transform operator. The `wavelet` and `levels` parameters control the type of wavelet and the
number of decomposition levels used in the transform.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array of the same size as `x`.
- `wavelet`: (optional) Type of wavelet to use, specified as a `WT.Wavelet`. Default is `WT.db2` (Daubechies 2).
- `levels`: (optional) Number of decomposition levels in the wavelet transform. Default is `2`.
"""
struct L1Wavelet3D{T, W} <: Regularization
    λ::T
    wavelet::W
    levels::Int
    function L1Wavelet3D(λ; wavelet = WT.db2, levels = 2)
        return new{typeof(λ), typeof(wavelet)}(λ, wavelet, levels)
    end
end

function get_operator(reg::L1Wavelet3D, x::AbstractArray{T}; threaded::Bool = true) where {T}
    @argcheck ndims(x) >= 3 "L1Wavelet3D requires at least 3 dimensions in the input variable"
    img_3D_size = size(x)[1:3]
    divisions = 2^reg.levels
    if img_3D_size[1] % divisions != 0 ||
            img_3D_size[2] % divisions != 0 ||
            img_3D_size[3] % divisions != 0
        padded_size = (
            ceil(Int, img_3D_size[1] / divisions) * divisions,
            ceil(Int, img_3D_size[2] / divisions) * divisions,
            ceil(Int, img_3D_size[3] / divisions) * divisions,
        )
        𝒲 = WaveletOp(T, wavelet(reg.wavelet), padded_size, reg.levels)
        x_view = view(x, :, :, :, fill(1, ndims(x) - 3)...)
        𝒵 = ZeroPad(x_view, padded_size .- img_3D_size)
        𝒲 = 𝒲 * 𝒵
    else
        𝒲 = WaveletOp(T, wavelet(reg.wavelet), img_3D_size, reg.levels)
    end
    if ndims(x) > 3
        𝒲 = BatchOp(𝒲, size(x)[4:end]; threaded)
    end
    if x isa NamedDimsArray
        𝒲 = NamedDimsOp{dimnames(x), dimnames(x)}(𝒲)
    end
    return 𝒲
end

function get_affected_dims(::L1Wavelet3D, acq_info::AcquisitionInfo, image_dims)
    return image_dims[1:3]
end

function materialize(
        reg::Union{L1Wavelet2D, L1Wavelet3D}, x::Variable{T}; threaded::Bool
    ) where {T}
    𝒲 = get_operator(reg, ~x; threaded)
    λ = real(T)(reg.λ)
    repr = if reg.λ isa AbstractArray
        "‖Γ .* 𝒲$(get_name(x))‖₁"
    else
        @sprintf "%g ⋅ ‖𝒲%s‖₁" λ get_name(x)
    end
    return StructuredOptimization.Term(1, NormL1(λ), 𝒲 * x, repr)
end
