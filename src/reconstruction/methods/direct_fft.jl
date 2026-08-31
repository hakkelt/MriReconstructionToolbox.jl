"""
    _direct_ifft(acq::CartesianAcquisitionInfo, k::AbstractArray; dims = (1, 2))

Computes the inverse Fourier transform from k-space to image space, respecting `shifted_kspace_dims`
and `shifted_image_dims` from `acq`.
"""
function _direct_ifft(acq::CartesianAcquisitionInfo, k::AbstractArray; dims = (1, 2))
    is3D = acq.is3D
    sK = _normalize_shifted_dims(acq.shifted_kspace_dims, is3D, acq.kspace_data, "shifted_kspace_dims", (:kx, :ky, :kz))
    sI = _normalize_shifted_dims(acq.shifted_image_dims, is3D, acq.kspace_data, "shifted_image_dims", (:x, :y, :z))

    k_to_shift = tuple([d for d in dims if d ∉ sK]...)
    img_to_shift = tuple([d for d in dims if d ∈ sI]...)

    res = if !isempty(k_to_shift)
        ifftshift(k, k_to_shift)
    else
        k
    end

    res = ifft(res, dims)

    if !isempty(img_to_shift)
        res = ifftshift(res, img_to_shift)
    end
    return res
end

"""
    _direct_fft(acq::CartesianAcquisitionInfo, x::AbstractArray; dims = (1, 2))

Computes the forward Fourier transform from image space to k-space, respecting `shifted_kspace_dims`
and `shifted_image_dims` from `acq`.
"""
function _direct_fft(acq::CartesianAcquisitionInfo, x::AbstractArray; dims = (1, 2))
    is3D = acq.is3D
    sK = _normalize_shifted_dims(acq.shifted_kspace_dims, is3D, acq.kspace_data, "shifted_kspace_dims", (:kx, :ky, :kz))
    sI = _normalize_shifted_dims(acq.shifted_image_dims, is3D, acq.kspace_data, "shifted_image_dims", (:x, :y, :z))

    k_to_shift = tuple([d for d in dims if d ∉ sK]...)
    img_to_shift = tuple([d for d in dims if d ∈ sI]...)

    res = if !isempty(img_to_shift)
        fftshift(x, img_to_shift)
    else
        x
    end

    res = fft(res, dims)

    if !isempty(k_to_shift)
        res = fftshift(res, k_to_shift)
    end
    return res
end
