"""
    _cartesian_fourier_op(acq::CartesianAcquisitionInfo, template::AbstractArray; threaded = true, fast_planning = true)

Bare Cartesian Fourier operator (image → k-space) for `acq`, planned for an array shaped like
`template`. Honors `shifted_kspace_dims` / `shifted_image_dims`; `op * x` is the forward transform,
`op' * k` the inverse. Reuses [`get_fourier_operator`](@ref) — no `fft`/`fftshift` calls here.

`template` must be a plain array (not a `NamedDimsArray`) so the operator stays unwrapped and its
outputs stay plain; the direct methods re-attach dimension names themselves.
"""
function _cartesian_fourier_op(
        acq::CartesianAcquisitionInfo, template::AbstractArray; threaded::Bool = true, fast_planning::Bool = true
    )
    is3D = acq.is3D
    sk = _normalize_shifted_dims(acq.shifted_kspace_dims, is3D, acq.kspace_data, "shifted_kspace_dims", (:kx, :ky, :kz))
    si = _normalize_shifted_dims(acq.shifted_image_dims, is3D, acq.kspace_data, "shifted_image_dims", (:x, :y, :z))
    return get_fourier_operator(template, is3D; shifted_kspace_dims = sk, shifted_image_dims = si, threaded, fast_planning)
end

# Thin direction-named aliases over a `_cartesian_fourier_op` result.
_direct_fft(op, x::AbstractArray) = op * x       # image → k-space
_direct_ifft(op, k::AbstractArray) = op' * k     # k-space → image

"""
    _kspace_to_image(ksp, coil_combine, sens, acq::CartesianAcquisitionInfo)

Transform a completed multi-coil Cartesian k-space `ksp` (coils on dim 3) to an image, applying
`coil_combine` (`AdjointSensitivity` with `sens`, `RootSumSquares`, or `NoCoilCombination`) and
dropping the coil axis unless combination is skipped. Shared tail for the direct methods
(GRAPPA, SPIRiT, ...).
"""
function _kspace_to_image(
        ksp::AbstractArray,
        coil_combine::CoilCombination,
        sens::Union{Nothing, AbstractArray},
        acq::CartesianAcquisitionInfo,
    )
    kplain = unname(ksp)
    op = _cartesian_fourier_op(acq, kplain)
    # `op` is `BACKWARD`-normalized (forward = plain fft, adjoint = fully N-normalized ifft), so
    # `op' * kplain` already matches DirectReconstruction's `𝒜' * kspace_data` convention with no
    # further scaling needed.
    coil_imgs = op' * kplain

    c_dim = 3
    img_out = if coil_combine isa AdjointSensitivity
        @argcheck !isnothing(sens) "AdjointSensitivity coil combination requires sensitivity maps."
        sum(coil_imgs .* conj.(unname(sens)); dims = c_dim)
    elseif coil_combine isa RootSumSquares
        sqrt.(sum(abs2, coil_imgs; dims = c_dim))
    elseif coil_combine isa NoCoilCombination
        coil_imgs
    else
        throw(ArgumentError("Unsupported coil combination: $(typeof(coil_combine))"))
    end

    combined = !(coil_combine isa NoCoilCombination)
    combined && (img_out = dropdims(img_out; dims = c_dim))

    if acq.kspace_data isa NamedDimsArray
        out_d = combined ? filter(!=(:coil), get_image_dims(acq)) : get_image_dims(acq)
        return NamedDimsArray{out_d}(img_out)
    else
        return img_out
    end
end
