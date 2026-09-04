"""
    estimate_noise_covariance(noise_data::AbstractArray; coil_dim = nothing)

Estimate the inter-coil noise covariance matrix ``\\Psi \\in \\mathbb{C}^{N_c \\times N_c}`` from a pure-noise
k-space or calibration acquisition.

# Arguments
- `noise_data`: Multi-coil noise acquisition array.
- `coil_dim`: (optional) Dimension index or name corresponding to receiver coils (defaults to `:coil` or dim 3 for 2D/dim 4 for 3D).
"""
function estimate_noise_covariance(noise_data::AbstractArray; coil_dim = nothing)
    c_idx = _resolve_coil_dim(noise_data, coil_dim)

    Nc = size(noise_data, c_idx)
    # Permute coil dimension to first dimension and flatten remaining dimensions
    perm = _front_perm(c_idx, ndims(noise_data))
    perm_data = permutedims(unname(noise_data), perm)
    flat_noise = reshape(perm_data, Nc, :)
    Nsamples = size(flat_noise, 2)
    @argcheck Nsamples > 0 "Noise acquisition has no samples"

    Ψ = (flat_noise * flat_noise') ./ Nsamples
    return Ψ
end

"""
    prewhiten(acq::AcquisitionInfo, Ψ::AbstractMatrix; coil_dim = nothing)
    prewhiten(data::AbstractArray, Ψ::AbstractMatrix; coil_dim = nothing)

Prewhitens multi-coil k-space data (and sensitivity maps if present) using the noise covariance matrix ``\\Psi``.
Applies ``L^{-1}`` where ``\\Psi = L L^*`` is the Cholesky factorization of the noise covariance.
"""
function prewhiten(acq::AcquisitionInfo, Ψ::AbstractMatrix; coil_dim = nothing)
    whitened_ksp = prewhiten(acq.kspace_data, Ψ; coil_dim)
    whitened_sens = isnothing(acq.sensitivity_maps) ? nothing : prewhiten(acq.sensitivity_maps, Ψ; coil_dim)
    return AcquisitionInfo(acq; kspace_data = whitened_ksp, sensitivity_maps = whitened_sens)
end

function prewhiten(data::AbstractArray, Ψ::AbstractMatrix; coil_dim = nothing)
    c_idx = _resolve_coil_dim(data, coil_dim)

    Nc = size(data, c_idx)
    @argcheck size(Ψ, 1) == Nc && size(Ψ, 2) == Nc "Noise covariance size $(size(Ψ)) does not match coil count $Nc"

    L = cholesky(Hermitian(Ψ)).L
    orig_dims = size(data)
    perm = _front_perm(c_idx, ndims(data))
    inv_perm = _front_inv_perm(c_idx, ndims(data))

    perm_data = permutedims(unname(data), perm)
    flat_data = reshape(perm_data, Nc, :)
    whitened_flat = L \ flat_data
    whitened_perm = reshape(whitened_flat, size(perm_data))
    whitened_data = permutedims(whitened_perm, inv_perm)

    if data isa NamedDimsArray
        return NamedDimsArray{dimnames(data)}(whitened_data)
    else
        return whitened_data
    end
end
