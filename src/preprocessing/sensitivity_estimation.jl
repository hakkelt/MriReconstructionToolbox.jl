"""
    SensitivityEstimationMethod

Abstract type representing coil sensitivity estimation algorithms.
"""
abstract type SensitivityEstimationMethod end

"""
    SelfCalibrating(; calib_size = 24)

Direct low-resolution sensitivity estimation from the central auto-calibration signal (ACS) region
normalized by root-sum-of-squares (McKenzie et al. 2002, Bydder et al. 2002).
"""
struct SelfCalibrating{T} <: SensitivityEstimationMethod
    calib_size::T
    function SelfCalibrating(; calib_size = 24)
        return new{typeof(calib_size)}(calib_size)
    end
end

"""
    AdaptiveCombine(; kernel_size = 5)

Adaptive coil sensitivity estimation and combination via local correlation matrix eigenanalysis (Walsh et al. 2000).
Needs no separate calibration region and produces smooth, SNR-optimal sensitivity maps.
"""
struct AdaptiveCombine{T} <: SensitivityEstimationMethod
    kernel_size::T
    function AdaptiveCombine(; kernel_size = 5)
        return new{typeof(kernel_size)}(kernel_size)
    end
end

"""
    ESPIRiT(; calib_size = 24, kernel_size = 6, eigenvalue_threshold = 0.02, subspace_threshold = 0.05)

Eigenvalue-based sensitivity estimation (Uecker et al. 2014) from the central calibration subspace.
"""
struct ESPIRiT{T1, T2, T3, T4} <: SensitivityEstimationMethod
    calib_size::T1
    kernel_size::T2
    eigenvalue_threshold::T3
    subspace_threshold::T4
    function ESPIRiT(;
            calib_size = 24,
            kernel_size = 6,
            eigenvalue_threshold = 0.02,
            subspace_threshold = 0.05,
        )
        return new{typeof(calib_size), typeof(kernel_size), typeof(eigenvalue_threshold), typeof(subspace_threshold)}(
            calib_size, kernel_size, eigenvalue_threshold, subspace_threshold
        )
    end
end

"""
    estimate_sensitivities(acq::AcquisitionInfo; method = SelfCalibrating())
    estimate_sensitivities(kspace::AbstractArray; method = SelfCalibrating(), is3D = false, coil_dim = nothing)

Estimates coil sensitivity maps from multi-coil k-space data using the specified method.
When passed an `AcquisitionInfo`, returns a new `AcquisitionInfo` with the `sensitivity_maps` field populated.
"""
function estimate_sensitivities(
        acq::AcquisitionInfo;
        method::SensitivityEstimationMethod = SelfCalibrating(),
    )
    sens = estimate_sensitivities(
        acq.kspace_data;
        method,
        is3D = acq isa CartesianAcquisitionInfo ? acq.is3D : false,
    )
    return AcquisitionInfo(acq; sensitivity_maps = sens)
end

function estimate_sensitivities(
        kspace::AbstractArray;
        method::SensitivityEstimationMethod = SelfCalibrating(),
        is3D::Bool = false,
        coil_dim = nothing,
    )
    c_idx = _resolve_coil_dim(kspace, coil_dim; fallback = is3D ? 4 : 3)

    raw_ksp = unname(kspace)
    sens_arr = _estimate_sensitivities_core(raw_ksp, method, c_idx, is3D)

    if kspace isa NamedDimsArray
        k_dims = dimnames(kspace)
        img_dims = ntuple(length(k_dims)) do i
            d = k_dims[i]
            d == :kx ? :x : (d == :ky ? :y : (d == :kz ? :z : d))
        end
        return NamedDimsArray{img_dims}(sens_arr)
    else
        return sens_arr
    end
end

# Core algorithm implementations

function _estimate_sensitivities_core(
        kspace::AbstractArray{T, N},
        method::SelfCalibrating,
        c_idx::Int,
        is3D::Bool,
    ) where {T, N}
    # Move the coil axis to the trailing position; `inv_perm` restores the layout.
    perm = _trailing_perm(c_idx, N)
    inv_perm = _trailing_inv_perm(c_idx, N)
    ksp_trailing = permutedims(kspace, perm)

    spatial_dims = size(ksp_trailing)[1:(N - 1)]
    Nc = size(ksp_trailing, N)
    cal_size = method.calib_size

    # Windowed calibration region
    cal_ksp = zeros(complex(T), size(ksp_trailing))
    ranges = ntuple(length(spatial_dims)) do d
        K_d = cal_size isa Tuple ? cal_size[d] : min(spatial_dims[d], cal_size)
        c_d = spatial_dims[d] ÷ 2 + 1
        (c_d - K_d ÷ 2):(c_d + K_d ÷ 2 - 1)
    end

    W = ones(real(T), map(length, ranges))
    for d in 1:length(spatial_dims)
        Kd = length(ranges[d])
        wd = 0.5 .- 0.5 .* cos.(2.0 * π .* (0:(Kd - 1)) ./ max(1, Kd - 1))
        w_shape = ntuple(i -> i == d ? Kd : 1, length(spatial_dims))
        W .*= reshape(wd, w_shape)
    end

    for c in 1:Nc
        cal_ksp[ranges..., c] = ksp_trailing[ranges..., c] .* W
    end

    # IFFT from centered k-space to uncentered image space
    lowres_img = zeros(complex(T), size(ksp_trailing))
    f_dims = ntuple(identity, length(spatial_dims))
    ℱ = _axis_dft_op(zeros(complex(T), spatial_dims...), f_dims; kspace_shift = true)
    for c in 1:Nc
        cal_c = selectdim(cal_ksp, N, c)
        selectdim(lowres_img, N, c) .= (ℱ' * collect(cal_c)) .* sqrt(prod(spatial_dims))
    end

    rss = sqrt.(sum(abs2.(lowres_img), dims = N))
    sens_trailing = lowres_img ./ (rss .+ eps(real(T)))
    return permutedims(sens_trailing, inv_perm)
end

function _estimate_sensitivities_core(
        kspace::AbstractArray{T, N},
        method::AdaptiveCombine,
        c_idx::Int,
        is3D::Bool,
    ) where {T, N}
    # Move the coil axis to the trailing position; `inv_perm` restores the layout.
    perm = _trailing_perm(c_idx, N)
    inv_perm = _trailing_inv_perm(c_idx, N)
    ksp_trailing = permutedims(kspace, perm)

    spatial_dims = size(ksp_trailing)[1:(N - 1)]
    Nc = size(ksp_trailing, N)
    f_dims = ntuple(identity, length(spatial_dims))

    coil_imgs = zeros(complex(T), size(ksp_trailing))
    ℱ = _axis_dft_op(zeros(complex(T), spatial_dims...), f_dims; kspace_shift = true)
    for c in 1:Nc
        ksp_c = selectdim(ksp_trailing, N, c)
        selectdim(coil_imgs, N, c) .= (ℱ' * collect(ksp_c)) .* sqrt(prod(spatial_dims))
    end

    K = method.kernel_size
    pad = K ÷ 2
    sens_trailing = zeros(complex(T), size(ksp_trailing))

    for idx in CartesianIndices(spatial_dims)
        patch_ranges = ntuple(length(spatial_dims)) do d
            max(1, idx[d] - pad):min(spatial_dims[d], idx[d] + pad)
        end
        patch = coil_imgs[patch_ranges..., :]
        patch_mat = reshape(permutedims(patch, (N, 1:(N - 1)...)), Nc, :)
        R = patch_mat * patch_mat'

        F = eigen(Hermitian(R))
        v = F.vectors[:, end]
        if abs(v[1]) > 1.0e-6
            v .*= cis(-angle(v[1]))
        end
        sens_trailing[idx, :] = v
    end

    return permutedims(sens_trailing, inv_perm)
end

function _estimate_sensitivities_core(
        kspace::AbstractArray{T, N},
        method::ESPIRiT,
        c_idx::Int,
        is3D::Bool,
    ) where {T, N}
    # Move the coil axis to the trailing position; `inv_perm` restores the layout.
    perm = _trailing_perm(c_idx, N)
    inv_perm = _trailing_inv_perm(c_idx, N)
    ksp_trailing = permutedims(kspace, perm)

    spatial_dims = size(ksp_trailing)[1:(N - 1)]
    Nc = size(ksp_trailing, N)
    cal_size = method.calib_size
    K = method.kernel_size

    ranges = ntuple(length(spatial_dims)) do d
        K_d = cal_size isa Tuple ? cal_size[d] : min(spatial_dims[d], cal_size)
        c_d = spatial_dims[d] ÷ 2 + 1
        (c_d - K_d ÷ 2):(c_d + K_d ÷ 2 - 1)
    end
    calib = ksp_trailing[ranges..., :]

    cal_dims = map(length, ranges)
    kernel_dims = ntuple(i -> K isa Tuple ? K[i] : K, length(spatial_dims))
    num_patches_per_dim = ntuple(i -> cal_dims[i] - kernel_dims[i] + 1, length(spatial_dims))
    num_patches = prod(num_patches_per_dim)
    patch_dim = prod(kernel_dims) * Nc

    C = zeros(complex(T), num_patches, patch_dim)
    p_idx = 1
    for p_offset in CartesianIndices(num_patches_per_dim)
        patch_box = ntuple(i -> (p_offset[i]):(p_offset[i] + kernel_dims[i] - 1), length(spatial_dims))
        patch = calib[patch_box..., :]
        C[p_idx, :] = reshape(patch, :)
        p_idx += 1
    end

    F_svd = svd(C)
    s_thresh = F_svd.S[1] * method.subspace_threshold
    num_vecs = count(s -> s >= s_thresh, F_svd.S)
    num_vecs = max(1, num_vecs)
    V_sub = F_svd.V[:, 1:num_vecs]

    f_dims = ntuple(identity, length(spatial_dims))
    ℱ = _axis_dft_op(zeros(complex(T), spatial_dims...), f_dims)
    V_img = zeros(complex(T), spatial_dims..., Nc, num_vecs)
    for v_idx in 1:num_vecs
        kernel_arr = reshape(V_sub[:, v_idx], kernel_dims..., Nc)
        flipped = reverse(conj(kernel_arr), dims = f_dims)
        for c in 1:Nc
            padded = zeros(complex(T), spatial_dims...)
            init_ranges = ntuple(i -> 1:kernel_dims[i], length(spatial_dims))
            padded[init_ranges...] = flipped[init_ranges..., c]
            shift_amounts = ntuple(i -> -(kernel_dims[i] ÷ 2), length(spatial_dims))
            padded = circshift(padded, shift_amounts)
            V_img[fill(:, length(spatial_dims))..., c, v_idx] .= (ℱ * padded) .* sqrt(prod(spatial_dims))
        end
    end

    sens_trailing = zeros(complex(T), size(ksp_trailing))
    for idx in CartesianIndices(spatial_dims)
        V_r = reshape(V_img[idx, :, :], Nc, num_vecs)
        W = V_r * V_r'
        F_eig = eigen(Hermitian(W))
        val = F_eig.values[end]
        vec_max = F_eig.vectors[:, end]
        if val >= method.eigenvalue_threshold
            sens_trailing[idx, :] = vec_max
        end
    end

    return permutedims(sens_trailing, inv_perm)
end
