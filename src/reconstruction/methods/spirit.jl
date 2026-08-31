"""
    SPIRiTConsistency{T} <: Regularization

Self-consistency k-space convolution regularization term ``\\tfrac{\\lambda}{2}\\|(I - G)\\,k\\|_2^2``
for SPIRiT reconstruction (Lustig & Pauly 2010, MRM 64:457-471), where ``k`` is the multi-channel
k-space and ``G`` the calibrated coil-mixing convolution.

!!! warning
    This term operates on a **k-space** optimization variable and requires
    `IterativeReconstruction(...; domain = KSpaceDomain())`, whose dispatch machinery is not wired
    up yet. `materialize` therefore throws. For a working SPIRiT reconstruction today use the
    direct method [`SPIRiT`](@ref).
"""
struct SPIRiTConsistency{T} <: Regularization
    kernel::T
    λ::Float64
    function SPIRiTConsistency(kernel; λ = 1.0)
        return new{typeof(kernel)}(kernel, Float64(λ))
    end
end

function materialize(::SPIRiTConsistency, ::Variable; threaded::Bool)
    throw(ArgumentError(
        "SPIRiTConsistency requires a k-space optimization variable " *
            "(IterativeReconstruction(...; domain = KSpaceDomain())), which is not yet supported. " *
            "Use the direct `SPIRiT()` method instead.",
    ))
end

get_affected_dims(::SPIRiTConsistency, ::Nothing, image_dims) = image_dims
get_affected_dims(::SPIRiTConsistency, ::AcquisitionInfo, image_dims) = image_dims

"""
    SPIRiT{C <: CoilCombination} <: AbstractDirectMethod

Iterative Self-consistent Parallel Imaging Reconstruction (Lustig & Pauly 2010).
Calibrates local multi-channel k-space convolution kernels from central ACS data and solves
for missing k-space samples using self-consistency iterations.

# Fields
- `kernel_size`: Size of convolution kernel `(Kx, Ky)` (default: `(5, 5)`).
- `calib_size`: ACS calibration region size (default: `(24, 24)`).
- `maxit`: Number of iterations (default: `25`).
- `λ`: Regularization parameter on self-consistency (default: `1.0`).
- `coil_combination`: Method for combining reconstructed multi-coil channels (`RootSumSquares()` or `AdjointSensitivity()`).
"""
struct SPIRiT{C <: CoilCombination} <: AbstractDirectMethod
    kernel_size::Tuple{Int, Int}
    calib_size::Tuple{Int, Int}
    maxit::Int
    λ::Float64
    coil_combination::C
    function SPIRiT(;
            kernel_size = (5, 5),
            calib_size = (24, 24),
            maxit = 25,
            λ = 1.0,
            coil_combination::CoilCombination = RootSumSquares(),
        )
        return new{typeof(coil_combination)}(
            (kernel_size[1], kernel_size[2]),
            (calib_size[1], calib_size[2]),
            maxit,
            Float64(λ),
            coil_combination,
        )
    end
end

function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::SPIRiT)
    @argcheck !isnothing(acq.subsampling) "SPIRiT reconstruction requires an undersampled Cartesian acquisition"

    raw_ksp = _get_full_kspace(acq)
    Nx, Ny = size(raw_ksp, 1), size(raw_ksp, 2)
    Nc = size(raw_ksp, 3)
    T = complex(real(eltype(raw_ksp)))
    raw_ksp = convert(Array{T}, raw_ksp)

    mask = to_displayable_mask(acq.subsampling, (Nx, Ny))
    mask_3d = reshape(mask, Nx, Ny, 1)

    cal_kx, cal_ky = min(Nx, method.calib_size[1]), min(Ny, method.calib_size[2])
    cx, cy = Nx ÷ 2 + 1, Ny ÷ 2 + 1
    cal_range_x = (cx - cal_kx ÷ 2):(cx + cal_kx ÷ 2 - 1)
    cal_range_y = (cy - cal_ky ÷ 2):(cy + cal_ky ÷ 2 - 1)

    calib = raw_ksp[cal_range_x, cal_range_y, :]

    Kx, Ky = method.kernel_size
    pad_x, pad_y = Kx ÷ 2, Ky ÷ 2

    num_bx = cal_kx - Kx + 1
    num_by = cal_ky - Ky + 1
    num_b = num_bx * num_by

    A_mat = zeros(T, num_b, Kx * Ky * Nc)
    b_idx = 1
    for bx in 1:num_bx, by in 1:num_by
        patch = calib[bx:(bx + Kx - 1), by:(by + Ky - 1), :]
        A_mat[b_idx, :] = reshape(patch, :)
        b_idx += 1
    end

    center_idx = pad_x + 1 + pad_y * Kx
    G_kernels = zeros(T, Kx, Ky, Nc, Nc)

    for target_c in 1:Nc
        target_feat_idx = center_idx + (target_c - 1) * Kx * Ky
        src_cols = [i for i in 1:(Kx * Ky * Nc) if i != target_feat_idx]
        y_tgt = A_mat[:, target_feat_idx]
        X_src = A_mat[:, src_cols]
        w = X_src \ y_tgt

        full_w = zeros(T, Kx * Ky * Nc)
        full_w[src_cols] = w
        full_w_3d = reshape(full_w, Kx, Ky, Nc)
        for src_c in 1:Nc
            G_kernels[:, :, src_c, target_c] = full_w_3d[:, :, src_c]
        end
    end

    # Iterative projection onto data consistency + G convolution
    # x_{k+1} = P y + (I - P) G(x_k)
    x_ksp = copy(raw_ksp)

    # Pad kernels to full grid for fast FFT convolution
    G_fft = zeros(T, Nx, Ny, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        padded = zeros(T, Nx, Ny)
        padded[1:Kx, 1:Ky] = G_kernels[:, :, j, i]
        padded = circshift(padded, (-pad_x, -pad_y))
        G_fft[:, :, j, i] = fft(padded)
    end

    for _ in 1:(method.maxit)
        # Apply G via FFT convolution across coils
        Gx = zeros(T, Nx, Ny, Nc)
        for j in 1:Nc
            x_j_fft = fft(x_ksp[:, :, j])
            for i in 1:Nc
                Gx[:, :, i] .+= ifft(x_j_fft .* G_fft[:, :, j, i])
            end
        end

        # Enforce consistency: keep acquired data, update unacquired with Gx
        x_ksp = ifelse.(mask_3d, raw_ksp, Gx)
    end

    # Transform completed k-space to image space
    f_dims = (1, 2)
    coil_imgs = ifft(ifftshift(x_ksp, f_dims), f_dims) .* sqrt(Nx * Ny)

    img_out = if method.coil_combination isa AdjointSensitivity && !isnothing(acq.sensitivity_maps)
        sens = unname(acq.sensitivity_maps)
        sum(coil_imgs .* conj.(sens); dims = 3)
    else
        sqrt.(sum(abs2, coil_imgs; dims = 3))
    end

    # Drop the reduced coil axis (singleton at position 3) instead of truncating with
    # `reshape`, which would discard any trailing batch/time dimension.
    img_out = dropdims(img_out; dims = 3)

    if acq.kspace_data isa NamedDimsArray
        out_dims = filter(!=(:coil), get_image_dims(acq))
        return NamedDimsArray{out_dims}(img_out)
    else
        return img_out
    end
end
