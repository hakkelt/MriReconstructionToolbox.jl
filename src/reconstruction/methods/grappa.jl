"""
    GRAPPA{C <: CoilCombination} <: AbstractDirectMethod

Generalized Autocalibrating Partially Parallel Acquisitions (Griswold et al. 2002, MRM 47:1202-1210).
A direct parallel imaging method that synthesizes missing k-space lines via localized multi-channel convolution
calibrated from fully sampled central autocalibration signal (ACS) lines.

# Fields
- `kernel_size`: Convolution kernel size `(Kx, Ky)` (default: `(4, 3)`).
- `calib_size`: ACS calibration region size (default: `(24, 24)`).
- `coil_combination`: Method for combining synthesized multi-coil channels (`RootSumSquares()` or `AdjointSensitivity()`).
"""
struct GRAPPA{C <: CoilCombination} <: AbstractDirectMethod
    kernel_size::Tuple{Int, Int}
    calib_size::Tuple{Int, Int}
    coil_combination::C
    function GRAPPA(;
            kernel_size = (4, 3),
            calib_size = (24, 24),
            coil_combination::CoilCombination = RootSumSquares(),
        )
        return new{typeof(coil_combination)}(
            (kernel_size[1], kernel_size[2]),
            (calib_size[1], calib_size[2]),
            coil_combination,
        )
    end
end

function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::GRAPPA)
    @argcheck !isnothing(acq.subsampling) "GRAPPA reconstruction requires an undersampled Cartesian acquisition"
    @argcheck !isnothing(acq.sensitivity_maps) || method.coil_combination isa RootSumSquares "GRAPPA requires sensitivity maps when using AdjointSensitivity coil combination"

    raw_ksp = _get_full_kspace(acq)
    Nx, Ny = size(raw_ksp, 1), size(raw_ksp, 2)
    Nc = size(raw_ksp, 3)

    mask = to_displayable_mask(acq.subsampling, (Nx, Ny))
    cal_kx, cal_ky = min(Nx, method.calib_size[1]), min(Ny, method.calib_size[2])

    cx, cy = Nx ÷ 2 + 1, Ny ÷ 2 + 1
    cal_range_x = (cx - cal_kx ÷ 2):(cx + cal_kx ÷ 2 - 1)
    cal_range_y = (cy - cal_ky ÷ 2):(cy + cal_ky ÷ 2 - 1)

    calib = raw_ksp[cal_range_x, cal_range_y, :]

    # Detect undersampling factor R along ky
    acquired_lines = findall(vec(any(mask; dims = 1)))
    diffs = diff(acquired_lines)
    R_acc = maximum(diffs)
    if R_acc <= 1
        R_acc = 2
    end

    Kx, Ky = method.kernel_size
    pad_x = Kx ÷ 2
    Ky_src = 2
    ky_span = (Ky_src - 1) * R_acc + 1

    num_bx = cal_kx - Kx + 1
    num_by = cal_ky - ky_span + 1
    num_b = num_bx * num_by
    n_src_feats = Kx * Ky_src * Nc

    S_mat = zeros(ComplexF32, num_b, n_src_feats)
    T_mat = zeros(ComplexF32, num_b, Nc)

    b_idx = 1
    for bx in 1:num_bx, by in 1:num_by
        src_patch = zeros(ComplexF32, Kx, Ky_src, Nc)
        for (i_y, y_off) in enumerate(0:R_acc:(ky_span - 1))
            src_patch[:, i_y, :] = calib[bx:(bx + Kx - 1), by + y_off, :]
        end
        S_mat[b_idx, :] = reshape(src_patch, :)
        T_mat[b_idx, :] = calib[bx + pad_x, by + 1, :]
        b_idx += 1
    end

    W_grappa = S_mat \ T_mat

    # Synthesize missing lines
    ksp_recon = copy(raw_ksp)
    for ky in 1:Ny
        if !mask[1, ky] && ky ∉ cal_range_y
            for kx in 1:Nx
                src_feat = zeros(ComplexF32, Kx, Ky_src, Nc)
                for (i_y, y_off) in enumerate((-1, 1))
                    ky_src = ky + y_off
                    if 1 <= ky_src <= Ny
                        for (i_x, x_off) in enumerate((-pad_x):pad_x)
                            kx_src = mod1(kx + x_off, Nx)
                            src_feat[i_x, i_y, :] = raw_ksp[kx_src, ky_src, :]
                        end
                    end
                end
                ksp_recon[kx, ky, :] = reshape(src_feat, 1, :) * W_grappa
            end
        end
    end

    # Transform completed k-space to image space
    f_dims = (1, 2)
    coil_imgs = ifft(ifftshift(ksp_recon, f_dims), f_dims) .* sqrt(Nx * Ny)

    img_out = if method.coil_combination isa AdjointSensitivity && !isnothing(acq.sensitivity_maps)
        sens = unname(acq.sensitivity_maps)
        sum(coil_imgs .* conj.(sens); dims = 3)
    else
        sqrt.(sum(abs2, coil_imgs; dims = 3))
    end

    if acq.kspace_data isa NamedDimsArray
        img_dims = get_image_dims(acq)
        out_dims = filter(!=(:coil), img_dims)
        out_arr = reshape(img_out, ntuple(i -> size(img_out, i), length(out_dims)))
        return NamedDimsArray{out_dims}(out_arr)
    else
        return img_out
    end
end
