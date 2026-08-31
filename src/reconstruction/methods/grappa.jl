"""
    GRAPPA{C <: CoilCombination} <: AbstractDirectMethod

Generalized Autocalibrating Partially Parallel Acquisitions (Griswold et al. 2002, MRM 47:1202-1210).
A direct parallel imaging method that synthesizes missing k-space lines via localized multi-channel convolution
calibrated from fully sampled central autocalibration signal (ACS) lines. Handles arbitrary integer
undersampling factors `R` along `ky`: the stride is detected from the sampling mask and a separate kernel
is fitted for each of the `R - 1` missing-line positions.

# Fields
- `kernel_size`: `(Kx, Ky_src)` — number of `kx` taps and number of source `ky` lines (spaced `R`
  apart) used per fitted kernel (default: `(4, 3)`).
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
    T = complex(real(eltype(raw_ksp)))

    mask = to_displayable_mask(acq.subsampling, (Nx, Ny))
    acquired = vec(any(mask; dims = 1))

    cal_kx, cal_ky = min(Nx, method.calib_size[1]), min(Ny, method.calib_size[2])
    cx, cy = Nx ÷ 2 + 1, Ny ÷ 2 + 1
    cal_range_x = (cx - cal_kx ÷ 2):(cx + cal_kx ÷ 2 - 1)
    cal_range_y = (cy - cal_ky ÷ 2):(cy + cal_ky ÷ 2 - 1)
    calib = raw_ksp[cal_range_x, cal_range_y, :]

    # Detect the regular undersampling stride R along ky: the smallest gap between
    # consecutive acquired lines that exceeds one (gaps of one come from the ACS block).
    acquired_lines = findall(acquired)
    gaps = filter(>(1), diff(acquired_lines))
    R_acc = isempty(gaps) ? 1 : minimum(gaps)

    if R_acc == 1
        # Nothing missing (fully sampled) - fall through to the transform with raw k-space.
        ksp_recon = copy(raw_ksp)
    else
        Kx = method.kernel_size[1]
        Ky_src = max(2, method.kernel_size[2])
        xc = Kx ÷ 2                       # 0-based index of the target kx tap
        jc = (Ky_src - 1) ÷ 2             # source-block row that sits just below the target
        x_taps = (0:(Kx - 1)) .- xc       # kx source offsets relative to the target column
        # Source ky rows relative to the acquired line `ky0` just below a target at `ky0 + t`.
        src_row_offsets(t) = ((0:(Ky_src - 1)) .- jc) .* R_acc

        n_src_feats = Kx * Ky_src * Nc
        # One weight set per missing-line offset t = 1 .. R-1.
        W_by_offset = Vector{Matrix{T}}(undef, R_acc - 1)
        for t in 1:(R_acc - 1)
            rows = src_row_offsets(t)
            by_lo = 1 - minimum(rows)
            by_hi = cal_ky - maximum(rows)
            # target row must also be inside the ACS: by + t <= cal_ky
            by_hi = min(by_hi, cal_ky - t)
            by_range = by_lo:by_hi
            num_b = (cal_kx - Kx + 1) * length(by_range)
            @argcheck num_b > n_src_feats ÷ Nc "GRAPPA calibration region is too small for kernel_size=$(method.kernel_size) at R=$R_acc"
            S_mat = zeros(T, num_b, n_src_feats)
            T_mat = zeros(T, num_b, Nc)
            b = 1
            for bx in 1:(cal_kx - Kx + 1), by in by_range
                patch = zeros(T, Kx, Ky_src, Nc)
                for (jy, ro) in enumerate(rows)
                    patch[:, jy, :] = calib[bx:(bx + Kx - 1), by + ro, :]
                end
                S_mat[b, :] = reshape(patch, :)
                T_mat[b, :] = calib[bx + xc, by + t, :]
                b += 1
            end
            W_by_offset[t] = S_mat \ T_mat
        end

        # Synthesize every missing line from its two surrounding acquired lines.
        ksp_recon = copy(raw_ksp)
        for ky in 1:Ny
            acquired[ky] && continue
            ky0 = ky
            while ky0 >= 1 && !acquired[ky0]
                ky0 -= 1
            end
            t = ky - ky0
            (ky0 < 1 || t < 1 || t > R_acc - 1) && continue
            rows = ky0 .+ src_row_offsets(t)
            all(r -> 1 <= r <= Ny, rows) || continue
            W = W_by_offset[t]
            for kx in 1:Nx
                patch = zeros(T, Kx, Ky_src, Nc)
                for (jy, r) in enumerate(rows), (ix, xo) in enumerate(x_taps)
                    patch[ix, jy, :] = raw_ksp[mod1(kx + xo, Nx), r, :]
                end
                ksp_recon[kx, ky, :] = reshape(patch, 1, :) * W
            end
        end
    end

    # Transform completed k-space to image space
    f_dims = (1, 2)
    coil_imgs = _direct_ifft(acq, ksp_recon; dims = f_dims) .* sqrt(Nx * Ny)

    img_out = if method.coil_combination isa AdjointSensitivity && !isnothing(acq.sensitivity_maps)
        sens = unname(acq.sensitivity_maps)
        sum(coil_imgs .* conj.(sens); dims = 3)
    else
        sqrt.(sum(abs2, coil_imgs; dims = 3))
    end

    # `img_out` still carries the reduced coil axis as a singleton at position 3; drop it
    # rather than truncating with `reshape`, which would silently discard any trailing
    # batch/time dimension.
    img_out = dropdims(img_out; dims = 3)

    if acq.kspace_data isa NamedDimsArray
        out_dims = filter(!=(:coil), get_image_dims(acq))
        return NamedDimsArray{out_dims}(img_out)
    else
        return img_out
    end
end
