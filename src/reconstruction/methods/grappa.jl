"""
    GRAPPA{C <: CoilCombination} <: DirectMethod

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
struct GRAPPA{C <: CoilCombination} <: DirectMethod
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

# One tick per ky line of the synthesis loop below; already-acquired lines tick too, so the bar
# is over the full ky extent rather than only the missing lines.
progress_total(::GRAPPA, acq_data) = get_image_size(acq_data)[2]

"""
    _grappa_ky_pattern(acq::CartesianAcquisitionInfo)

Describe the phase-encoding pattern of `acq` as `(acquired, R, acs)`: the per-`ky` acquired flags,
the undersampling stride `R` (`1` when nothing is missing) and the `UnitRange` of the fully sampled
autocalibration block (empty when there is none). Throws when the frequency-encoding direction is
itself subsampled, since GRAPPA's kernel assumes complete `kx` lines.
"""
function _grappa_ky_pattern(acq::CartesianAcquisitionInfo)
    Nx, Ny = get_image_size(acq)[1], get_image_size(acq)[2]
    mask = to_displayable_mask(acq.subsampling, (Nx, Ny))
    acquired = vec(any(mask; dims = 1))
    all(ky -> !acquired[ky] || all(@view mask[:, ky]), 1:Ny) ||
        throw(ArgumentError("GRAPPA requires fully sampled readout (kx) lines, but the sampling pattern subsamples the frequency-encoding direction as well."))

    lines = findall(acquired)
    isempty(lines) && throw(ArgumentError("GRAPPA reconstruction requires at least one acquired phase-encoding line."))
    gaps = diff(lines)
    strides = filter(>(1), gaps)
    # The stride is the *most common* gap larger than one, not the smallest: the two gaps that
    # bridge the ACS block back onto the lattice are whatever the block's edges happen to leave
    # (often smaller than R), so `minimum` mis-detects R = 2 for, say, an R = 3 acquisition.
    R = if isempty(strides)
        1
    else
        counts = Dict{Int, Int}()
        for g in strides
            counts[g] = get(counts, g, 0) + 1
        end
        best = maximum(values(counts))
        minimum(g for (g, n) in counts if n == best)
    end

    # The autocalibration block is the longest run of consecutive acquired lines.
    runs = UnitRange{Int}[]
    start = 1
    for i in eachindex(gaps)
        if gaps[i] != 1
            push!(runs, lines[start]:lines[i])
            start = i + 1
        end
    end
    push!(runs, lines[start]:lines[end])
    filter!(r -> length(r) > 1, runs)
    acs = isempty(runs) ? (1:0) : argmax(length, runs)
    return acquired, R, acs, gaps
end

function check_applicable(method::GRAPPA, acq::NonCartesianAcquisitionInfo)
    throw(ArgumentError("GRAPPA is a Cartesian method: it interpolates missing phase-encoding lines on a regular grid. Use an iterative reconstruction (e.g. `IterativeReconstruction`) for non-Cartesian data."))
end

function check_applicable(method::GRAPPA, acq::CartesianAcquisitionInfo)
    isnothing(acq.subsampling) &&
        throw(ArgumentError("GRAPPA reconstruction requires an undersampled Cartesian acquisition, but `acq.subsampling` is `nothing`."))
    isnothing(acq.sensitivity_maps) && !(method.coil_combination isa RootSumSquares) &&
        throw(ArgumentError("GRAPPA requires sensitivity maps when using AdjointSensitivity coil combination"))

    acquired, R, acs, gaps = _grappa_ky_pattern(acq)
    R == 1 && return nothing   # fully sampled along ky: nothing to synthesize

    # The kernel is fitted once per missing-line offset `t = 1 … R-1` and reused everywhere, so
    # outside the ACS block every acquired line must sit on one lattice of stride `R`. A random or
    # variable-density mask has no such lattice: the kernel would be applied to source lines it was
    # never calibrated for (or to lines that were not acquired at all), silently producing garbage.
    lines = findall(acquired)
    lattice = setdiff(lines, acs)
    origin = isempty(lattice) ? first(lines) : first(lattice)
    off_lattice = filter(l -> mod(l - origin, R) != 0, lattice)
    isempty(off_lattice) ||
        throw(
        ArgumentError(
            "GRAPPA requires a regularly undersampled phase-encoding pattern (a fixed stride R plus a contiguous ACS block), " *
                "but $(length(off_lattice)) of the $(length(lattice)) acquired lines outside the ACS block do not lie on the " *
                "stride-$R lattice (line-to-line gaps $(sort(unique(gaps)))). " *
                "Random / variable-density patterns are not GRAPPA-reconstructible; use a calibrationless or compressed-sensing method instead."
        ),
    )
    isempty(acs) &&
        throw(ArgumentError("GRAPPA requires a fully sampled autocalibration (ACS) region, but the sampling mask has no run of consecutive phase-encoding lines."))
    length(acs) >= method.kernel_size[2] * R ||
        throw(
        ArgumentError(
            "GRAPPA's ACS region is $(length(acs)) lines, too few for kernel_size=$(method.kernel_size) at R=$R " *
                "(needs at least $(method.kernel_size[2] * R))."
        ),
    )
    return nothing
end

function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::GRAPPA; progress = nothing)
    @argcheck !isnothing(acq.subsampling) "GRAPPA reconstruction requires an undersampled Cartesian acquisition"
    @argcheck !isnothing(acq.sensitivity_maps) || method.coil_combination isa RootSumSquares "GRAPPA requires sensitivity maps when using AdjointSensitivity coil combination"

    raw_ksp = _get_full_kspace(acq)
    Nx, Ny = size(raw_ksp, 1), size(raw_ksp, 2)
    Nc = size(raw_ksp, 3)
    T = complex(real(eltype(raw_ksp)))

    # Same pattern analysis `check_applicable` validates against, so synthesis and validation can
    # never disagree about R.
    acquired, R_acc, _, _ = _grappa_ky_pattern(acq)

    cal_kx, cal_ky = min(Nx, method.calib_size[1]), min(Ny, method.calib_size[2])
    cx, cy = Nx ÷ 2 + 1, Ny ÷ 2 + 1
    cal_range_x = (cx - cal_kx ÷ 2):(cx + cal_kx ÷ 2 - 1)
    cal_range_y = (cy - cal_ky ÷ 2):(cy + cal_ky ÷ 2 - 1)
    calib = raw_ksp[cal_range_x, cal_range_y, :]

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
            patch = zeros(T, Kx, Ky_src, Nc)
            b = 1
            for bx in 1:(cal_kx - Kx + 1), by in by_range
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
            isnothing(progress) || progress()
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
            patch = zeros(T, Kx, Ky_src, Nc)
            for kx in 1:Nx
                for (jy, r) in enumerate(rows), (ix, xo) in enumerate(x_taps)
                    patch[ix, jy, :] = raw_ksp[mod1(kx + xo, Nx), r, :]
                end
                ksp_recon[kx, ky, :] = reshape(patch, 1, :) * W
            end
        end
    end

    return _kspace_to_image(ksp_recon, method.coil_combination, acq.sensitivity_maps, acq)
end
