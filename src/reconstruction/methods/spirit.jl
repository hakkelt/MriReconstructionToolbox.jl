"""
    SPIRiTConsistency{K, R <: Real} <: Regularization

Self-consistency k-space convolution regularization term ``\\tfrac{\\lambda}{2}\\|(I - G)\\,k\\|_2^2``
for SPIRiT reconstruction (Lustig & Pauly 2010, MRM 64:457-471), where ``k`` is the multi-channel
k-space and ``G`` the calibrated coil-mixing convolution.
"""
struct SPIRiTConsistency{K, R <: Real} <: Regularization
    kernel::K
    λ::R
end

SPIRiTConsistency(kernel; λ = 1.0) = SPIRiTConsistency(kernel, Float64(λ))

scale_regularization(reg::SPIRiTConsistency, factor::Real) = SPIRiTConsistency(reg.kernel, reg.λ * factor)

bind_dimensions(reg::SPIRiTConsistency, ::Any) = reg

function get_operator(reg::SPIRiTConsistency, x::AbstractArray; threaded::Bool = true)
    Kx, Ky, Nc, _ = size(reg.kernel)
    Nx, Ny = size(x, 1), size(x, 2)
    T = eltype(x)
    pad_x = Kx ÷ 2
    pad_y = Ky ÷ 2

    G_fft = zeros(T, Nx, Ny, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        padded = zeros(T, Nx, Ny)
        padded[1:Kx, 1:Ky] = reg.kernel[:, :, j, i]
        padded = circshift(padded, (-pad_x, -pad_y))
        G_fft[:, :, j, i] = fft(padded)
    end

    fwd! = (y, x_in) -> begin
        fill!(y, zero(T))
        for i in 1:Nc
            y[:, :, i] .= x_in[:, :, i]
        end
        for j in 1:Nc
            xj_fft = fft(x_in[:, :, j])
            for i in 1:Nc
                y[:, :, i] .-= ifft(xj_fft .* G_fft[:, :, j, i])
            end
        end
    end

    adj! = (y, x_in) -> begin
        fill!(y, zero(T))
        for i in 1:Nc
            y[:, :, i] .= x_in[:, :, i]
        end
        for i in 1:Nc
            xi_fft = fft(x_in[:, :, i])
            for j in 1:Nc
                y[:, :, j] .-= ifft(xi_fft .* conj.(G_fft[:, :, j, i]))
            end
        end
    end

    op = MyLinOp(T, (Nx, Ny, Nc), (Nx, Ny, Nc), fwd!, adj!)
    if x isa NamedDimsArray
        return NamedDimsOp{dimnames(x), dimnames(x)}(op)
    else
        return op
    end
end

function materialize(reg::SPIRiTConsistency, x::Variable{T}; threaded::Bool) where {T}
    op = get_operator(reg, ~x; threaded)
    R = real(T)
    repr = @sprintf "½‖(I - G) ⋅ %s‖₂²" get_name(x)
    return StructuredOptimization.Term(1, SqrNormL2(R(reg.λ)), op * x, repr)
end

get_affected_dims(::SPIRiTConsistency, ::Nothing, image_dims) = image_dims

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
- `iterative`: If `true`, lowers to `IterativeReconstruction` with `KSpaceDomain` and `DouglasRachford`.
"""
struct SPIRiT{C <: CoilCombination} <: AbstractDirectMethod
    kernel_size::Tuple{Int, Int}
    calib_size::Tuple{Int, Int}
    maxit::Int
    λ::Float64
    coil_combination::C
    iterative::Bool
    function SPIRiT(;
            kernel_size = (5, 5),
            calib_size = (24, 24),
            maxit = 25,
            λ = 1.0,
            coil_combination::CoilCombination = RootSumSquares(),
            iterative::Bool = false,
        )
        return new{typeof(coil_combination)}(
            kernel_size,
            calib_size,
            maxit,
            Float64(λ),
            coil_combination,
            iterative,
        )
    end
end

function _calibrate_spirit_kernel(acq::CartesianAcquisitionInfo, method::SPIRiT)
    raw_ksp = _get_full_kspace(acq)
    Nx, Ny = size(raw_ksp, 1), size(raw_ksp, 2)
    Nc = size(raw_ksp, 3)
    T = eltype(raw_ksp)

    cal_kx = min(Nx, method.calib_size[1])
    cal_ky = min(Ny, method.calib_size[2])
    cx, cy = Nx ÷ 2 + 1, Ny ÷ 2 + 1
    cal_range_x = (cx - cal_kx ÷ 2):(cx + (cal_kx - 1) ÷ 2)
    cal_range_y = (cy - cal_ky ÷ 2):(cy + (cal_ky - 1) ÷ 2)
    calib = raw_ksp[cal_range_x, cal_range_y, 1:Nc]

    Kx, Ky = method.kernel_size
    pad_x = Kx ÷ 2
    pad_y = Ky ÷ 2
    center_idx = pad_x + 1 + pad_y * Kx

    num_b = (cal_kx - Kx + 1) * (cal_ky - Ky + 1)
    @argcheck num_b > (Kx * Ky * Nc) "SPIRiT calibration region $(method.calib_size) is too small for kernel_size=$(method.kernel_size)"

    A_mat = zeros(T, num_b, Kx * Ky * Nc)
    b = 1
    for bx in 1:(cal_kx - Kx + 1), by in 1:(cal_ky - Ky + 1)
        patch = calib[bx:(bx + Kx - 1), by:(by + Ky - 1), :]
        A_mat[b, :] = reshape(patch, :)
        b += 1
    end

    G_kernels = zeros(T, Kx, Ky, Nc, Nc)
    for target_c in 1:Nc
        target_feat_idx = center_idx + (target_c - 1) * (Kx * Ky)
        src_cols = filter(!=(target_feat_idx), 1:(Kx * Ky * Nc))

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
    return G_kernels
end

function lower(method::SPIRiT, acq::CartesianAcquisitionInfo)
    if !method.iterative
        return method
    end
    kernel = _calibrate_spirit_kernel(acq, method)
    return IterativeReconstruction(
        SPIRiTConsistency(kernel; λ = method.λ);
        domain = KSpaceDomain(method.coil_combination),
        fidelity = HardConsistency(),
        algorithm = FISTA(adaptive = true),
    )
end

function _kspace_to_image(ksp::AbstractArray, coil_combine::CoilCombination, sens::Union{Nothing, AbstractArray}, acq::CartesianAcquisitionInfo)
    Nx, Ny = get_image_size(acq)[1:2]
    f_dims = (1, 2)
    coil_imgs = _direct_ifft(acq, unname(ksp); dims = f_dims) .* sqrt(Nx * Ny)

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

    if !(coil_combine isa NoCoilCombination)
        img_out = dropdims(img_out; dims = c_dim)
    end

    if acq.kspace_data isa NamedDimsArray
        out_d = !(coil_combine isa NoCoilCombination) ? filter(!=(:coil), get_image_dims(acq)) : get_image_dims(acq)
        return NamedDimsArray{out_d}(img_out)
    else
        return img_out
    end
end

function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::SPIRiT)
    raw_ksp = _get_full_kspace(acq)
    Nx, Ny = size(raw_ksp, 1), size(raw_ksp, 2)
    Nc = size(raw_ksp, 3)
    T = eltype(raw_ksp)

    Kx, Ky = method.kernel_size
    pad_x = Kx ÷ 2
    pad_y = Ky ÷ 2

    mask = to_displayable_mask(acq.subsampling, (Nx, Ny))
    mask_3d = reshape(mask, Nx, Ny, fill(1, ndims(raw_ksp) - 2)...)

    G_kernels = _calibrate_spirit_kernel(acq, method)

    # Iterative projection onto data consistency + G convolution
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
        Gx = zeros(T, Nx, Ny, Nc)
        for j in 1:Nc
            x_j_fft = fft(x_ksp[:, :, j])
            for i in 1:Nc
                Gx[:, :, i] .+= ifft(x_j_fft .* G_fft[:, :, j, i])
            end
        end
        x_ksp = ifelse.(mask_3d, raw_ksp, Gx)
    end

    return _kspace_to_image(x_ksp, method.coil_combination, acq.sensitivity_maps, acq)
end
