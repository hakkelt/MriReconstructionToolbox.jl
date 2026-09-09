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

"""
    _spirit_plane_dft(T, Nx, Ny, batch; threaded)

Planned batched 2D `DFT` over the `(1, 2)` axes of an `Nx×Ny×batch` complex array, `BACKWARD`
normalized (`dft * ·` is the unnormalized forward transform, `dft' * ·` the `1/N` inverse) — the
same convention the raw `fft` / `ifft` calls it replaces had.
"""
function _spirit_plane_dft(::Type{T}, Nx::Integer, Ny::Integer, batch::Integer; threaded::Bool) where {T}
    return DFT(
        zeros(T, Nx, Ny, batch), (1, 2);
        normalization = FFTWOperators.BACKWARD,
        num_threads = threaded ? nthreads() : 1,
    )
end

"""
    _spirit_gfft(kernel, Nx, Ny; threaded)

Zero-pad the calibrated `Kx×Ky×Nc×Nc` SPIRiT kernel onto the full `Nx×Ny` grid and Fourier
transform every coil-pair slice in one batched apply, yielding the `Nx×Ny×Nc×Nc` frequency-domain
convolution kernel `Ĝ` with `Ĝ[:, :, src, target]` layout.

The kernel is *reversed* along both k-space axes before zero-padding. Calibration
([`_calibrate_spirit_kernel`](@ref)) fits a **correlation** — the target sample is the weighted sum
of `x[n + m]` over the patch offsets `m` — while multiplying by `Ĝ` in the frequency domain applies
a **convolution**, `Σₘ h[m] x[n − m]`. Reversing the taps (`h[m] = kernel[−m]`) makes the two agree;
without it the fitted neighbourhood is applied point-reflected through the kernel centre and the
self-consistency relation is badly violated (Lustig & Pauly's reference implementation performs the
same flip).
"""
function _spirit_gfft(kernel::AbstractArray, Nx::Integer, Ny::Integer; threaded::Bool = true)
    Kx, Ky, Nc, _ = size(kernel)
    T = eltype(kernel)
    pad_x, pad_y = Kx ÷ 2, Ky ÷ 2
    # Position of the kernel centre after the reversal below, so the circshift lands it on index 1.
    shift_x, shift_y = Kx - pad_x - 1, Ky - pad_y - 1
    padded = zeros(T, Nx, Ny, Nc * Nc)
    for tgt in 1:Nc, src in 1:Nc
        # Column-major flattening of the trailing `(src, target)` pair, so the `reshape` below puts
        # `kernel[:, :, src, tgt]` back at `Ĝ[:, :, src, tgt]` — the layout `mul!` indexes with.
        idx = src + (tgt - 1) * Nc
        slice = @view padded[:, :, idx]
        slice[1:Kx, 1:Ky] .= @view kernel[Kx:-1:1, Ky:-1:1, src, tgt]
        padded[:, :, idx] .= circshift(slice, (-shift_x, -shift_y))
    end
    dft = _spirit_plane_dft(T, Nx, Ny, Nc * Nc; threaded)
    return reshape(dft * padded, Nx, Ny, Nc, Nc)
end

"""
    SPIRiTConsistencyOp{T} <: AbstractOperators.LinearOperator

The SPIRiT self-consistency operator `(I − G)` on the multi-channel k-space `ℂ^{Nx×Ny×Nc}`, where
`G` is the calibrated coil-mixing convolution. The forward/adjoint transforms go through one cached
planned `DFT`, and the frequency-domain coil mix accumulates into reused scratch, so `mul!` is
allocation-free in steady state. Owns its scratch ⇒ not thread-safe; `copy_operator` gives a copy
its own buffers and plan.
"""
struct SPIRiTConsistencyOp{
        T, D <: AbstractOperators.AbstractOperator, A4 <: AbstractArray{T, 4}, A3 <: AbstractArray{T, 3},
    } <: AbstractOperators.LinearOperator
    G_fft::A4
    dft::D
    x_freq::A3
    acc::A3
    Gx::A3
end

function SPIRiTConsistencyOp(G_fft::AbstractArray{T, 4}, Nx::Integer, Ny::Integer; threaded::Bool = true) where {T}
    Nc = size(G_fft, 3)
    dft = _spirit_plane_dft(T, Nx, Ny, Nc; threaded)
    scratch() = zeros(T, Nx, Ny, Nc)
    return SPIRiTConsistencyOp(G_fft, dft, scratch(), scratch(), scratch())
end

function _spirit_consistency_mul!(y::AbstractArray, op::SPIRiTConsistencyOp, x::AbstractArray, adjoint::Bool)
    AbstractOperators.check(y, op, x)   # domain and codomain coincide, so one check covers both directions
    Nc = size(x, 3)
    mul!(op.x_freq, op.dft, x)                    # batched unnormalized forward DFT, all coils
    fill!(op.acc, zero(eltype(op.acc)))
    Ĝ, xf, ac = op.G_fft, op.x_freq, op.acc
    Nkx, Nky = size(ac, 1), size(ac, 2)
    @inbounds for d in 1:Nc, c in 1:Nc, py in 1:Nky, px in 1:Nkx
        g = adjoint ? conj(Ĝ[px, py, d, c]) : Ĝ[px, py, c, d]
        ac[px, py, d] += xf[px, py, c] * g
    end
    mul!(op.Gx, op.dft', op.acc)                  # batched 1/N inverse DFT, all coils
    @. y = x - op.Gx
    return y
end

mul!(y::AbstractArray, op::SPIRiTConsistencyOp, x::AbstractArray) = _spirit_consistency_mul!(y, op, x, false)
function mul!(y::AbstractArray, adj::AbstractOperators.AdjointOperator{<:SPIRiTConsistencyOp}, x::AbstractArray)
    return _spirit_consistency_mul!(y, adj.A, x, true)
end

Base.size(op::SPIRiTConsistencyOp) = (size(op.x_freq), size(op.x_freq))
AbstractOperators.domain_type(::SPIRiTConsistencyOp{T}) where {T} = T
AbstractOperators.codomain_type(::SPIRiTConsistencyOp{T}) where {T} = T
AbstractOperators.domain_array_type(::SPIRiTConsistencyOp{T}) where {T} = Array{T}
AbstractOperators.codomain_array_type(::SPIRiTConsistencyOp{T}) where {T} = Array{T}
AbstractOperators.fun_name(::SPIRiTConsistencyOp) = "(I-G)"
AbstractOperators.is_thread_safe(::SPIRiTConsistencyOp) = false

function AbstractOperators._copy_operator_impl(op::SPIRiTConsistencyOp; storage_type = nothing, threaded = nothing)
    Nx, Ny, Nc = size(op.x_freq)
    dft = _spirit_plane_dft(eltype(op.x_freq), Nx, Ny, Nc; threaded = threaded === nothing ? true : threaded)
    scratch() = zeros(eltype(op.x_freq), Nx, Ny, Nc)
    return SPIRiTConsistencyOp(copy(op.G_fft), dft, scratch(), scratch(), scratch())
end

function get_operator(reg::SPIRiTConsistency, x::AbstractArray; threaded::Bool = true)
    Nx, Ny = size(x, 1), size(x, 2)
    G_fft = _spirit_gfft(reg.kernel, Nx, Ny; threaded)
    op = SPIRiTConsistencyOp(complex(eltype(x)).(G_fft), Nx, Ny; threaded)
    return x isa NamedDimsArray ? NamedDimsOp{dimnames(x), dimnames(x)}(op) : op
end

function materialize(reg::SPIRiTConsistency, x::Variable{T}; threaded::Bool) where {T}
    op = get_operator(reg, ~x; threaded)
    R = real(T)
    repr = @sprintf "½‖(I - G) ⋅ %s‖₂²" get_name(x)
    return StructuredOptimization.Term(1, SqrNormL2(R(reg.λ)), op * x, repr)
end

get_affected_dims(::SPIRiTConsistency, ::Nothing, image_dims) = image_dims

"""
    SPIRiT{C <: CoilCombination} <: DirectMethod

Iterative Self-consistent Parallel Imaging Reconstruction (Lustig & Pauly 2010).
Calibrates local multi-channel k-space convolution kernels from central ACS data and solves
for missing k-space samples using self-consistency iterations.

# Fields
- `kernel_size`: Size of convolution kernel `(Kx, Ky)` (default: `(5, 5)`).
- `calib_size`: ACS calibration region size (default: `(24, 24)`).
- `maxit`: Number of iterations (default: `25`).
- `λ`: Regularization parameter on self-consistency (default: `1.0`).
- `calib_λ`: Relative Tikhonov regularization of the kernel calibration solve (default: `1.0e-4`).
  The ACS system is close to rank-deficient — neighbouring k-space samples are highly correlated —
  so the unregularized least-squares fit produces a kernel that amplifies noise and extrapolation
  error. The penalty is scaled by `‖X‖²/n` so the value is dimensionless; set it to `0` to recover
  the plain least-squares solve.
- `coil_combination`: Method for combining reconstructed multi-coil channels (`RootSumSquares()` or `AdjointSensitivity()`).
- `iterative`: If `true`, lowers to an `IterativeReconstruction` with a `KSpaceToImage` signal model
  and hard data consistency.
"""
struct SPIRiT{C <: CoilCombination} <: DirectMethod
    kernel_size::Tuple{Int, Int}
    calib_size::Tuple{Int, Int}
    maxit::Int
    λ::Float64
    calib_λ::Float64
    coil_combination::C
    iterative::Bool
    function SPIRiT(;
            kernel_size = (5, 5),
            calib_size = (24, 24),
            maxit = 25,
            λ = 1.0,
            calib_λ = 1.0e-4,
            coil_combination::CoilCombination = RootSumSquares(),
            iterative::Bool = false,
        )
        @argcheck calib_λ >= 0 "SPIRiT calib_λ must be non-negative"
        return new{typeof(coil_combination)}(
            kernel_size,
            calib_size,
            maxit,
            Float64(λ),
            Float64(calib_λ),
            coil_combination,
            iterative,
        )
    end
end

function _calibrate_spirit_kernel(acq::CartesianAcquisitionInfo, method::SPIRiT, raw_ksp::AbstractArray = _get_full_kspace(acq))
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
        w = if method.calib_λ > 0
            # Relative Tikhonov: the penalty tracks the scale of the calibration data, so the same
            # `calib_λ` is meaningful across datasets and normalizations.
            XhX = X_src' * X_src
            μ = real(eltype(XhX))(method.calib_λ * real(tr(XhX)) / size(X_src, 2))
            (XhX + μ * I) \ (X_src' * y_tgt)
        else
            X_src \ y_tgt
        end

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
        signal_model = KSpaceToImage(method.coil_combination),
        fidelity = HardConsistency(),
        algorithm = FISTA(adaptive = true),
        maxit = method.maxit,
    )
end

progress_total(method::SPIRiT, acq_data) = method.maxit

function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::SPIRiT; progress = nothing)
    raw_ksp = _get_full_kspace(acq)
    Nx, Ny = size(raw_ksp, 1), size(raw_ksp, 2)

    mask = to_displayable_mask(acq.subsampling, (Nx, Ny))
    mask_3d = reshape(mask, Nx, Ny, fill(1, ndims(raw_ksp) - 2)...)

    G_fft = _spirit_gfft(_calibrate_spirit_kernel(acq, method, raw_ksp), Nx, Ny)
    op = SPIRiTConsistencyOp(complex(eltype(raw_ksp)).(G_fft), Nx, Ny)

    # Iterative projection onto data consistency + G convolution. `op` applies (I − G), so the
    # G-convolved estimate is `x_ksp − (I − G) x_ksp`.
    x_ksp = copy(raw_ksp)
    img_minus_g = similar(x_ksp)
    for _ in 1:(method.maxit)
        mul!(img_minus_g, op, x_ksp)
        x_ksp = ifelse.(mask_3d, raw_ksp, x_ksp .- img_minus_g)
        isnothing(progress) || progress()
    end

    return _kspace_to_image(x_ksp, method.coil_combination, acq.sensitivity_maps, acq)
end
