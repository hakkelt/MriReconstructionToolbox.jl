@testitem "Gradient delay correction in non-Cartesian MRI" tags = [:preprocessing, :acquisition, :nfft] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nsamples = 32
    Nspokes = 16
    angles = range(0, 2π, length = Nspokes + 1)[1:Nspokes]
    r = range(-0.5f0, 0.5f0, length = Nsamples)

    traj_true = zeros(Float32, 2, Nsamples, Nspokes)
    for s in 1:Nspokes
        traj_true[1, :, s] = r .* cos(angles[s])
        traj_true[2, :, s] = r .* sin(angles[s])
    end

    delay_true = (0.02, -0.015)
    traj_delayed = copy(traj_true)
    for s in 1:Nspokes
        traj_delayed[1, :, s] .+= delay_true[1] * cos(angles[s])
        traj_delayed[2, :, s] .+= delay_true[2] * sin(angles[s])
    end

    # Peak centered signal along readout
    ksp = zeros(ComplexF32, Nsamples, Nspokes)
    for s in 1:Nspokes
        shift_s = delay_true[1] * cos(angles[s]) + delay_true[2] * sin(angles[s])
        ksp[:, s] = exp.(-50.0f0 .* (r .- shift_s) .^ 2)
    end

    acq_noncart = NonCartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky)}(ksp);
        trajectory = NamedDimsArray{(:dim, :kx, :ky)}(traj_delayed),
        image_size = (32, 32),
    )

    # 1. Validation test on Cartesian
    acq_cart = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky)}(zeros(ComplexF32, 16, 16));
        is3D = false,
    )
    @test_throws ArgumentError correct_gradient_delays(acq_cart)

    # 2. Estimation and correction with OpposingSpokes
    delays_est = estimate_gradient_delays(acq_noncart; method = OpposingSpokes())
    @test isapprox(delays_est[1], delay_true[1]; atol = 1.0e-3)
    @test isapprox(delays_est[2], delay_true[2]; atol = 1.0e-3)

    acq_corr = correct_gradient_delays(acq_noncart; method = OpposingSpokes())
    @test acq_corr isa NonCartesianAcquisitionInfo
    @test norm(unname(acq_corr.trajectory) - traj_true) < 2.0e-3

    # 3. Estimation and correction with RING (anisotropic delay tensor)
    Sxx, Syy, Sxy = 0.02, -0.015, 0.005
    traj_ring = copy(traj_true)
    for s in 1:Nspokes
        θ = angles[s]
        shift = Sxx * cos(θ)^2 + Syy * sin(θ)^2 + 2.0 * Sxy * cos(θ) * sin(θ)
        traj_ring[1, :, s] .+= shift * cos(θ)
        traj_ring[2, :, s] .+= shift * sin(θ)
    end
    ksp_ring = zeros(ComplexF32, Nsamples, Nspokes)
    for s in 1:Nspokes
        θ = angles[s]
        shift_s = Sxx * cos(θ)^2 + Syy * sin(θ)^2 + 2.0 * Sxy * cos(θ) * sin(θ)
        ksp_ring[:, s] = exp.(-50.0f0 .* (r .- shift_s) .^ 2)
    end
    acq_ring_data = NonCartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky)}(ksp_ring);
        trajectory = NamedDimsArray{(:dim, :kx, :ky)}(traj_ring),
        image_size = (32, 32),
    )
    delays_ring = estimate_gradient_delays(acq_ring_data; method = RING())
    @test delays_ring isa NamedTuple
    @test isapprox(delays_ring.dx, Sxx; atol = 1.0e-3)
    @test isapprox(delays_ring.dy, Syy; atol = 1.0e-3)
    @test isapprox(delays_ring.dxy, Sxy; atol = 1.0e-3)

    acq_ring_corr = correct_gradient_delays(acq_ring_data; method = RING())
    @test acq_ring_corr isa NonCartesianAcquisitionInfo
    @test norm(unname(acq_ring_corr.trajectory) - traj_true) < 2.0e-3
end

@testitem "Partial Fourier reconstruction: Homodyne, StepRamp, POCS" tags = [:reconstruction, :acquisition] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims
    using FFTW

    Nx, Ny = 32, 32
    mag = zeros(Float32, Nx, Ny)
    mag[8:24, 8:24] .= 1.0f0
    X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
    Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
    phase_true = 0.5f0 .* (X .+ Y)
    img_true = NamedDimsArray{(:x, :y)}(ComplexF32.(mag .* cis.(phase_true)))

    # Full k-space
    ksp_full = fftshift(fft(unname(img_true))) ./ sqrt(Nx * Ny)

    # Partial Fourier subsampling: lines 10 to 32 acquired (23 lines out of 32)
    mask_y = falses(Ny)
    mask_y[10:32] .= true
    subs = (:, mask_y)

    ksp_partial = NamedDimsArray{(:kx, :ky)}(ksp_full[:, mask_y])
    acq_pf = CartesianAcquisitionInfo(
        ksp_partial;
        is3D = false,
        image_size = (Nx, Ny),
        subsampling = subs,
    )

    # 1. Test partial_fourier_band
    band = partial_fourier_band(acq_pf)
    @test band.dim == 2
    @test first(band.acquired_range) == 10
    @test last(band.acquired_range) == 32

    # 2. Test Homodyne with LinearRamp
    rec_homodyne_linear = reconstruct(acq_pf, Homodyne(filter = LinearRamp()); verbose = false)
    @test rec_homodyne_linear isa NamedDimsArray
    @test dimnames(rec_homodyne_linear) == (:x, :y)
    mask_obj = mag .> 0.5
    @test isapprox(abs.(unname(rec_homodyne_linear))[mask_obj], mag[mask_obj]; rtol = 0.08)

    # 3. Test Homodyne with StepRamp
    rec_homodyne_step = reconstruct(acq_pf, Homodyne(filter = StepRamp()); verbose = false)
    @test isapprox(abs.(unname(rec_homodyne_step))[mask_obj], mag[mask_obj]; rtol = 0.08)

    # 4. Test POCS
    rec_pocs = reconstruct(acq_pf, POCS(maxit = 15); verbose = false)
    @test isapprox(abs.(unname(rec_pocs))[mask_obj], mag[mask_obj]; rtol = 0.08)
end

@testitem "Parallel imaging: GRAPPA and SPIRiT" tags = [:reconstruction, :acquisition, :encoding] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims
    using FFTW

    Nx, Ny, Nc = 32, 32, 4
    img = zeros(Float32, Nx, Ny)
    img[8:24, 8:24] .= 1.0f0

    sens_true = zeros(ComplexF32, Nx, Ny, Nc)
    X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
    Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
    for c in 1:Nc
        angle = (c - 1) * 2π / Nc
        sens_true[:, :, c] = exp.(-((X .- cos(angle) / 2) .^ 2 .+ (Y .- sin(angle) / 2) .^ 2)) .* cis.(0.5f0 .* (X .* cos(angle) .+ Y .* sin(angle)))
    end
    rss = sqrt.(sum(abs2, sens_true; dims = 3))
    sens_true ./= (rss .+ 1.0f-8)

    # Full k-space
    ksp_full = zeros(ComplexF32, Nx, Ny, Nc)
    for c in 1:Nc
        ksp_full[:, :, c] = fftshift(fft(img .* sens_true[:, :, c])) ./ sqrt(Nx * Ny)
    end

    # R = 2 undersampling with central 12-line ACS region
    R_acc = 2
    cal_range = 11:22
    mask_y = falses(Ny)
    mask_y[1:R_acc:Ny] .= true
    mask_y[cal_range] .= true
    subs = (:, mask_y)

    ksp_under = NamedDimsArray{(:kx, :ky, :coil)}(ksp_full[:, mask_y, :])
    sens_named = NamedDimsArray{(:x, :y, :coil)}(sens_true)

    acq = CartesianAcquisitionInfo(
        ksp_under;
        is3D = false,
        image_size = (Nx, Ny),
        subsampling = subs,
        sensitivity_maps = sens_named,
    )

    mask_obj = img .> 0.5

    # 1. GRAPPA reconstruction
    rec_grappa = reconstruct(acq, GRAPPA(kernel_size = (3, 2), calib_size = (32, 12)); verbose = false)
    @test rec_grappa isa NamedDimsArray
    @test dimnames(rec_grappa) == (:x, :y)
    @test isapprox(abs.(unname(rec_grappa))[mask_obj], img[mask_obj]; rtol = 0.05)

    # 2. SPIRiT reconstruction
    rec_spirit = reconstruct(acq, SPIRiT(kernel_size = (5, 5), calib_size = (32, 12), maxit = 20); verbose = false)
    @test rec_spirit isa NamedDimsArray
    @test dimnames(rec_spirit) == (:x, :y)
    @test norm(abs.(unname(rec_spirit))[mask_obj] .- img[mask_obj]) / norm(img[mask_obj]) < 0.18
end

@testitem "Partial Fourier: PhaseConstrained recovers a phased phantom" tags = [:reconstruction, :acquisition] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims
    using FFTW

    Nx, Ny, Nc = 40, 40, 4
    img = zeros(ComplexF64, Nx, Ny)
    img[10:30, 12:28] .= 1.0
    img[18:24, 18:24] .= 0.4
    img .*= cis.(0.4 .* [x / Nx + y / Ny for x in 1:Nx, y in 1:Ny])

    sens = zeros(ComplexF64, Nx, Ny, Nc)
    X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
    Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
    for c in 1:Nc
        a = (c - 1) * 2π / Nc
        sens[:, :, c] = exp.(-((X .- cos(a) / 2) .^ 2 .+ (Y .- sin(a) / 2) .^ 2)) .* cis.(0.5 .* (X .* cos(a) .+ Y .* sin(a)))
    end
    sens ./= sqrt.(sum(abs2, sens; dims = 3)) .+ 1.0e-8

    full = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        full[:, :, c] = fftshift(fft(img .* sens[:, :, c])) ./ sqrt(Nx * Ny)
    end

    pf = 26
    mask_y = falses(Ny)
    mask_y[1:pf] .= true
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(full[:, 1:pf, :]);
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
    )

    rec = reconstruct(acq, PhaseConstrained(); verbose = false)
    @test rec isa NamedDimsArray
    @test dimnames(rec) == (:x, :y)
    obj = abs.(img) .> 0.2
    @test isapprox(abs.(unname(rec))[obj], abs.(img)[obj]; rtol = 0.05)
end

@testitem "GRAPPA: arbitrary undersampling factor and default even kernel" tags = [:reconstruction, :acquisition, :encoding] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims
    using FFTW

    Nx, Ny, Nc = 40, 40, 6
    img = zeros(ComplexF64, Nx, Ny)
    img[10:30, 10:30] .= 1.0
    img[15:20, 22:26] .= 0.5

    sens = zeros(ComplexF64, Nx, Ny, Nc)
    X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
    Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
    for c in 1:Nc
        a = (c - 1) * 2π / Nc
        sens[:, :, c] = exp.(-((X .- cos(a) / 2) .^ 2 .+ (Y .- sin(a) / 2) .^ 2)) .* cis.(0.6 .* (X .* cos(a) .+ Y .* sin(a)))
    end
    sens ./= sqrt.(sum(abs2, sens; dims = 3)) .+ 1.0e-8

    full = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        full[:, :, c] = fftshift(fft(img .* sens[:, :, c])) ./ sqrt(Nx * Ny)
    end

    for R in (2, 3)
        mask_y = falses(Ny)
        mask_y[1:R:Ny] .= true
        c0 = Ny ÷ 2 + 1
        mask_y[(c0 - 6):(c0 + 5)] .= true
        acq = CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil)}(full[:, mask_y, :]);
            is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y),
            sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
        )
        # default kernel_size = (4, 3) is even along kx - must not throw
        rec = reconstruct(acq, GRAPPA(calib_size = (40, 12)); verbose = false)
        @test dimnames(rec) == (:x, :y)
        rel = norm(abs.(unname(rec)) .- abs.(img)) / norm(abs.(img))
        @test rel < (R == 2 ? 0.05 : 0.2)
    end
end

@testitem "Direct methods: trailing time batch dimension is preserved" tags = [:reconstruction, :acquisition, :encoding] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: Homodyne, POCS, PhaseConstrained
    using NamedDims
    using FFTW

    Nx, Ny, Nc, Nt = 32, 32, 4, 3
    sens = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        sens[:, :, c] .= cis(2π * c / Nc) / sqrt(Nc)
    end
    imgs = zeros(ComplexF64, Nx, Ny, Nt)
    for t in 1:Nt
        imgs[8:24, 8:24, t] .= 1.0 + 0.1t
    end
    full = zeros(ComplexF64, Nx, Ny, Nc, Nt)
    for c in 1:Nc, t in 1:Nt
        full[:, :, c, t] = fftshift(fft(imgs[:, :, t] .* sens[:, :, c])) ./ sqrt(Nx * Ny)
    end
    mask_y = falses(Ny)
    mask_y[1:22] .= true                       # partial Fourier band
    acq_pf = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil, :time)}(full[:, 1:22, :, :]);
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
    )
    for M in (Homodyne(), POCS(maxit = 4), PhaseConstrained())
        rec = reconstruct(acq_pf, M; verbose = false)
        @test size(rec) == (Nx, Ny, Nt)
        @test dimnames(rec) == (:x, :y, :time)
    end

    mask_r = falses(Ny)
    mask_r[1:2:Ny] .= true
    c0 = Ny ÷ 2 + 1
    mask_r[(c0 - 6):(c0 + 5)] .= true
    acq_r = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil, :time)}(full[:, mask_r, :, :]);
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_r),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
    )
    for M in (GRAPPA(calib_size = (32, 12)), SPIRiT(calib_size = (32, 12), maxit = 6))
        rec = reconstruct(acq_r, M; verbose = false)
        @test size(rec) == (Nx, Ny, Nt)
        @test dimnames(rec) == (:x, :y, :time)
    end
end

@testitem "SPIRiTConsistency: operator adjoint test and KSpaceToImage reconstruction" tags = [:reconstruction, :regularization] begin
    using Test
    using MriReconstructionToolbox
    using NamedDims
    using LinearAlgebra
    using FFTW

    Nx, Ny, Nc = 16, 16, 4
    Kx, Ky = 3, 3
    kernel = rand(ComplexF64, Kx, Ky, Nc, Nc)
    cx, cy = 2, 2
    for c in 1:Nc
        kernel[cx, cy, c, c] = 0.0
    end

    reg = SPIRiTConsistency(kernel; λ = 1.0)
    @test MriReconstructionToolbox.scale_regularization(reg, 2.0).λ == 2.0
    @test MriReconstructionToolbox.bind_dimensions(reg, (:x, :y)) === reg

    # 1. Adjoint dot-test on (I - G)
    x = randn(ComplexF64, Nx, Ny, Nc)
    y = randn(ComplexF64, Nx, Ny, Nc)
    op = MriReconstructionToolbox.get_operator(reg, x; threaded = false)
    Ax = op * x
    Aty = op' * y
    @test isapprox(dot(y, Ax), dot(Aty, x); rtol = 1.0e-10)

    # 2. Fully-sampled KSpaceToImage solve matches direct IFFT
    img = zeros(ComplexF64, Nx, Ny)
    img[4:12, 4:12] .= 1.0
    sens = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        sens[:, :, c] .= cis(2π * c / Nc) / sqrt(Nc)
    end
    ksp_full = zeros(ComplexF64, Nx, Ny, Nc)
    for c in 1:Nc
        ksp_full[:, :, c] = fftshift(fft(img .* sens[:, :, c])) ./ sqrt(Nx * Ny)
    end
    acq_full = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(ksp_full);
        is3D = false, image_size = (Nx, Ny),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens),
    )
    rec_kspace = reconstruct(acq_full, IterativeReconstruction(; signal_model = KSpaceToImage(AdjointSensitivity()), algorithm = CGNR(maxit = 5), fidelity = L2Loss()); verbose = false)
    @test isapprox(abs.(unname(rec_kspace)), abs.(img); atol = 1.0e-5)
end

@testitem "Iterative SPIRiT reconstruction (lowering)" tags = [:reconstruction, :acquisition] begin
    using Test
    using MriReconstructionToolbox
    using NamedDims
    using LinearAlgebra
    using FFTW

    Nx, Ny, Nc = 32, 32, 4
    img = zeros(Float32, Nx, Ny)
    img[8:24, 8:24] .= 1.0f0

    sens_true = zeros(ComplexF32, Nx, Ny, Nc)
    X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
    Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
    for c in 1:Nc
        angle = (c - 1) * 2π / Nc
        sens_true[:, :, c] = exp.(-((X .- cos(angle) / 2) .^ 2 .+ (Y .- sin(angle) / 2) .^ 2)) .* cis.(0.5f0 .* (X .* cos(angle) .+ Y .* sin(angle)))
    end
    rss = sqrt.(sum(abs2, sens_true; dims = 3))
    sens_true ./= (rss .+ 1.0f-8)

    ksp_full = zeros(ComplexF32, Nx, Ny, Nc)
    for c in 1:Nc
        ksp_full[:, :, c] = fftshift(fft(img .* sens_true[:, :, c])) ./ sqrt(Nx * Ny)
    end

    R_acc = 2
    cal_range = 11:22
    mask_y = falses(Ny)
    mask_y[1:R_acc:Ny] .= true
    mask_y[cal_range] .= true
    subs = (:, mask_y)

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(ksp_full[:, mask_y, :]);
        is3D = false,
        image_size = (Nx, Ny),
        subsampling = subs,
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(sens_true),
    )

    rec_iter = reconstruct(acq, SPIRiT(kernel_size = (5, 5), calib_size = (32, 12), maxit = 20, iterative = true); verbose = false)
    @test rec_iter isa NamedDimsArray
    @test dimnames(rec_iter) == (:x, :y)
    mask_obj = img .> 0.5
    @test norm(abs.(unname(rec_iter))[mask_obj] .- img[mask_obj]) / norm(img[mask_obj]) < 0.2
end

@testitem "Direct FFT methods respect shifted_kspace_dims" tags = [:reconstruction, :acquisition] begin
    using Test
    using MriReconstructionToolbox
    using NamedDims
    using FFTW
    using LinearAlgebra

    Nx, Ny = 32, 32
    mag = zeros(Float32, Nx, Ny)
    mag[8:24, 8:24] .= 1.0f0
    img_true = NamedDimsArray{(:x, :y)}(ComplexF32.(mag))

    # Standard DC-centered k-space
    ksp_centered = fftshift(fft(unname(img_true))) ./ sqrt(Nx * Ny)
    # Pre-shifted (corner-centered) k-space
    ksp_unshifted = fft(unname(img_true)) ./ sqrt(Nx * Ny)

    acq_default = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky)}(ksp_centered);
        is3D = false, image_size = (Nx, Ny),
    )
    acq_shifted = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky)}(ksp_unshifted);
        is3D = false, image_size = (Nx, Ny),
        shifted_kspace_dims = (1, 2),
    )

    rec_default = reconstruct(acq_default, DirectReconstruction(); verbose = false)
    rec_shifted = reconstruct(acq_shifted, DirectReconstruction(); verbose = false)

    @test isapprox(abs.(unname(rec_default)), abs.(unname(rec_shifted)); atol = 1.0e-5)
end

@testitem "Fourier operator helpers match raw FFT (even and odd sizes)" tags = [:reconstruction, :encoding] begin
    using Test
    using MriReconstructionToolbox
    using NamedDims
    using FFTW
    const MRT = MriReconstructionToolbox

    for (Nx, Ny, Nc) in ((32, 32, 3), (31, 33, 3))
        k = randn(ComplexF64, Nx, Ny, Nc)
        x = randn(ComplexF64, Nx, Ny, Nc)
        even = iseven(Nx) && iseven(Ny)

        # _cartesian_fourier_op must be a consistent forward/adjoint pair, and on even sizes match
        # the pre-refactor `_direct_fft` / `_direct_ifft` bit for bit.
        acq = CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil)}(k); is3D = false, image_size = (Nx, Ny),
        )
        ℱ = MRT._cartesian_fourier_op(acq, k)
        @test ℱ' * (ℱ * x) ≈ x
        if even
            @test ℱ' * k ≈ ifft(ifftshift(k, (1, 2)), (1, 2))
            @test ℱ * x ≈ fftshift(fft(x, (1, 2)), (1, 2))
        end

        # _axis_dft_op: readout-only, k-space-side shift — exact on even and odd
        ro = MRT._axis_dft_op(k, (1,); kspace_shift = true)
        @test ro' * k ≈ ifft(ifftshift(k, 1), 1)
        @test ro * x ≈ fftshift(fft(x, 1), 1)

        # _axis_dft_op: no shift — exact on even and odd
        plain = MRT._axis_dft_op(x, (1, 2))
        @test plain * x ≈ fft(x, (1, 2))
        @test plain' * k ≈ ifft(k, (1, 2))
    end
end
