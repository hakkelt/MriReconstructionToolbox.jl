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

    # 2. Estimation and correction
    delays_est = estimate_gradient_delays(acq_noncart; method = OpposingSpokes())
    @test isapprox(delays_est[1], delay_true[1]; atol = 1.0e-3)
    @test isapprox(delays_est[2], delay_true[2]; atol = 1.0e-3)

    acq_corr = correct_gradient_delays(acq_noncart; method = OpposingSpokes())
    @test acq_corr isa NonCartesianAcquisitionInfo
    @test norm(unname(acq_corr.trajectory) - traj_true) < 2.0e-3
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
    @test isapprox(abs.(unname(rec_spirit))[mask_obj], img[mask_obj]; rtol = 0.18)
end
