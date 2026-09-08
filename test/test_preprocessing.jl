@testitem "Prewhitening and noise covariance estimation" tags = [:acquisition, :encoding] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nc = 4
    Nx, Ny = 16, 16
    # Synthesize non-trivial positive definite noise covariance
    A = randn(ComplexF32, Nc, Nc)
    Ψ_true = A * A' + 0.2f0 * I

    L_true = cholesky(Hermitian(Ψ_true)).L
    Nsamples = 10000
    noise_raw = L_true * randn(ComplexF32, Nc, Nsamples)
    noise_named = NamedDimsArray{(:kx, :ky, :coil)}(
        reshape(permutedims(reshape(noise_raw, Nc, Nx, :), (2, 3, 1)), Nx, :, Nc)
    )

    # Estimate covariance
    Ψ_est = estimate_noise_covariance(noise_named)
    @test size(Ψ_est) == (Nc, Nc)
    @test isapprox(Ψ_est, Ψ_true; rtol = 0.1)

    # Prewhiten acquisition
    img = NamedDimsArray{(:x, :y)}(randn(ComplexF32, Nx, Ny))
    sens = NamedDimsArray{(:x, :y, :coil)}(randn(ComplexF32, Nx, Ny, Nc))
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc));
        is3D = false,
        sensitivity_maps = sens,
    )
    acq_sim = simulate_acquisition(img, acq)
    # Add correlated noise to k-space
    ksp_flat = reshape(permutedims(unname(acq_sim.kspace_data), (3, 1, 2)), Nc, :)
    ksp_noisy_flat = ksp_flat + L_true * randn(ComplexF32, Nc, size(ksp_flat, 2))
    ksp_noisy = NamedDimsArray{(:kx, :ky, :coil)}(
        permutedims(reshape(ksp_noisy_flat, Nc, Nx, Ny), (2, 3, 1))
    )
    acq_noisy = CartesianAcquisitionInfo(acq_sim; kspace_data = ksp_noisy)

    acq_white = prewhiten(acq_noisy, Ψ_est)
    @test acq_white isa CartesianAcquisitionInfo
    @test acq_white.kspace_data isa NamedDimsArray
    @test dimnames(acq_white.kspace_data) == (:kx, :ky, :coil)
    @test acq_white.sensitivity_maps isa NamedDimsArray
    @test dimnames(acq_white.sensitivity_maps) == (:x, :y, :coil)
end

@testitem "Coil compression with SVDCompression" tags = [:acquisition, :encoding] setup = [SyntheticCoils] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nc = 8
    Nv = 4
    Nx, Ny = 16, 16
    img = NamedDimsArray{(:x, :y)}(randn(ComplexF32, Nx, Ny))

    sens = NamedDimsArray{(:x, :y, :coil)}(synthetic_sensitivities(ComplexF32, Nx, Ny, Nc))

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc));
        is3D = false,
        sensitivity_maps = sens,
    )
    acq_sim = simulate_acquisition(img, acq)

    acq_comp, C = compress_coils(acq_sim, Nv; method = SVDCompression())
    @test size(C) == (Nv, Nc)
    @test size(acq_comp.kspace_data, :coil) == Nv
    @test size(acq_comp.sensitivity_maps, :coil) == Nv
    @test dimnames(acq_comp.kspace_data) == (:kx, :ky, :coil)
    @test dimnames(acq_comp.sensitivity_maps) == (:x, :y, :coil)

    # Direct reconstruction from compressed data
    rec_orig = reconstruct(acq_sim, DirectReconstruction(); verbosity = Silent())
    rec_comp = reconstruct(acq_comp, DirectReconstruction(); verbosity = Silent())
    @test isapprox(rec_comp, rec_orig; rtol = 0.05)

    # GeometricCompression
    acq_geom, C_geom = compress_coils(acq_sim, Nv; method = GeometricCompression())
    @test size(C_geom) == (Nv, Nc, Nx)
    @test size(acq_geom.kspace_data, :coil) == Nv
    @test size(acq_geom.sensitivity_maps, :coil) == Nv
    rec_geom = reconstruct(acq_geom, DirectReconstruction(); verbosity = Silent())
    @test isapprox(rec_geom, rec_orig; rtol = 0.05)
end

@testitem "Sensitivity map estimation: SelfCalibrating, AdaptiveCombine, ESPIRiT" tags = [:acquisition, :encoding, :simulation] setup = [SyntheticCoils] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nc = 32, 32, 4
    img = NamedDimsArray{(:x, :y)}(zeros(ComplexF32, Nx, Ny))
    img[8:24, 8:24] .= 1.0f0

    sens_true = NamedDimsArray{(:x, :y, :coil)}(synthetic_sensitivities(ComplexF32, Nx, Ny, Nc))

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc));
        is3D = false,
        sensitivity_maps = sens_true,
    )
    acq_sim = simulate_acquisition(img, acq)

    # 1. SelfCalibrating
    acq_selfcal = estimate_sensitivities(acq_sim; method = SelfCalibrating(calib_size = 16))
    @test !isnothing(acq_selfcal.sensitivity_maps)
    @test size(acq_selfcal.sensitivity_maps) == (Nx, Ny, Nc)
    @test dimnames(acq_selfcal.sensitivity_maps) == (:x, :y, :coil)

    # 2. AdaptiveCombine
    acq_adaptive = estimate_sensitivities(acq_sim; method = AdaptiveCombine(kernel_size = 5))
    @test !isnothing(acq_adaptive.sensitivity_maps)
    @test size(acq_adaptive.sensitivity_maps) == (Nx, Ny, Nc)
    @test dimnames(acq_adaptive.sensitivity_maps) == (:x, :y, :coil)

    # 3. ESPIRiT
    acq_espirit = estimate_sensitivities(acq_sim; method = ESPIRiT(calib_size = 16, kernel_size = 6))
    @test !isnothing(acq_espirit.sensitivity_maps)
    @test size(acq_espirit.sensitivity_maps) == (Nx, Ny, Nc)
    @test dimnames(acq_espirit.sensitivity_maps) == (:x, :y, :coil)

    # Verify phase-aligned sensitivity map accuracy in object support
    mask = abs.(unname(img)) .> 0.5
    for sens_est in (acq_selfcal.sensitivity_maps, acq_adaptive.sensitivity_maps, acq_espirit.sensitivity_maps)
        dot_prod = sum(unname(sens_est) .* conj(unname(sens_true)), dims = 3)
        phase_diff = cis.(-angle.(dot_prod))
        aligned = unname(sens_est) .* phase_diff
        rel_err = norm(aligned[mask, :] - unname(sens_true)[mask, :]) / norm(unname(sens_true)[mask, :])
        @test rel_err < 0.05
    end

    # Verify direct reconstruction magnitude with estimated maps
    rec_selfcal = reconstruct(acq_selfcal, DirectReconstruction(); verbosity = Silent())
    rec_adaptive = reconstruct(acq_adaptive, DirectReconstruction(); verbosity = Silent())
    rec_espirit = reconstruct(acq_espirit, DirectReconstruction(); verbosity = Silent())

    @test isapprox(abs.(unname(rec_selfcal))[mask], abs.(unname(img))[mask]; rtol = 0.15)
    @test isapprox(abs.(unname(rec_adaptive))[mask], abs.(unname(img))[mask]; rtol = 0.15)
    @test isapprox(abs.(unname(rec_espirit))[mask], abs.(unname(img))[mask]; rtol = 0.15)
end

@testitem "Sensitivity estimation: measured k-space smaller than image_size is zero-padded" tags = [:preprocessing, :acquisition] setup = [SyntheticCoils] begin
    using Test
    using MriReconstructionToolbox
    using NamedDims

    Nx, Ny, Nc, measured = 64, 64, 4, 41
    img = NamedDimsArray{(:x, :y)}(zeros(ComplexF32, Nx, Ny))
    img[16:48, 16:48] .= 1.0f0
    sens_true = NamedDimsArray{(:x, :y, :coil)}(synthetic_sensitivities(ComplexF32, Nx, Ny, Nc))

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc));
        is3D = false, sensitivity_maps = sens_true,
    )
    acq_sim = simulate_acquisition(img, acq)

    # Only the central `measured` phase-encode lines are kept, mirroring a real acquisition where
    # `kspace_data` stores just the measured extent rather than a zero-filled `image_size` grid.
    lo = (Ny - measured) ÷ 2 + 1
    ksp_measured = unname(acq_sim.kspace_data)[:, lo:(lo + measured - 1), :]
    acq_measured = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(ksp_measured); is3D = false, image_size = (Nx, Ny),
    )

    # Previously threw an ArgCheck size-mismatch when re-attaching maps sized (Nx, measured, Nc).
    acq_out = estimate_sensitivities(acq_measured; method = ESPIRiT(calib_size = 24, kernel_size = 6))
    @test size(acq_out.sensitivity_maps) == (Nx, Ny, Nc)

    # The raw-array method accepts the same `image_size` keyword directly, and is a no-op without it.
    padded = estimate_sensitivities(
        ksp_measured; method = ESPIRiT(calib_size = 24, kernel_size = 6), image_size = (Nx, Ny)
    )
    @test size(padded) == (Nx, Ny, Nc)
    unpadded = estimate_sensitivities(ksp_measured; method = ESPIRiT(calib_size = 24, kernel_size = 6))
    @test size(unpadded) == (Nx, measured, Nc)
end

@testitem "Sensitivity estimation: coil axis need not be trailing" tags = [:preprocessing] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nc = 32, 32, 4
    ksp = randn(ComplexF64, Nx, Ny, Nc)

    s_trailing = estimate_sensitivities(ksp; method = SelfCalibrating(), coil_dim = 3)
    s_leading = estimate_sensitivities(permutedims(ksp, (3, 1, 2)); method = SelfCalibrating(), coil_dim = 1)
    @test permutedims(s_leading, (2, 3, 1)) ≈ s_trailing

    # NamedDims with a non-trailing coil axis
    kn = NamedDimsArray{(:coil, :kx, :ky)}(permutedims(ksp, (3, 1, 2)))
    sn = estimate_sensitivities(kn; method = SelfCalibrating())
    @test dimnames(sn) == (:coil, :x, :y)
    @test permutedims(unname(sn), (2, 3, 1)) ≈ s_trailing
end

@testitem "Gradient delays: multi-coil k-space is combined over the coil axis" tags = [:preprocessing, :acquisition, :nfft] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: NonCartesianAcquisitionInfo

    Nsamples, Nspokes, Nc = 64, 30, 8
    angles = range(0, 2π, length = Nspokes + 1)[1:Nspokes]
    r = range(-0.5, 0.5, length = Nsamples)
    traj = zeros(2, Nsamples, Nspokes)
    for s in 1:Nspokes
        traj[1, :, s] = r .* cos(angles[s])
        traj[2, :, s] = r .* sin(angles[s])
    end
    delay_true = (0.02, -0.015)
    traj_d = copy(traj)
    for s in 1:Nspokes
        traj_d[1, :, s] .+= delay_true[1] * cos(angles[s])
        traj_d[2, :, s] .+= delay_true[2] * sin(angles[s])
    end
    ksp = zeros(ComplexF64, Nsamples, Nspokes, Nc)
    for s in 1:Nspokes, c in 1:Nc
        sh = delay_true[1] * cos(angles[s]) + delay_true[2] * sin(angles[s])
        ksp[:, s, c] = (0.8 + 0.4c / Nc) .* exp.(-50 .* (r .- sh) .^ 2)
    end
    acq = NonCartesianAcquisitionInfo(ksp; trajectory = traj_d, image_size = (64, 64))
    d = estimate_gradient_delays(acq; method = OpposingSpokes())
    @test isapprox(d[1], delay_true[1]; atol = 2.0e-3)
    @test isapprox(d[2], delay_true[2]; atol = 2.0e-3)
end

@testitem "Gradient delay correction in non-Cartesian MRI" tags = [:preprocessing, :acquisition, :nfft] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: NonCartesianAcquisitionInfo
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
