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

@testitem "Coil compression with SVDCompression" tags = [:acquisition, :encoding] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nc = 8
    Nv = 4
    Nx, Ny = 16, 16
    img = NamedDimsArray{(:x, :y)}(randn(ComplexF32, Nx, Ny))

    sens = NamedDimsArray{(:x, :y, :coil)}(zeros(ComplexF32, Nx, Ny, Nc))
    X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
    Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
    for c in 1:Nc
        angle = (c - 1) * 2π / Nc
        sens[:, :, c] = exp.(-((X .- cos(angle) / 2) .^ 2 .+ (Y .- sin(angle) / 2) .^ 2)) .* cis.(0.5f0 .* (X .* cos(angle) .+ Y .* sin(angle)))
    end
    rss = sqrt.(sum(abs2.(unname(sens)), dims = 3))
    sens ./= (rss .+ 1.0f-8)

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
    rec_orig = reconstruct(acq_sim, DirectReconstruction(); verbose = false)
    rec_comp = reconstruct(acq_comp, DirectReconstruction(); verbose = false)
    @test isapprox(rec_comp, rec_orig; rtol = 0.05)
end

@testitem "Sensitivity map estimation: SelfCalibrating, AdaptiveCombine, ESPIRiT" tags = [:acquisition, :encoding, :simulation] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nc = 32, 32, 4
    img = NamedDimsArray{(:x, :y)}(zeros(ComplexF32, Nx, Ny))
    img[8:24, 8:24] .= 1.0f0

    sens_true = NamedDimsArray{(:x, :y, :coil)}(zeros(ComplexF32, Nx, Ny, Nc))
    X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
    Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
    for c in 1:Nc
        angle = (c - 1) * 2π / Nc
        sens_true[:, :, c] = exp.(-((X .- cos(angle) / 2) .^ 2 .+ (Y .- sin(angle) / 2) .^ 2)) .* cis.(0.5f0 .* (X .* cos(angle) .+ Y .* sin(angle)))
    end
    rss_true = sqrt.(sum(abs2.(unname(sens_true)), dims = 3))
    sens_true ./= (rss_true .+ 1.0f-8)

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
    rec_selfcal = reconstruct(acq_selfcal, DirectReconstruction(); verbose = false)
    rec_adaptive = reconstruct(acq_adaptive, DirectReconstruction(); verbose = false)
    rec_espirit = reconstruct(acq_espirit, DirectReconstruction(); verbose = false)

    @test isapprox(abs.(unname(rec_selfcal))[mask], abs.(unname(img))[mask]; rtol = 0.15)
    @test isapprox(abs.(unname(rec_adaptive))[mask], abs.(unname(img))[mask]; rtol = 0.15)
    @test isapprox(abs.(unname(rec_espirit))[mask], abs.(unname(img))[mask]; rtol = 0.15)
end
