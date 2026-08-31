@testitem "Pseudo-replica analysis and g-factor maps" tags = [:reconstruction, :acquisition] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims
    using Random

    Nx, Ny = 16, 16
    img = NamedDimsArray{(:x, :y)}(ones(ComplexF32, Nx, Ny))
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky)}(zeros(ComplexF32, Nx, Ny));
        is3D = false,
    )
    acq_sim = simulate_acquisition(img, acq)

    # 1. Validation tests
    @test_throws ArgumentError pseudo_replica(acq_sim; replicas = 1)
    @test_throws ArgumentError pseudo_replica(acq_sim; replicas = 10, normalization = BartScaling())

    # 2. Fully sampled single-coil Cartesian: g-factor must be approximately 1.0 everywhere
    rng = Random.MersenneTwister(42)
    res = pseudo_replica(acq_sim; replicas = 100, noise_std = 1.0, rng)
    @test haskey(res, :mean)
    @test haskey(res, :std)
    @test haskey(res, :g_factor)
    @test res.mean isa NamedDimsArray
    @test dimnames(res.mean) == (:x, :y)
    @test size(res.g_factor) == (Nx, Ny)

    # Theoretical noise std in image space for 16x16 with 1/sqrt(256)=1/16 scaling is 0.0625
    @test isapprox(unname(res.g_factor), ones(Float32, Nx, Ny); rtol = 0.15)
end

@testitem "Subspace reconstruction: T2 decay simulation with TemporalBasis" tags = [:reconstruction, :simulation] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nt = 16, 16, 12
    K = 3

    # Generate synthetic T2-decay dynamic image series: img(x, y, t) = rho(x, y) * exp(-t / T2(x, y))
    rho = zeros(Float32, Nx, Ny)
    rho[4:12, 4:12] .= 1.0f0
    T2_map = fill(5.0f0, Nx, Ny) # T2 = 5 frames

    img_series = NamedDimsArray{(:x, :y, :time)}(zeros(ComplexF32, Nx, Ny, Nt))
    for t in 1:Nt
        img_series[:, :, t] = rho .* exp.(-(t - 1) ./ T2_map)
    end

    # Build low-dimensional temporal subspace basis from exponential decay dictionary
    T2_candidates = range(1.0f0, 20.0f0, length = 50)
    dict = zeros(Float32, Nt, length(T2_candidates))
    for (i, t2) in enumerate(T2_candidates)
        dict[:, i] = exp.(-(0:(Nt - 1)) ./ t2)
    end
    F_svd = svd(dict)
    Φ = Matrix{ComplexF32}(F_svd.U[:, 1:K]) # (Nt, K)

    # Simulate acquisition
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :time)}(zeros(ComplexF32, Nx, Ny, Nt));
        is3D = false,
    )
    acq_sim = simulate_acquisition(img_series, acq)

    # Reconstruct with subspace model
    method_subspace = IterativeReconstruction(;
        algorithm = CGNR(maxit = 20, tol = 1.0e-5),
        signal_model = TemporalBasis(Φ; time_dim = :time),
    )
    rec = reconstruct(acq_sim, method_subspace; verbose = false)

    @test rec isa NamedDimsArray
    @test dimnames(rec) == (:x, :y, :time)
    @test size(rec) == (Nx, Ny, Nt)

    # Subspace model should reconstruct the series accurately
    mask = abs.(unname(img_series)) .> 0.1
    @test isapprox(unname(rec)[mask], unname(img_series)[mask]; rtol = 0.05)
end
