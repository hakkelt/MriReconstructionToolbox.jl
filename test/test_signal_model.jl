@testitem "Signal model operator: TemporalBasis forward and adjoint" tags = [:encoding, :reconstruction] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nt, K = 8, 8, 5, 3
    Φ = randn(ComplexF32, Nt, K)

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :time)}(zeros(ComplexF32, Nx, Ny, Nt));
        is3D = false,
    )

    model = TemporalBasis(Φ; time_dim = :time)
    ℳ = signal_model_operator(model, acq)
    @test !isnothing(ℳ)
    @test ℳ isa MriReconstructionToolbox.NamedDimsOp

    # Check forward evaluation
    c = NamedDimsArray{(:x, :y, :coeff)}(randn(ComplexF32, Nx, Ny, K))
    img = ℳ * c
    @test size(img) == (Nx, Ny, Nt)
    @test dimnames(img) == (:x, :y, :time)

    c_mat = reshape(unname(c), Nx * Ny, K)
    expected_img = reshape(c_mat * Matrix(transpose(Φ)), Nx, Ny, Nt)
    @test isapprox(unname(img), expected_img; rtol = 1.0e-5, atol = 1.0e-5)

    # Check adjoint evaluation
    c_adj = ℳ' * img
    @test size(c_adj) == (Nx, Ny, K)
    @test dimnames(c_adj) == (:x, :y, :coeff)
    expected_c_adj = reshape(reshape(unname(img), Nx * Ny, Nt) * conj(Φ), Nx, Ny, K)
    @test isapprox(unname(c_adj), expected_c_adj; rtol = 1.0e-5, atol = 1.0e-5)
end

@testitem "build_encoding_operator with signal model" tags = [:encoding] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nt, K = 8, 8, 4, 2
    Φ = randn(ComplexF32, Nt, K)

    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :time)}(zeros(ComplexF32, Nx, Ny, Nt));
        is3D = false,
    )

    method_nomodel = IterativeReconstruction()
    𝒜_nomodel = build_encoding_operator(acq, method_nomodel)
    @test 𝒜_nomodel isa MriReconstructionToolbox.NamedDimsOp

    method_model = IterativeReconstruction(signal_model = TemporalBasis(Φ; time_dim = :time))
    𝒜_model = build_encoding_operator(acq, method_model)
    @test 𝒜_model isa MriReconstructionToolbox.NamedDimsOp
    @test dimnames(𝒜_model, 2) == (:x, :y, :coeff)
    @test dimnames(𝒜_model, 1) == (:kx, :ky, :time)
end

@testitem "KSpaceToImage signal model: dim queries and encoding operator" tags = [:encoding, :reconstruction] begin
    using Test
    using MriReconstructionToolbox
    using NamedDims
    const MRT = MriReconstructionToolbox

    Nx, Ny, Nc = 8, 8, 4
    ksp = NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, Nx, Ny, Nc))
    acq = CartesianAcquisitionInfo(ksp; is3D = false, image_size = (Nx, Ny))

    @test KSpaceToImage().coil_combination === RootSumSquares()

    m_rss = IterativeReconstruction(signal_model = KSpaceToImage(RootSumSquares()))
    @test MRT.variable_dims(m_rss, acq) == (:kx, :ky, :coil)
    @test MRT.variable_size(m_rss, acq) == (Nx, Ny, Nc)
    @test MRT.output_dims(m_rss, acq) == (:x, :y)

    m_nocc = IterativeReconstruction(signal_model = KSpaceToImage(NoCoilCombination()))
    @test MRT.output_dims(m_nocc, acq) == (:x, :y, :coil)

    # Fully sampled ⇒ identity encoding operator (variable is k-space, no subsampling)
    𝒜 = build_encoding_operator(acq, m_rss)
    @test 𝒜 isa MriReconstructionToolbox.NamedDimsOp
    @test dimnames(𝒜, 1) == (:kx, :ky, :coil)
    @test dimnames(𝒜, 2) == (:kx, :ky, :coil)
end

@testitem "Subspace reconstruction with TemporalBasis identity and permutation" tags = [:reconstruction, :minimizer] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims

    Nx, Ny, Nt = 8, 8, 4
    img_true = NamedDimsArray{(:x, :y, :time)}(randn(ComplexF32, Nx, Ny, Nt))
    ksp_dummy = NamedDimsArray{(:kx, :ky, :time)}(zeros(ComplexF32, Nx, Ny, Nt))
    acq = CartesianAcquisitionInfo(ksp_dummy; is3D = false)
    acq_data = simulate_acquisition(img_true, acq)

    # 1. Identity basis: subspace reconstruction should match no-model reconstruction exactly
    Φ_eye = Matrix{ComplexF32}(I, Nt, Nt)
    method_eye = IterativeReconstruction(;
        algorithm = CGNR(maxit = 20, tol = 1.0e-6),
        signal_model = TemporalBasis(Φ_eye; time_dim = :time),
    )
    rec_eye = reconstruct(acq_data, method_eye; verbose = false)
    @test isapprox(rec_eye, img_true; rtol = 1.0e-4, atol = 1.0e-4)

    # 2. Permutation basis: reconstruct with permuted basis
    perm = [2, 3, 4, 1]
    Φ_perm = Φ_eye[perm, :]
    method_perm = IterativeReconstruction(;
        algorithm = CGNR(maxit = 20, tol = 1.0e-6),
        signal_model = TemporalBasis(Φ_perm; time_dim = :time),
    )
    rec_perm = reconstruct(acq_data, method_perm; verbose = false)
    @test isapprox(rec_perm, img_true; rtol = 1.0e-4, atol = 1.0e-4)
end

@testitem "Signal model: non-trailing time dimension" tags = [:reconstruction, :minimizer] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using NamedDims
    using FFTW

    Nx, Ny, Nt, Nsl, K = 12, 12, 6, 3, 3
    Φ = Matrix(qr(randn(ComplexF64, Nt, K)).Q[:, 1:K])
    coeff = randn(ComplexF64, Nx, Ny, K, Nsl)
    imgs = similar(coeff, Nx, Ny, Nt, Nsl)
    for x in 1:Nx, y in 1:Ny, s in 1:Nsl
        imgs[x, y, :, s] = Φ * coeff[x, y, :, s]
    end
    img_true = NamedDimsArray{(:x, :y, :time, :slice)}(imgs)
    ksp_dummy = NamedDimsArray{(:kx, :ky, :time, :slice)}(zeros(ComplexF64, Nx, Ny, Nt, Nsl))
    acq = simulate_acquisition(img_true, CartesianAcquisitionInfo(ksp_dummy; is3D = false))

    method = IterativeReconstruction(;
        algorithm = CGNR(maxit = 30, tol = 1.0e-8),
        signal_model = TemporalBasis(Φ; time_dim = :time),   # not the last image dim
    )
    # 1. operator forward matches the manual Φ expansion along a non-trailing axis
    op = MriReconstructionToolbox.signal_model_operator(method, acq)
    c = NamedDimsArray{(:x, :y, :coeff, :slice)}(randn(ComplexF64, Nx, Ny, K, Nsl))
    y = op * c
    man = similar(unname(c), Nx, Ny, Nt, Nsl)
    for x in 1:Nx, yy in 1:Ny, s in 1:Nsl
        man[x, yy, :, s] = Φ * unname(c)[x, yy, :, s]
    end
    @test unname(y) ≈ man

    # 2. full reconstruction over an extra batch dim (problem decomposition + shape-changing model)
    rec = reconstruct(acq, method; verbose = false)
    @test size(rec) == (Nx, Ny, Nt, Nsl)
    @test dimnames(rec) == (:x, :y, :time, :slice)
    @test norm(unname(rec) .- imgs) / norm(imgs) < 1.0e-3
end
