using TestItems

@testitem "StructuredLowRank regularization" tags = [:regularization] setup = [RegTestSetup, ProxOf] begin
    using LinearAlgebra
    import Random
    Random.seed!(0)

    @testset "Constructor" begin
        reg = StructuredLowRank(λ = 0.1, window = (5, 5))
        @test reg.λ == 0.1
        @test reg.max_rank === nothing
        @test reg.window == (5, 5)
        @test reg.structure === :c
        @test reg.batch_dims === nothing

        reg2 = StructuredLowRank(max_rank = 8, window = (4, 4, 3), batch_dims = (:time,))
        @test reg2.λ === nothing
        @test reg2.max_rank == 8
        @test reg2.batch_dims == (:time,)

        # `λ` and `max_rank` are mutually exclusive (NAMING.md rule 1.3)
        @test_throws ArgumentError StructuredLowRank(window = (5, 5))
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, max_rank = 4, window = (5, 5))
        @test_throws ArgumentError StructuredLowRank(λ = -0.1, window = (5, 5))
        @test_throws ArgumentError StructuredLowRank(max_rank = 0, window = (5, 5))
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5,))
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 0))
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 5), structure = :s)
    end

    @testset "get_operator is the identity" for threaded in [false, true]
        x = randn(ComplexF64, 12, 10, 4)
        op = get_operator(StructuredLowRank(λ = 0.1, window = (4, 3)), x; threaded)
        @test op * x ≈ x
        named = NamedDimsArray{(:kx, :ky, :coil)}(x)
        @test get_operator(StructuredLowRank(λ = 0.1, window = (4, 3)), named; threaded) isa
            MriReconstructionToolbox.NamedDimsOp
    end

    @testset "get_affected_dims" begin
        reg = StructuredLowRank(λ = 0.1, window = (4, 3))
        @test get_affected_dims(reg, nothing, (:kx, :ky, :coil)) == (:kx, :ky, :coil)
        reg_t = StructuredLowRank(λ = 0.1, window = (4, 3), batch_dims = (:time,))
        @test get_affected_dims(reg_t, nothing, (:kx, :ky, :coil, :time)) == (:kx, :ky, :coil)
    end

    @testset "scale_regularization" begin
        # the nuclear norm is homogeneous of degree 1, a rank cap is scale-invariant
        @test scale_regularization(StructuredLowRank(λ = 0.2, window = (4, 3)), 3.0).λ ≈ 0.6
        scaled = scale_regularization(StructuredLowRank(max_rank = 5, window = (4, 3)), 3.0)
        @test scaled.max_rank == 5
        @test scaled.λ === nothing
    end

    @testset "materialize both forms" for threaded in [false, true]
        for w in ((4, 3), (3, 3, 2))
            dims = length(w) == 2 ? (12, 10, 4) : (8, 7, 6, 3)
            x = Variable(randn(ComplexF64, dims...))
            @test materialize(StructuredLowRank(λ = 0.05, window = w), x; threaded) !== nothing
            @test materialize(StructuredLowRank(max_rank = 4, window = w), x; threaded) !== nothing
        end
    end

    @testset "materialize rejects an input that cannot be lifted" begin
        # fewer dims than the window plus a channel axis
        @test_throws ArgumentError materialize(
            StructuredLowRank(λ = 0.1, window = (4, 3)), Variable(randn(ComplexF64, 12, 10)); threaded = false
        )
        # window larger than the k-space grid
        @test_throws ArgumentError materialize(
            StructuredLowRank(λ = 0.1, window = (20, 3)), Variable(randn(ComplexF64, 12, 10, 2)); threaded = false
        )
    end

    @testset "nuclear-norm prox == Cadzow reference" begin
        gx, gy, nc = 12, 10, 3
        w = (4, 3)
        x = randn(ComplexF64, gx, gy, nc)
        γ, λ = 0.7, 0.15
        y, _ = prox_of(StructuredLowRank(λ = λ, window = w), x, γ)

        H = AbstractOperators.Hankel(ComplexF64, (gx, gy), w; nchannels = nc, channels = true)
        M = H * x
        F = svd(M)
        S = max.(0.0, F.S .- λ * γ)
        Mhat = F.U * Diagonal(S) * F.Vt
        invmult = 1.0 ./ real.(AbstractOperators.diag_AcA(H))
        ref = (H' * Mhat) .* invmult
        @test y ≈ ref
    end

    @testset "rank prox == hard-truncation Cadzow reference" begin
        gx, gy, nc = 16, 14, 2
        w = (5, 4)
        x = randn(ComplexF64, gx, gy, nc)
        r = 6
        y, _ = prox_of(StructuredLowRank(max_rank = r, window = w), x, 1.0)

        H = AbstractOperators.Hankel(ComplexF64, (gx, gy), w; nchannels = nc, channels = true)
        F = svd(H * x)
        S = copy(F.S)
        S[(r + 1):end] .= 0
        @test count(!iszero, S) == r                        # the lifted matrix was truncated to rank r
        Mhat = F.U * Diagonal(S) * F.Vt
        invmult = 1.0 ./ real.(AbstractOperators.diag_AcA(H))
        ref = (H' * Mhat) .* invmult
        @test y ≈ ref
    end

    @testset "the prox is threading-invariant" begin
        x = randn(ComplexF64, 12, 10, 3, 2)
        seq, _ = prox_of(StructuredLowRank(λ = 0.1, window = (4, 3)), x, 0.5)
        # `prox_of` builds the term with threaded = false; rebuild it threaded and compare
        term = materialize(StructuredLowRank(λ = 0.1, window = (4, 3)), Variable(x); threaded = true)
        f = MriReconstructionToolbox.StructuredOptimization.extract_functions(term)
        par = similar(x)
        MriReconstructionToolbox.ProximalCore.prox!(par, f, x, 0.5)
        @test par ≈ seq
    end

    @testset "calculate: nuclear norm of the lifted matrix" begin
        gx, gy, nc = 12, 10, 3
        w = (4, 3)
        x = randn(ComplexF64, gx, gy, nc)
        λ = 0.1
        v = calculate(StructuredLowRank(λ = λ, window = w), x; threaded = false)
        H = AbstractOperators.Hankel(ComplexF64, (gx, gy), w; nchannels = nc, channels = true)
        @test v ≈ λ * sum(svdvals(H * x))
    end

    @testset "calculate: the rank form is an indicator" begin
        # a single 2D complex exponential per coil lifts to a rank-1 block-Hankel matrix
        gx, gy, nc = 12, 10, 2
        w = (4, 3)
        x = ComplexF64[cispi(0.13 * (i - 1) + 0.21 * (j - 1)) * (c + 1) for i in 1:gx, j in 1:gy, c in 1:nc]
        @test calculate(StructuredLowRank(max_rank = 1, window = w), x; threaded = false) == 0
        noisy = x .+ 0.5 .* randn(ComplexF64, gx, gy, nc)
        @test calculate(StructuredLowRank(max_rank = 1, window = w), noisy; threaded = false) == Inf
    end
end

@testitem "StructuredLowRank end-to-end calibrationless" tags = [:regularization, :reconstruction, :integration] begin
    using MriReconstructionToolbox
    using LinearAlgebra, NamedDims
    import Random
    Random.seed!(1)

    Nx, Ny, Nc = 24, 24, 6
    img = zeros(ComplexF64, Nx, Ny)
    img[7:18, 7:18] .= 1.0
    img[10:14, 10:14] .= 0.5
    sens = NamedDimsArray{(:x, :y, :coil)}(ComplexF64.(coil_sensitivities(Nx, Ny, Nc)))
    patt = create_sampling_pattern(UniformRandomSampling(2.0, 0.15), (Nx, Ny))

    # Simulate through the package's own forward model so that the amplitude convention of the
    # k-space matches what `reconstruct` inverts.
    acq_sim = CartesianAcquisitionInfo(;
        is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens, subsampling = patt,
    )
    ksp = simulate_acquisition(NamedDimsArray{(:x, :y)}(img), acq_sim).kspace_data

    # The reconstruction gets the coil data and nothing else: no sensitivity maps, no ACS
    # kernel. That is what makes this calibrationless.
    acq = CartesianAcquisitionInfo(
        ksp; is3D = false, image_size = (Nx, Ny), subsampling = patt,
    )

    rss(x) = sqrt.(sum(abs2, x; dims = 3))[:, :, 1]
    nrmse(a, b) = norm(a - b) / norm(b)
    # Reference magnitude and the zero-filled baseline, both in the package's own scale.
    acq_full = CartesianAcquisitionInfo(;
        is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens,
    )
    ksp_full = simulate_acquisition(NamedDimsArray{(:x, :y)}(img), acq_full).kspace_data
    truth = rss(
        abs.(
            unname(
                reconstruct(
                    CartesianAcquisitionInfo(ksp_full; is3D = false, image_size = (Nx, Ny)),
                    DirectReconstruction(); verbosity = Silent(),
                )
            )
        )
    )
    zf = rss(abs.(unname(reconstruct(acq, DirectReconstruction(); verbosity = Silent()))))

    solve(reg) = abs.(
        unname(
            reconstruct(
                acq,
                IterativeReconstruction(
                    reg; signal_model = KSpaceToImage(RootSumSquares()), algorithm = ADMM(), maxit = 100,
                );
                verbosity = Silent(),
            )
        )
    )

    e_zf = nrmse(zf, truth)
    # LORAKS-C: the convex nuclear-norm form
    e_nuc = nrmse(solve(StructuredLowRank(λ = 0.03, window = (5, 5))), truth)
    @test e_nuc < 0.08
    @test e_nuc < e_zf
    # SAKE: the non-convex hard-rank form, from the same zero-filled starting point
    e_rank = nrmse(solve(StructuredLowRank(max_rank = 30, window = (5, 5))), truth)
    @test e_rank < 0.08
    @test e_rank < e_zf
end
