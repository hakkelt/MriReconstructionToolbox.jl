using TestItems

@testitem "TotalGeneralizedVariation2D regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_operator, get_encoding_operator, materialize, get_affected_dims, scale_regularization
    using StructuredOptimization
    using AbstractOperators
    using NamedDims

    const MRT = MriReconstructionToolbox

    @testset "Constructor" begin
        reg = TotalGeneralizedVariation2D(0.1)
        @test reg.λ == 0.1
        @test reg.ratio == 2.0
        @test TotalGeneralizedVariation2D(0.1; ratio = 3.0).ratio == 3.0
        @test_throws ArgumentError TotalGeneralizedVariation2D(-0.1)
        @test_throws ArgumentError TotalGeneralizedVariation2D(0.1; ratio = 0)
    end

    @testset "get_operator is the flattened gradient" for threaded in [false, true]
        x = randn(8, 8)
        op = get_operator(TotalGeneralizedVariation2D(0.1), x; threaded)
        @test size(op, 1) == (64, 2)
        @test op * x ≈ reshape(get_operator(TotalVariation2D(0.1), x; threaded) * x, 64, 2)
        @test_throws ArgumentError get_operator(TotalGeneralizedVariation2D(0.1), randn(8); threaded)
    end

    @testset "materialize introduces one auxiliary field" begin
        x = Variable(randn(8, 8, 3))
        terms, auxiliaries = MRT.materialize_with_auxiliaries(
            TotalGeneralizedVariation2D(0.1), x; threaded = false
        )
        @test length(auxiliaries) == 1
        w = auxiliaries[1]
        # one vector per voxel of the whole array, in `Variation`'s layout
        @test size(~w) == (8 * 8 * 3, 2)
        @test all(iszero, ~w)   # auxiliaries start at zero
        @test terms isa MRT.StructuredOptimization.TermSet
        # `materialize` gives the same terms without the auxiliaries
        @test MRT.materialize(TotalGeneralizedVariation2D(0.1), x; threaded = false) isa
            MRT.StructuredOptimization.TermSet
    end

    @testset "the symmetrized-gradient operator acts per batch slice" begin
        # Folding the batch into a spatial extent would let differences run across slice boundaries; this
        # checks that they do not.
        batched = MRT._tgv_symmetrized_operator(Float64, (8, 8), 3; threaded = false)
        single = SymmetrizedVariation(Float64, (8, 8); threaded = false)
        w = randn(8 * 8 * 3, 2)
        result = batched * w
        unfolded = reshape(w, 64, 3, 2)
        for k in 1:3
            @test result[((k - 1) * 64 + 1):(k * 64), :] ≈ single * unfolded[:, k, :]
        end
        # and it is still a correct adjoint pair after all the reshaping and permuting
        y = randn(8 * 8 * 3, 3)
        @test dot(batched * w, y) ≈ dot(w, batched' * y)
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 8, 8, 4)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test MRT.get_affected_dims(TotalGeneralizedVariation2D(0.1f0), info, 1:3) == 1:2
    end

    @testset "scale_regularization" begin
        reg = MRT.scale_regularization(TotalGeneralizedVariation2D(0.2; ratio = 3.0), 2.5)
        @test reg.λ ≈ 0.5
        @test reg.ratio == 3.0
    end
end

@testitem "TotalGeneralizedVariation2D denoising behaviour" tags = [:regularization, :minimizer] setup = [TestHelpers] begin
    using Test
    using LinearAlgebra
    using Random
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_operator, get_encoding_operator, materialize, get_affected_dims, scale_regularization
    using StructuredOptimization
    using AbstractOperators

    const MRT = MriReconstructionToolbox

    function denoise(reg, noisy; maxit = 2000)
        model, x, _ = MRT.build_model_with_variables(
            Eye(noisy), noisy, (reg,);
            threaded = false, x₀ = copy(noisy), disable_normalop_optimization = true,
        )
        solve(model, ADMM(; maxit, rho = 1.0))
        return copy(~x)
    end

    # A ramp with a jump in it: the case first-order TV gets wrong (staircasing on the ramp) and
    # second-order TV gets wrong (blurring across the jump).
    n = 32
    truth = [(i > n ÷ 2 ? 1.0 : 0.0) + 0.02 * j for i in 1:n, j in 1:n]
    noisy = truth .+ 0.05 .* randn(MersenneTwister(2), n, n)
    relative_error(z) = relative_error(z, truth)

    @testset "TGV beats TV on a ramp with an edge" begin
        tv = denoise(TotalVariation2D(0.05), noisy)
        tgv = denoise(TotalGeneralizedVariation2D(0.05), noisy)
        @test relative_error(tgv) < relative_error(tv)
        @test relative_error(tgv) < relative_error(noisy)
    end

    @testset "a large ratio degenerates to total variation" begin
        # Driving the second-order weight up forces the auxiliary field to a constant, which leaves exactly
        # the total variation term -- TV is a limiting case of TGV, not a different model.
        tv = denoise(TotalVariation2D(0.05), noisy)
        tgv = denoise(TotalGeneralizedVariation2D(0.05; ratio = 1.0e4), noisy)
        @test norm(tgv .- tv) / norm(tv) < 0.05
    end

    @testset "batch dimensions are denoised independently" begin
        # Both the data term and the penalty are separable across the batch, so the exact minimizers
        # coincide slice by slice. Only ADMM's finite iteration budget separates them — the two problems are
        # of different size, so the solver does not take identical steps — hence the loose tolerance.
        # The second slice is a different image, so a leak across the batch boundary would show up as a
        # disagreement with the slice reconstructed on its own.
        other = [0.03 * i for i in 1:n, _ in 1:n] .+ 0.05 .* randn(MersenneTwister(4), n, n)
        stacked = cat(noisy, other; dims = 3)
        result = denoise(TotalGeneralizedVariation2D(0.05), stacked; maxit = 1500)
        @test result[:, :, 1] ≈ denoise(TotalGeneralizedVariation2D(0.05), noisy; maxit = 1500) rtol = 2.0e-2
        @test result[:, :, 2] ≈ denoise(TotalGeneralizedVariation2D(0.05), other; maxit = 1500) rtol = 2.0e-2
    end
end

@testitem "Infimal-convolution total variation via components" tags = [:regularization, :components] setup = [TestHelpers] begin
    using Test
    using LinearAlgebra
    using Random
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_operator, get_encoding_operator, materialize, get_affected_dims, scale_regularization
    using StructuredOptimization
    using AbstractOperators

    const MRT = MriReconstructionToolbox

    # The infimal convolution of first- and second-order TV needs no regularization type of its own: it is
    # exactly an image decomposition into a piecewise-constant component and a piecewise-linear one, which
    # the component machinery already expresses.
    n = 32
    truth = [(i > n ÷ 2 ? 1.0 : 0.0) + 0.02 * j for i in 1:n, j in 1:n]
    noisy = truth .+ 0.05 .* randn(MersenneTwister(3), n, n)
    relative_error(z) = relative_error(z, truth)

    components = (
        Component(:cartoon, TotalVariation2D(0.05)),
        Component(:ramp, SecondOrderTotalVariation2D(0.05)),
    )

    model, vars, auxiliaries = MRT.build_model(
        Eye(noisy), noisy, components; threaded = false, x₀s = (copy(noisy), zero(noisy))
    )
    @test auxiliaries == ()
    solve(model, ADMM(maxit = 1000, rho = 1.0))
    total = reduce(+, map(v -> copy(~v), vars))

    @test relative_error(total) < relative_error(noisy)

    # It must also beat plain first-order TV on this ramp-plus-edge image, which is the whole point of
    # splitting the image into a cartoon and a ramp part.
    tv_model, tv_x, _ = MRT.build_model_with_variables(
        Eye(noisy), noisy, (TotalVariation2D(0.05),);
        threaded = false, x₀ = copy(noisy), disable_normalop_optimization = true,
    )
    solve(tv_model, ADMM(maxit = 1000, rho = 1.0))
    @test relative_error(total) < relative_error(copy(~tv_x))
end

@testitem "TotalGeneralizedVariation2D runs through reconstruct" tags = [:regularization, :integration] begin
    using Test
    using LinearAlgebra
    using GeometricMedicalPhantoms
    using Random
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using StructuredOptimization

    # Regression: TGV was only ever exercised through hand-built models passed to `solve`, so the
    # public entry point had no coverage. `build_model_with_variables` chose the `normalop_ls` data
    # term before the regularizations declared their auxiliary variables, and the operator that form
    # stores spans the image alone -- so once TGV added its auxiliary field the solver's `x0` spanned
    # (image, auxiliary) while the stored operator did not, and ADMM rejected the pair with
    # "A'b must have the same size as x0".
    Random.seed!(20260829)
    nx, ny, nc = 32, 32, 4
    img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
    smaps = coil_sensitivities(nx, ny, nc)
    pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15), (nx, ny))
    acq = simulate_acquisition(img_true, AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern))

    img_recon = reconstruct(
        acq,
        IterativeReconstruction(TotalGeneralizedVariation2D(0.005); algorithm = ADMM(rho = 1.0), maxit = 50); verbosity = Silent()
    )
    @test size(img_recon) == (nx, ny)
    @test all(isfinite, img_recon)
    # A sign-flipped or diverged solve lands near 2.0; measured ≈0.29 at 50 iterations.
    @test norm(img_recon - img_true) / norm(img_true) < 0.6

    # The auxiliary variable is what forces the plain `ls` form; a term without one must keep the
    # `normalop_ls` optimization.
    _, _, tgv_aux = MriReconstructionToolbox.build_model_with_variables(
        get_encoding_operator(acq), acq.kspace_data, (TotalGeneralizedVariation2D(0.005),); threaded = false,
    )
    @test length(tgv_aux) == 1
    tv_terms, _, tv_aux = MriReconstructionToolbox.build_model_with_variables(
        get_encoding_operator(acq), acq.kspace_data, (TotalVariation2D(0.001),); threaded = false,
    )
    @test tv_aux == ()
    @test any(t -> t.f isa MriReconstructionToolbox.StructuredOptimization.SqrNormL2WithNormalOp, tv_terms)
end
@testitem "TotalGeneralizedVariation3D regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_operator, materialize, get_affected_dims, scale_regularization
    using StructuredOptimization
    using AbstractOperators

    const MRT = MriReconstructionToolbox

    @testset "Constructor" begin
        reg = TotalGeneralizedVariation3D(0.1)
        @test reg.λ == 0.1
        @test reg.ratio == 2.0
        @test TotalGeneralizedVariation3D(0.1; ratio = 3.0).ratio == 3.0
        @test_throws ArgumentError TotalGeneralizedVariation3D(-0.1)
        @test_throws ArgumentError TotalGeneralizedVariation3D(0.1; ratio = 0)
    end

    @testset "get_operator is the flattened 3D gradient" for threaded in [false, true]
        x = randn(6, 6, 6)
        op = get_operator(TotalGeneralizedVariation3D(0.1), x; threaded)
        @test size(op, 1) == (216, 3)
        @test op * x ≈ reshape(get_operator(TotalVariation3D(0.1), x; threaded) * x, 216, 3)
        @test_throws ArgumentError get_operator(TotalGeneralizedVariation3D(0.1), randn(6, 6); threaded)
    end

    @testset "materialize introduces one auxiliary field with three components" begin
        x = Variable(randn(4, 4, 4, 2))
        terms, auxiliaries = MRT.materialize_with_auxiliaries(
            TotalGeneralizedVariation3D(0.1), x; threaded = false
        )
        @test length(auxiliaries) == 1
        w = auxiliaries[1]
        @test size(~w) == (4 * 4 * 4 * 2, 3)
        @test all(iszero, ~w)
        @test terms isa MRT.StructuredOptimization.TermSet
    end

    @testset "the symmetrized-gradient operator acts per batch slice" begin
        # Six independent components in 3D: the entries of the symmetric 3x3 matrix ℰw.
        batched = MRT._tgv_symmetrized_operator(Float64, (4, 4, 4), 2; threaded = false)
        single = SymmetrizedVariation(Float64, (4, 4, 4); threaded = false)
        w = randn(64 * 2, 3)
        result = batched * w
        @test size(result) == (128, 6)
        unfolded = reshape(w, 64, 2, 3)
        for k in 1:2
            @test result[((k - 1) * 64 + 1):(k * 64), :] ≈ single * unfolded[:, k, :]
        end
        y = randn(64 * 2, 6)
        @test dot(batched * w, y) ≈ dot(w, batched' * y)
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 6, 6, 6, 4)
        info = AcquisitionInfo(ksp; image_size = (6, 6, 6))
        @test MRT.get_affected_dims(TotalGeneralizedVariation3D(0.1f0), info, 1:4) == 1:3
    end

    @testset "scale_regularization" begin
        reg = MRT.scale_regularization(TotalGeneralizedVariation3D(0.2; ratio = 3.0), 2.5)
        @test reg.λ ≈ 0.5
        @test reg.ratio == 3.0
    end
end

@testitem "TotalGeneralizedVariation3D denoising behaviour" tags = [:regularization, :minimizer] setup = [TestHelpers] begin
    using Test
    using LinearAlgebra
    using Random
    using MriReconstructionToolbox
    using StructuredOptimization
    using AbstractOperators

    const MRT = MriReconstructionToolbox

    function denoise(reg, noisy; maxit = 800)
        model, x, _ = MRT.build_model_with_variables(
            Eye(noisy), noisy, (reg,);
            threaded = false, x₀ = copy(noisy), disable_normalop_optimization = true,
        )
        solve(model, ADMM(; maxit, rho = 1.0))
        return copy(~x)
    end

    # A volumetric ramp with a jump: the 3D analogue of the 2D staircasing case.
    n = 16
    truth = [(i > n ÷ 2 ? 1.0 : 0.0) + 0.02 * j + 0.01 * k for i in 1:n, j in 1:n, k in 1:n]
    noisy = truth .+ 0.05 .* randn(MersenneTwister(5), n, n, n)
    relative_error(z) = relative_error(z, truth)

    @test relative_error(denoise(TotalGeneralizedVariation3D(0.05), noisy)) <
        relative_error(denoise(TotalVariation3D(0.05), noisy))
end
