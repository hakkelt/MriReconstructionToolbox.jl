using TestItems

@testitem "TotalGeneralizedVariation2D regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using MriReconstructionToolbox
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

@testitem "TotalGeneralizedVariation2D denoising behaviour" tags = [:regularization, :minimizer] begin
    using Test
    using LinearAlgebra
    using Random
    using MriReconstructionToolbox
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
    relative_error(z) = norm(z .- truth) / norm(truth)

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
        stacked = cat(noisy, reverse(noisy; dims = 1); dims = 3)
        result = denoise(TotalGeneralizedVariation2D(0.05), stacked; maxit = 1500)
        single = denoise(TotalGeneralizedVariation2D(0.05), noisy; maxit = 1500)
        @test result[:, :, 1] ≈ single rtol = 2.0e-2
        @test result[:, :, 2] ≈ reverse(single; dims = 1) rtol = 2.0e-2
        # The second slice is the first one mirrored, so the two must be denoised to mirror images of each
        # other -- that part is exact, since it is the same problem twice.
        @test result[:, :, 2] ≈ reverse(result[:, :, 1]; dims = 1) rtol = 1.0e-6
    end
end

@testitem "Infimal-convolution total variation via components" tags = [:regularization, :components] begin
    using Test
    using LinearAlgebra
    using Random
    using MriReconstructionToolbox
    using AbstractOperators

    const MRT = MriReconstructionToolbox

    # The infimal convolution of first- and second-order TV needs no regularization type of its own: it is
    # exactly an image decomposition into a piecewise-constant component and a piecewise-linear one, which
    # the component machinery already expresses.
    n = 32
    truth = [(i > n ÷ 2 ? 1.0 : 0.0) + 0.02 * j for i in 1:n, j in 1:n]
    noisy = truth .+ 0.05 .* randn(MersenneTwister(3), n, n)
    relative_error(z) = norm(z .- truth) / norm(truth)

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
