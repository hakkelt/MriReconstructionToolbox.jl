@testmodule NFFTTestHelper begin
    using Test
    using AbstractOperators
    using LinearAlgebra, Random, NFFT, NFFTOperators

    function test_nufft_op(op, plan, image, dcf)
        ksp1 = similar(image, ComplexF64, size(op, 1))
        mul!(vec(ksp1), plan, image)
        ksp2 = similar(ksp1)
        mul!(ksp2, op, image)
        @test ksp2 == ksp1

        image2 = similar(image)
        if dcf === nothing
            mul!(image2, plan', vec(ksp2 .* op.dcf))
            @test norm(image2 .- image) / norm(image) < 0.5
        else
            image1 = similar(image)
            mul!(image1, plan', vec(ksp1 .*= dcf))
            mul!(image2, op', ksp2)
            @test image2 ≈ image1
        end

        normal_op = AbstractOperators.get_normal_op(op)
        image3 = similar(image)
        mul!(image3, normal_op, image)
        @test image3 ≈ image2
    end

    function test_2d_nufft(threaded)
        trajectory = rand(2, 128, 50) .- 0.5
        dcf = rand(128, 50)
        image_size = (128, 128)
        image = rand(ComplexF64, image_size)
        plan = plan_nfft(reshape(trajectory, 2, :), image_size)
        op = NFFTOp(image_size, trajectory, dcf; threaded)
        test_nufft_op(op, plan, image, dcf)
    end

    function test_3d_nufft(threaded)
        trajectory = rand(3, 128, 50) .- 0.5
        dcf = rand(128, 50)
        image_size = (64, 64, 64)
        image = rand(ComplexF64, image_size)
        plan = plan_nfft(reshape(trajectory, 3, :), image_size)
        op = NFFTOp(image_size, trajectory, dcf; threaded)
        test_nufft_op(op, plan, image, dcf)
    end

    function test_realistic_2d_nufft(threaded)
        trajectory = Array{Float64}(undef, 2, 256, 201)
        ϕstep = 2π / 201
        for i in 1:201
            ϕ = i * ϕstep
            trajectory[1, :, i] = cos(ϕ) .* ((-64:0.5:63.5) ./ 128)
            trajectory[2, :, i] = sin(ϕ) .* ((-64:0.5:63.5) ./ 128)
        end
        image_size = (128, 128)
        image = zeros(ComplexF64, image_size)
        for idx in CartesianIndices(image)
            d = norm([idx[1] - 64, idx[2] - 64])
            if d < 15
                image[idx] = 1.0
            end
        end
        plan = plan_nfft(reshape(trajectory, 2, :), image_size)
        # `:auto` explicitly requests the density-compensated approximate adjoint this test
        # exercises; the bare `dcf`-less call now means "no dcf" (`op.dcf` all ones), which
        # would not round-trip through the adjoint the way this test expects.
        op = NFFTOp(image_size, trajectory, :auto; threaded)
        test_nufft_op(op, plan, image, nothing)
    end

    function test_nfft_normal_op(threaded)
        trajectory = rand(2, 128, 50) .- 0.5
        dcf = rand(128, 50)
        image_size = (128, 128)
        image = rand(ComplexF64, image_size)
        op = NFFTOp(image_size, trajectory, dcf; threaded)
        normal_op = AbstractOperators.get_normal_op(op)

        image_out1 = similar(image)
        mul!(image_out1, normal_op, image)
        ksp = similar(image, ComplexF64, size(op, 1))
        mul!(ksp, op, image)
        image_out2 = similar(image)
        mul!(image_out2, op', ksp)
        @test image_out1 ≈ image_out2
    end
end

@testitem "NFFTOp 2D" tags = [:nfft, :NFFTOp] setup = [TestUtils, NFFTTestHelper] begin
    NFFTTestHelper.test_2d_nufft(false)
    NFFTTestHelper.test_2d_nufft(true)
end

@testitem "NFFTOp realistic 2D" tags = [:nfft, :NFFTOp] setup = [TestUtils, NFFTTestHelper] begin
    NFFTTestHelper.test_realistic_2d_nufft(false)
    NFFTTestHelper.test_realistic_2d_nufft(true)
end

@testitem "NFFTOp 3D" tags = [:nfft, :NFFTOp] setup = [TestUtils, NFFTTestHelper] begin
    NFFTTestHelper.test_3d_nufft(false)
    NFFTTestHelper.test_3d_nufft(true)
end

@testitem "NfftNormalOp" tags = [:nfft, :NfftNormalOp] setup = [TestUtils, NFFTTestHelper] begin
    NFFTTestHelper.test_nfft_normal_op(false)
    NFFTTestHelper.test_nfft_normal_op(true)
end

@testitem "NFFTOp dcf contract" tags = [:nfft, :NFFTOp] setup = [TestUtils] begin
    using AbstractOperators, NFFTOperators, LinearAlgebra, NFFT, NFFTTools, Random
    Random.seed!(0)

    image_size = (32, 32)
    trajectory = rand(2, 64, 8) .- 0.5
    ksp_shape = size(trajectory)[2:end]
    plan = plan_nfft(reshape(trajectory, 2, :), image_size)

    # `dcf = nothing` (the default): no density compensation, `op'` is the true adjoint of `op`.
    op_none = NFFTOp(image_size, trajectory; threaded = false)
    @test op_none.dcf == ones(eltype(trajectory), ksp_shape...)
    image = rand(ComplexF64, image_size)
    ksp = op_none * image
    ksp_ref = similar(ksp)
    mul!(vec(ksp_ref), plan, image)
    @test ksp ≈ ksp_ref
    image_ref = similar(image)
    mul!(image_ref, plan', vec(ksp_ref))  # no dcf weighting: the *true* NFFT adjoint
    @test op_none' * ksp ≈ image_ref

    # `dcf = :auto`: same estimator NFFTOp always ran automatically before this contract existed.
    op_auto = NFFTOp(image_size, trajectory, :auto; threaded = false)
    expected_dcf = reshape(NFFTTools.sdc(plan; iters = 20), ksp_shape)
    @test op_auto.dcf ≈ expected_dcf
    @test !(op_auto.dcf ≈ op_none.dcf)

    # An explicit array is used as given (already covered elsewhere, checked here for parity).
    dcf_arr = rand(ksp_shape...)
    op_arr = NFFTOp(image_size, trajectory, dcf_arr; threaded = false)
    @test op_arr.dcf == dcf_arr

    # An unrecognized Symbol is rejected rather than silently accepted.
    @test_throws ArgumentError NFFTOp(image_size, trajectory, :bogus; threaded = false)

    # dcf_estimation_iterations / dcf_correction_function only take effect for `:auto`.
    op_auto_custom = NFFTOp(
        image_size, trajectory, :auto;
        threaded = false, dcf_estimation_iterations = 5, dcf_correction_function = x -> 2 .* x,
    )
    expected_custom = 2 .* reshape(NFFTTools.sdc(plan; iters = 5), ksp_shape)
    @test op_auto_custom.dcf ≈ expected_custom
end

@testitem "NFFTOp multi-dimensional k-space" tags = [:nfft, :NFFTOp] setup = [TestUtils] begin
    using AbstractOperators, NFFTOperators, LinearAlgebra, NFFT, Random
    Random.seed!(0)

    # A trajectory laid out as (dim, samples, profiles, partitions) gives a 3-D k-space, so
    # `ksp_buffer` and `dcf` are 3-D arrays. The operator's type parameters are bounded by
    # `AbstractArray`, not `AbstractMatrix`, precisely so this shape is representable.
    trajectory = rand(2, 12, 5, 3) .- 0.5
    dcf = rand(12, 5, 3)
    image_size = (16, 16)
    image = rand(ComplexF64, image_size)
    plan = plan_nfft(reshape(trajectory, 2, :), image_size)
    op = NFFTOp(image_size, trajectory, dcf; threaded = false)

    @test size(op, 1) == (12, 5, 3)
    ksp = op * image
    @test size(ksp) == (12, 5, 3)
    ksp_ref = similar(ksp)
    mul!(vec(ksp_ref), plan, image)
    @test ksp == ksp_ref

    # The adjoint is the density-compensated one, as in every other NFFTOp test.
    image_ref = similar(image)
    mul!(image_ref, plan', vec(ksp_ref .* dcf))
    @test op' * ksp ≈ image_ref

    c = copy_operator(op)
    @test c.plan !== op.plan
    @test c * image ≈ ksp
end

@testitem "NFFTOp copy_operator preserves the gridding operating point" tags = [:nfft, :NFFTOp] setup = [TestUtils] begin
    using AbstractOperators, NFFTOperators, LinearAlgebra, NFFT, Random
    Random.seed!(0)

    # A storage-backend change rebuilds the plan from the trajectory. `m`, `σ` and `precompute`
    # are not recoverable from the trajectory, so they have to be carried over from the old plan:
    # rebuilding at the constructor defaults would give the copy a different transform from the
    # one the caller set up, silently.
    trajectory = rand(2, 24, 8) .- 0.5
    image_size = (16, 16)
    image = rand(ComplexF64, image_size)
    op = NFFTOp(image_size, trajectory; threaded = false, m = 3, σ = 1.25, precompute = NFFT.TENSOR)
    @test op.plan.params.m == 3

    c = copy_operator(op; storage_type = Array)
    @test c.plan !== op.plan
    @test c.plan.params.m == op.plan.params.m
    @test c.plan.params.σ == op.plan.params.σ
    @test c.plan.params.precompute == op.plan.params.precompute
    @test c * image ≈ op * image
end
