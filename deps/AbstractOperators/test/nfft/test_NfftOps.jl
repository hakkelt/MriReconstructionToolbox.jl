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
        op = NFFTOp(image_size, trajectory; threaded)
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
