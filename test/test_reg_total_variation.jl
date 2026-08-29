using TestItems

@testitem "TotalVariation2D Regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "TotalVariation2D Regularization" begin
        @testset "get_operator - 2D input" for threaded in [false, true]
            x = rand(10, 10)
            reg = TotalVariation2D(0.1)
            op = get_operator(reg, x; threaded)
            # Don't assert exact operator type; validate behavior below

            # Test operator application
            result = op * x
            @test length(result) == 2 * length(x)  # Gradient has 2 components
            # Compare to manual finite differences (forward at boundary, backward elsewhere)
            manual = similar(result)
            for i in axes(x, 1), j in axes(x, 2)
                # x-direction (dim 1)
                if i == first(axes(x, 1))
                    manual[i, j, 1] = x[i + 1, j] - x[i, j]
                else
                    manual[i, j, 1] = x[i, j] - x[i - 1, j]
                end
                # y-direction (dim 2)
                if j == first(axes(x, 2))
                    manual[i, j, 2] = x[i, j + 1] - x[i, j]
                else
                    manual[i, j, 2] = x[i, j] - x[i, j - 1]
                end
            end
            @test result == manual
        end

        @testset "get_operator - 3D input (batched)" for threaded in [false, true]
            x = rand(10, 10, 5)
            reg = TotalVariation2D(0.1)
            op = get_operator(reg, x; threaded)
            # Don't assert exact batching type; validate batched behavior below

            # Test operator application
            result = op * x
            @test length(result) == 2 * length(x)  # Gradient has 2 components per slice
            # Manual batched finite differences on each slice
            manual = similar(result)
            for k in axes(x, 3)
                for i in axes(x, 1), j in axes(x, 2)
                    if i == first(axes(x, 1))
                        manual[i, j, k, 1] = x[i + 1, j, k] - x[i, j, k]
                    else
                        manual[i, j, k, 1] = x[i, j, k] - x[i - 1, j, k]
                    end
                    if j == first(axes(x, 2))
                        manual[i, j, k, 2] = x[i, j + 1, k] - x[i, j, k]
                    else
                        manual[i, j, k, 2] = x[i, j, k] - x[i, j - 1, k]
                    end
                end
            end
            @test result == manual
        end

        @testset "get_operator - dimension check" for threaded in [false, true]
            x = rand(10)  # 1D
            reg = TotalVariation2D(0.1)
            @test_throws ArgumentError get_operator(reg, x; threaded)
        end

        @testset "materialize" for threaded in [false, true]
            x = rand(8, 8)
            λ = 0.3
            reg = TotalVariation2D(λ)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            @test result isa Real
            @test result ≥ 0  # TV norm is non-negative

            # Compare with manual calculation
            op = get_operator(reg, x; threaded)
            grad = op * x
            # L2,1 mixed norm: L2 across last dim, sum over positions
            manual_result = λ * sum(sqrt.(sum(abs2, grad; dims = 3)))
            @test result ≈ manual_result
        end

        @testset "materialize - constant image" for threaded in [false, true]
            x = ones(Float64, 6, 6)  # Constant image
            reg = TotalVariation2D(0.5)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            # TV of constant image should be zero (or very small due to boundary conditions)
            @test result ≈ 0.0 atol = 1.0e-10
        end

        @testset "materialize - step function" for threaded in [false, true]
            x = zeros(Float64, 8, 8)
            x[1:4, :] .= 1.0  # Step function
            reg = TotalVariation2D(1.0)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            # Should have non-zero TV due to the step
            @test result > 0
        end
    end
end

@testitem "TotalVariation3D Regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "TotalVariation3D Regularization" begin
        @testset "get_operator - 3D input" for threaded in [false, true]
            x = rand(8, 8, 8)
            reg = TotalVariation3D(0.1)
            op = get_operator(reg, x; threaded)
            # Don't assert exact operator type; validate behavior below

            # Test operator application
            result = op * x
            @test length(result) == 3 * length(x)  # Gradient has 3 components
            # Compare to manual finite differences
            manual = similar(result)
            for i in axes(x, 1), j in axes(x, 2), k in axes(x, 3)
                # dim 1
                if i == first(axes(x, 1))
                    manual[i, j, k, 1] = x[i + 1, j, k] - x[i, j, k]
                else
                    manual[i, j, k, 1] = x[i, j, k] - x[i - 1, j, k]
                end
                # dim 2
                if j == first(axes(x, 2))
                    manual[i, j, k, 2] = x[i, j + 1, k] - x[i, j, k]
                else
                    manual[i, j, k, 2] = x[i, j, k] - x[i, j - 1, k]
                end
                # dim 3
                if k == first(axes(x, 3))
                    manual[i, j, k, 3] = x[i, j, k + 1] - x[i, j, k]
                else
                    manual[i, j, k, 3] = x[i, j, k] - x[i, j, k - 1]
                end
            end
            @test result == manual
        end

        @testset "get_operator - 4D input (batched)" for threaded in [false, true]
            x = rand(8, 8, 8, 3)
            reg = TotalVariation3D(0.1)
            op = get_operator(reg, x; threaded)
            # Don't assert exact batching type; validate batched behavior below

            # Test operator application
            result = op * x
            @test length(result) == 3 * length(x)  # Gradient has 3 components per volume
            # Manual batched finite differences on each volume
            manual = similar(result)
            for t in 1:size(x, 4)
                for i in axes(x, 1), j in axes(x, 2), k in axes(x, 3)
                    # dim 1
                    if i == first(axes(x, 1))
                        manual[i, j, k, t, 1] = x[i + 1, j, k, t] - x[i, j, k, t]
                    else
                        manual[i, j, k, t, 1] = x[i, j, k, t] - x[i - 1, j, k, t]
                    end
                    # dim 2
                    if j == first(axes(x, 2))
                        manual[i, j, k, t, 2] = x[i, j + 1, k, t] - x[i, j, k, t]
                    else
                        manual[i, j, k, t, 2] = x[i, j, k, t] - x[i, j - 1, k, t]
                    end
                    # dim 3
                    if k == first(axes(x, 3))
                        manual[i, j, k, t, 3] = x[i, j, k + 1, t] - x[i, j, k, t]
                    else
                        manual[i, j, k, t, 3] = x[i, j, k, t] - x[i, j, k - 1, t]
                    end
                end
            end
            @test result == manual
        end

        @testset "get_operator - dimension check" for threaded in [false, true]
            x = rand(10, 10)  # 2D
            reg = TotalVariation3D(0.1)
            @test_throws ArgumentError get_operator(reg, x; threaded)
        end

        @testset "materialize" for threaded in [false, true]
            x = rand(6, 6, 6)
            λ = 0.4
            reg = TotalVariation3D(λ)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            @test result isa Real
            @test result ≥ 0  # TV norm is non-negative

            # Compare with manual calculation
            op = get_operator(reg, x; threaded)
            grad = op * x
            manual_result = λ * sum(sqrt.(sum(abs2, grad; dims = 4)))
            @test result ≈ manual_result
        end

        @testset "materialize - constant volume" for threaded in [false, true]
            x = ones(Float64, 4, 4, 4)  # Constant volume
            reg = TotalVariation3D(0.7)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            # TV of constant volume should be zero (or very small due to boundary conditions)
            @test result ≈ 0.0 atol = 1.0e-10
        end
    end
end


@testitem "TemporalTotalVariation regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    @testset "Constructor" begin
        reg = TemporalTotalVariation(0.1)
        @test reg.λ == 0.1
        @test isnothing(reg.time_dim)
        @test TemporalTotalVariation(0.2; time_dim = 3).time_dim == 3
        @test TemporalTotalVariation(0.3; time_dim = :time).time_dim == :time
        @test_throws ArgumentError TemporalTotalVariation(0.1; time_dim = -1)
    end

    @testset "get_operator" for threaded in [false, true]
        x = randn(Float32, 6, 6, 5)
        reg = TemporalTotalVariation(0.1f0; time_dim = 3)
        op = get_operator(reg, x; threaded)
        @test size(op, 2) == size(x)
        @test size(op, 1) == (6, 6, 4)
        @test op * x ≈ diff(x; dims = 3)
    end

    @testset "materialize" for threaded in [false, true]
        x = randn(6, 6, 5)
        λ = 0.4
        result = MriReconstructionToolbox.calculate(TemporalTotalVariation(λ; time_dim = 3), x; threaded)
        @test result ≈ λ * sum(abs, diff(x; dims = 3))
    end

    @testset "complex data" begin
        x = randn(ComplexF64, 4, 4, 3)
        λ = 0.25
        result = MriReconstructionToolbox.calculate(TemporalTotalVariation(λ; time_dim = 3), x; threaded = false)
        @test result ≈ λ * sum(abs, diff(x; dims = 3))
    end

    @testset "NamedDimsArray input" begin
        x = NamedDimsArray{(:x, :y, :time)}(randn(4, 4, 3))
        reg = TemporalTotalVariation(0.1; time_dim = :time)
        op = get_operator(reg, x; threaded = false)
        @test op isa MriReconstructionToolbox.NamedDimsOp
        @test MriReconstructionToolbox.calculate(reg, x; threaded = false) ≈
            0.1 * sum(abs, diff(unname(x); dims = 3))
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 8, 8, 10)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        reg = TemporalTotalVariation(0.1f0; time_dim = 3)
        @test MriReconstructionToolbox.get_affected_dims(reg, info, 1:3) == (3,)
        reg_named = TemporalTotalVariation(0.1f0; time_dim = :time)
        @test MriReconstructionToolbox.get_affected_dims(reg_named, info, (:x, :y, :time)) == (:time,)
    end

    @testset "scale_regularization" begin
        reg = MriReconstructionToolbox.scale_regularization(TemporalTotalVariation(0.2; time_dim = 3), 2.5)
        @test reg.λ ≈ 0.5
        @test reg.time_dim == 3
    end
end


@testitem "Second-order TotalVariation regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    @testset "get_operator applies the gradient twice" for threaded in [false, true]
        x = randn(6, 6)
        op = get_operator(SecondOrderTotalVariation2D(1.0), x; threaded)
        @test size(op, 2) == size(x)
        @test size(op, 1) == (6, 6, 2, 2)

        # The two trailing axes are (first-derivative direction, second-derivative direction), so slice
        # (:, :, d, :) must be the gradient of the d-th component of the gradient of x.
        Δ = get_operator(TotalVariation2D(1.0), x; threaded = false)
        gradient = Δ * x
        result = op * x
        for d in 1:2
            @test result[:, :, d, :] ≈ Δ * reshape(gradient[:, :, d], 6, 6)
        end
    end

    @testset "the operator is a correct adjoint pair" for reg in
            (SecondOrderTotalVariation2D(1.0), SecondOrderTotalVariation3D(1.0))
        n = reg isa SecondOrderTotalVariation2D ? (5, 5) : (4, 4, 4)
        op = get_operator(reg, randn(n...); threaded = false)
        u = randn(n...)
        w = randn(size(op, 1)...)
        @test dot(op * u, w) ≈ dot(u, op' * w)
    end

    @testset "a linear ramp is free, unlike for first-order TV" begin
        ramp2 = [2.0i + 3.0j for i in 1:6, j in 1:6]
        @test MriReconstructionToolbox.calculate(SecondOrderTotalVariation2D(1.0), ramp2) ≈ 0 atol = 1.0e-12
        @test MriReconstructionToolbox.calculate(TotalVariation2D(1.0), ramp2) > 1

        ramp3 = [i + 2.0j + 3.0k for i in 1:5, j in 1:5, k in 1:5]
        @test MriReconstructionToolbox.calculate(SecondOrderTotalVariation3D(1.0), ramp3) ≈ 0 atol = 1.0e-12
        @test MriReconstructionToolbox.calculate(TotalVariation3D(1.0), ramp3) > 1
    end

    @testset "calculate is the voxelwise ℓ₂ norm of all second derivatives" begin
        x = randn(6, 6)
        λ = 0.3
        op = get_operator(SecondOrderTotalVariation2D(λ), x; threaded = false)
        d2 = reshape(op * x, length(x), 4)
        expected = λ * sum(sqrt.(sum(abs2, d2; dims = 2)))
        @test MriReconstructionToolbox.calculate(SecondOrderTotalVariation2D(λ), x) ≈ expected
    end

    @testset "dimensions beyond the spatial ones are batch dimensions" begin
        x = randn(6, 6, 3)
        λ = 0.2
        expected = sum(
            MriReconstructionToolbox.calculate(SecondOrderTotalVariation2D(λ), x[:, :, k]) for k in 1:3
        )
        @test MriReconstructionToolbox.calculate(SecondOrderTotalVariation2D(λ), x) ≈ expected
    end

    @testset "complex input" begin
        x = randn(ComplexF64, 6, 6)
        @test MriReconstructionToolbox.calculate(SecondOrderTotalVariation2D(0.1), x) > 0
    end

    @testset "NamedDimsArray input" begin
        x = NamedDimsArray{(:x, :y, :time)}(randn(6, 6, 2))
        reg = SecondOrderTotalVariation2D(0.1)
        op = get_operator(reg, x; threaded = false)
        @test op isa MriReconstructionToolbox.NamedDimsOp
        @test MriReconstructionToolbox.calculate(reg, x; threaded = false) ≈
            MriReconstructionToolbox.calculate(reg, unname(x); threaded = false)
    end

    @testset "too few dimensions" begin
        @test_throws ArgumentError get_operator(SecondOrderTotalVariation2D(0.1), randn(6); threaded = false)
        @test_throws ArgumentError get_operator(SecondOrderTotalVariation3D(0.1), randn(6, 6); threaded = false)
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 8, 8, 4)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test MriReconstructionToolbox.get_affected_dims(SecondOrderTotalVariation2D(0.1f0), info, 1:3) == 1:2
        @test MriReconstructionToolbox.get_affected_dims(
            SecondOrderTotalVariation3D(0.1f0), info, (:x, :y, :z, :time)
        ) == (:x, :y, :z)
    end

    @testset "scale_regularization" begin
        @test MriReconstructionToolbox.scale_regularization(SecondOrderTotalVariation2D(0.2), 2.5).λ ≈ 0.5
        @test MriReconstructionToolbox.scale_regularization(SecondOrderTotalVariation3D(0.2), 2.5).λ ≈ 0.5
    end
end
