using TestItems

@testitem "Threading Tests" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "Threading Tests" begin
        @testset "L1Wavelet2D with threading" begin
            x = rand(16, 16, 4)
            reg = L1Wavelet2D(0.1)
            op_threaded = get_operator(reg, x; threaded = true)
            op_sequential = get_operator(reg, x; threaded = false)
            @test size(op_threaded) == size(op_sequential)

            # Test that results are the same
            result_threaded = op_threaded * x
            result_sequential = op_sequential * x
            @test result_threaded ≈ result_sequential
        end

        @testset "TotalVariation2D with threading" begin
            x = rand(10, 10, 3)
            reg = TotalVariation2D(0.1)
            op_threaded = get_operator(reg, x; threaded = true)
            op_sequential = get_operator(reg, x; threaded = false)
            @test size(op_threaded) == size(op_sequential)

            # Test that results are the same
            result_threaded = op_threaded * x
            result_sequential = op_sequential * x
            @test result_threaded ≈ result_sequential
        end
    end
end

@testitem "Type Stability Tests" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "Type Stability Tests" begin
        @testset "Float32 compatibility" begin
            x = randn(Float32, 8, 8)
            reg = Tikhonov(0.1f0)
            result = MriReconstructionToolbox.calculate(reg, x; threaded = false)
            @test typeof(result) == Float32
        end

        @testset "Complex number compatibility" begin
            x = randn(ComplexF64, 8, 8)
            reg = L1Image(0.1)
            result = MriReconstructionToolbox.calculate(reg, x; threaded = false)
            @test result isa Real
        end

        @testset "Operator type consistency" begin
            x = randn(Float32, 8, 8)
            reg = L1Wavelet2D(0.1f0)
            op = get_operator(reg, x; threaded = false)
            result = op * x
            @test eltype(result) == Float32
        end
    end
end

@testitem "Edge Cases" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "Edge Cases" begin
        @testset "Small arrays" begin
            x = rand(2, 2)
            reg = TotalVariation2D(0.1)
            op = get_operator(reg, x; threaded = false)
            result = op * x
            @test length(result) == 2 * length(x)
        end

        @testset "Zero regularization parameter" begin
            x = rand(5, 5)
            reg = Tikhonov(0.0)
            result = MriReconstructionToolbox.calculate(reg, x; threaded = false)
            @test result ≈ 0.0 atol = 1.0e-15
        end

        @testset "High levels wavelet" begin
            x = rand(32, 32)
            reg = L1Wavelet2D(0.1; levels = 4)  # High decomposition level
            op = get_operator(reg, x; threaded = false)
            result = op * x
            @test length(result) == length(x)
        end
    end
end

@testitem "NamedDimsArray inputs" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "NamedDimsArray inputs" begin
        @testset "Tikhonov NamedDims" for threaded in [false, true]
            x = rand(5, 5)
            x_named = NamedDimsArray(x, (:x, :y))
            λ = 0.3
            reg = Tikhonov(λ)
            op = get_operator(reg, x_named; threaded)
            result = op * x_named
            @test Array(result) ≈ x
            res_calc = MriReconstructionToolbox.calculate(reg, x_named; threaded)
            @test res_calc ≈ sum(abs2, λ .* x)
        end

        @testset "L1Image NamedDims" for threaded in [false, true]
            x = rand(6, 6)
            x_named = NamedDimsArray(x, (:x, :y))
            λ = 0.2
            reg = L1Image(λ)
            op = get_operator(reg, x_named; threaded)
            result = op * x_named
            @test Array(result) ≈ x
            res_calc = MriReconstructionToolbox.calculate(reg, x_named; threaded)
            @test res_calc ≈ λ * norm(x, 1)
        end

        @testset "L1Wavelet2D NamedDims" for threaded in [false, true]
            x = rand(16, 16)
            x_named = NamedDimsArray(x, (:x, :y))
            λ = 0.1
            reg = L1Wavelet2D(λ; levels = 2)
            op = get_operator(reg, x_named; threaded)
            coeffs = op * x_named
            manual = dwt(x, wavelet(WT.db2), 2)
            @test Array(coeffs) == manual
            res_calc = MriReconstructionToolbox.calculate(reg, x_named; threaded)
            @test res_calc ≈ λ * sum(abs, manual)
        end

        @testset "L1Wavelet3D NamedDims" for threaded in [false, true]
            x = rand(8, 8, 8)
            x_named = NamedDimsArray(x, (:x, :y, :z))
            λ = 0.25
            reg = L1Wavelet3D(λ; levels = 1)
            op = get_operator(reg, x_named; threaded)
            coeffs = op * x_named
            x_rec = op' * coeffs
            @test Array(x_rec) ≈ x rtol = 1.0e-10
            res_calc = MriReconstructionToolbox.calculate(reg, x_named; threaded)
            @test res_calc ≈ λ * sum(abs, Array(coeffs))
        end

        @testset "TotalVariation2D NamedDims" for threaded in [false, true]
            x = rand(8, 8)
            x_named = NamedDimsArray(x, (:x, :y))
            λ = 0.3
            reg = TotalVariation2D(λ)
            op = get_operator(reg, x_named; threaded)
            g = op * x_named
            manual = similar(Array(g))
            for i in axes(x, 1), j in axes(x, 2)
                manual[i, j, 1] = (i == first(axes(x, 1))) ? x[i + 1, j] - x[i, j] : x[i, j] - x[i - 1, j]
                manual[i, j, 2] = (j == first(axes(x, 2))) ? x[i, j + 1] - x[i, j] : x[i, j] - x[i, j - 1]
            end
            @test Array(g) == manual
            res_calc = MriReconstructionToolbox.calculate(reg, x_named; threaded)
            @test res_calc ≈ λ * sum(sqrt.(sum(abs2, manual; dims = 3)))
        end

        @testset "TotalVariation3D NamedDims" for threaded in [false, true]
            x = rand(6, 6, 6)
            x_named = NamedDimsArray(x, (:x, :y, :z))
            λ = 0.4
            reg = TotalVariation3D(λ)
            op = get_operator(reg, x_named; threaded)
            g = op * x_named
            manual = similar(Array(g))
            for i in axes(x, 1), j in axes(x, 2), k in axes(x, 3)
                manual[i, j, k, 1] = (i == first(axes(x, 1))) ? x[i + 1, j, k] - x[i, j, k] : x[i, j, k] - x[i - 1, j, k]
                manual[i, j, k, 2] = (j == first(axes(x, 2))) ? x[i, j + 1, k] - x[i, j, k] : x[i, j, k] - x[i, j - 1, k]
                manual[i, j, k, 3] = (k == first(axes(x, 3))) ? x[i, j, k + 1] - x[i, j, k] : x[i, j, k] - x[i, j, k - 1]
            end
            @test Array(g) == manual
            res_calc = MriReconstructionToolbox.calculate(reg, x_named; threaded)
            @test res_calc ≈ λ * sum(sqrt.(sum(abs2, manual; dims = 4)))
        end
    end
end

@testitem "scale_regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox

    # L1-homogeneous terms: λ scales linearly with the factor (used by regularized problem
    # decomposition to compensate for solving all slices with one shared data scale).
    @testset "L1-type terms scale λ linearly" begin
        factor = 3.5
        @test MriReconstructionToolbox.scale_regularization(L1Image(0.1), factor).λ ≈ 0.1 * factor
        @test MriReconstructionToolbox.scale_regularization(TotalVariation2D(0.2), factor).λ ≈ 0.2 * factor
        @test MriReconstructionToolbox.scale_regularization(TotalVariation3D(0.2), factor).λ ≈ 0.2 * factor
        @test MriReconstructionToolbox.scale_regularization(TemporalFourier(0.3), factor).λ ≈ 0.3 * factor
        @test MriReconstructionToolbox.scale_regularization(LowRank(0.4), factor).λ ≈ 0.4 * factor
        @test MriReconstructionToolbox.scale_regularization(L1Wavelet2D(0.5), factor).λ ≈ 0.5 * factor
        @test MriReconstructionToolbox.scale_regularization(L1Wavelet3D(0.5), factor).λ ≈ 0.5 * factor

        # array-valued λ is scaled elementwise
        λ_arr = rand(4, 4)
        @test MriReconstructionToolbox.scale_regularization(L1Image(λ_arr), factor).λ ≈ λ_arr .* factor
    end

    # Quadratic penalty (λ²‖x‖²) and rank constraints are already scale-consistent: the data
    # term and the regularization term scale identically with x, so no correction is needed.
    @testset "Quadratic/rank-constraint terms need no correction" begin
        factor = 3.5
        @test MriReconstructionToolbox.scale_regularization(Tikhonov(0.1), factor).λ == 0.1
        @test MriReconstructionToolbox.scale_regularization(RankLimit(4), factor).max_rank == 4
    end

    @testset "auxiliary fields are preserved" begin
        reg = L1Wavelet2D(0.5; levels = 3)
        scaled = MriReconstructionToolbox.scale_regularization(reg, 2.0)
        @test scaled.wavelet == reg.wavelet
        @test scaled.levels == reg.levels

        reg_lr = LowRank(0.4; time_dim = 3)
        scaled_lr = MriReconstructionToolbox.scale_regularization(reg_lr, 2.0)
        @test scaled_lr.time_dim == 3
    end
end
