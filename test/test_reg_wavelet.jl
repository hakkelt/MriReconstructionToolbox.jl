using TestItems

@testitem "L1Wavelet2D Regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "L1Wavelet2D Regularization" begin
        @testset "constructor" begin
            reg = L1Wavelet2D(0.1; wavelet = WT.db4, levels = 3)
            @test reg.λ == 0.1
            @test reg.wavelet == WT.db4
            @test reg.levels == 3
        end

        @testset "get_operator - no padding needed" for threaded in [false, true]
            x = rand(16, 16)  # divisible by 2^2 = 4
            reg = L1Wavelet2D(0.1; levels = 2)
            op = get_operator(reg, x; threaded)
            @test op isa WaveletOp

            # Test operator application
            result = op * x
            manual_result = dwt(x, wavelet(WT.db2), 2)
            @test result == manual_result

            # Test inverse
            x_reconstructed = op' * result
            @test x_reconstructed ≈ x rtol = 1.0e-10
        end

        @testset "get_operator - padding needed" for threaded in [false, true]
            x = rand(15, 15)  # not divisible by 2^2 = 4
            reg = L1Wavelet2D(0.1; levels = 2)
            op = get_operator(reg, x; threaded)
            @test op isa Compose  # WaveletOp * ZeroPad

            # Test operator application
            result = op * x
            @test length(result) >= length(x)  # Due to padding
            padded_x = zeros(16, 16)
            padded_x[1:15, 1:15] .= x
            manual_result = dwt(padded_x, wavelet(WT.db2), 2)
            @test result == manual_result
        end

        @testset "get_operator - 3D input (batched)" for threaded in [false, true]
            x = rand(16, 16, 5)
            reg = L1Wavelet2D(0.1; levels = 2)
            op = get_operator(reg, x; threaded)
            @test op isa BatchOp

            # Test operator application
            result = op * x
            @test length(result) == length(x)
            manual_result = zeros(16, 16, 5)
            for i in 1:5
                manual_result[:, :, i] .= dwt(x[:, :, i], wavelet(WT.db2), 2)
            end
            @test result == manual_result
        end

        @testset "get_operator - dimension check" for threaded in [false, true]
            x = rand(10)  # 1D
            reg = L1Wavelet2D(0.1)
            @test_throws ArgumentError get_operator(reg, x; threaded)
        end

        @testset "materialize" for threaded in [false, true]
            x = rand(16, 16) # rand(16, 16)
            λ = 0.2
            reg = L1Wavelet2D(λ)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            @test result isa Real
            @test result ≥ 0  # L1 norm is non-negative

            # Compare with manual wavelet transform
            op = get_operator(reg, x; threaded = false)
            wavelet_coeffs = op * x
            manual_wavelet_coeffs = dwt(x, wavelet(reg.wavelet), reg.levels)
            @test wavelet_coeffs ≈ manual_wavelet_coeffs
            manual_result = λ * norm(wavelet_coeffs, 1)
            @test result ≈ manual_result
        end

        @testset "materialize - different wavelets" for threaded in [false, true]
            x = rand(32, 32)
            λ = 0.1
            reg = L1Wavelet2D(λ; wavelet = WT.haar, levels = 3)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            manual_result = λ * sum(abs, dwt(x, wavelet(WT.haar), 3))
            @test result ≈ manual_result
        end
    end
end

@testitem "L1Wavelet3D Regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "L1Wavelet3D Regularization" begin
        @testset "constructor" begin
            reg = L1Wavelet3D(0.15; wavelet = WT.haar, levels = 1)
            @test reg.λ == 0.15
            @test reg.wavelet == WT.haar
            @test reg.levels == 1
        end

        @testset "get_operator - no padding needed" for threaded in [false, true]
            x = rand(8, 8, 8)  # divisible by 2^1 = 2
            reg = L1Wavelet3D(0.1; levels = 1)
            op = get_operator(reg, x; threaded)
            @test op isa WaveletOp

            # Test operator application
            result = op * x
            @test length(result) == length(x)

            # Test inverse
            x_reconstructed = op' * result
            @test x_reconstructed ≈ x rtol = 1.0e-10
        end

        @testset "get_operator - padding needed" for threaded in [false, true]
            x = rand(7, 9, 11)  # not all divisible by 2^1 = 2
            reg = L1Wavelet3D(0.1; levels = 1)
            op = get_operator(reg, x; threaded)
            @test op isa Compose  # WaveletOp * ZeroPad

            # Test operator application
            result = op * x
            @test length(result) >= length(x)  # Due to padding
            # Check inverse with cropping via adjoint
            x_reconstructed = op' * result
            @test x_reconstructed ≈ x rtol = 1.0e-10
        end

        @testset "get_operator - 4D input (batched)" for threaded in [false, true]
            x = rand(8, 8, 8, 3)
            reg = L1Wavelet3D(0.1; levels = 1)
            op = get_operator(reg, x; threaded)
            @test op isa BatchOp

            # Test operator application
            result = op * x
            @test length(result) == length(x)
            # Inverse consistency per batch
            x_reconstructed = op' * result
            @test x_reconstructed ≈ x rtol = 1.0e-10
        end

        @testset "get_operator - dimension check" for threaded in [false, true]
            x = rand(10, 10)  # 2D
            reg = L1Wavelet3D(0.1)
            @test_throws ArgumentError get_operator(reg, x; threaded)
        end

        @testset "materialize" for threaded in [false, true]
            x = rand(8, 8, 8)
            λ = 0.25
            reg = L1Wavelet3D(λ)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            @test result isa Real
            @test result ≥ 0  # L1 norm is non-negative

            # Compare with operator-applied wavelet coefficients
            op = get_operator(reg, x; threaded)
            wavelet_coeffs = op * x
            manual_result = λ * sum(abs, wavelet_coeffs)
            @test result ≈ manual_result
        end
    end
end

