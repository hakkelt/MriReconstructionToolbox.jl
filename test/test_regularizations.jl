@testset "Regularization Tests" begin
	
	@testset "Tikhonov Regularization" for threaded in [false, true]
		@testset "get_operator" begin
			x = rand(10, 10)
			reg = Tikhonov(0.1)
			op = get_operator(reg, x; threaded)
			@test op isa Eye
			@test size(op) == (size(x), size(x))
			result = op * x
			@test result ≈ x
		end
		
		@testset "materialize - scalar λ" for threaded in [false, true]
			x = rand(5, 5)
			λ = 0.5
			reg = Tikhonov(λ)
			result = MriReconstructionToolbox.calculate(reg, x; threaded)
			manual_result = sum(abs2, λ .* x)
			@test result ≈ manual_result
		end
		
		@testset "materialize - matrix λ" for threaded in [false, true]
			x = ones(3, 3)
			λ_matrix = [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
			reg = Tikhonov(λ_matrix)
			result = MriReconstructionToolbox.calculate(reg, x; threaded)
			manual_result = sum(abs2, λ_matrix .* x)
			@test result ≈ manual_result
		end
	end
	
	@testset "L1Image Regularization" begin
		@testset "get_operator" for threaded in [false, true]
			x = rand(8, 8)
			reg = L1Image(0.2)
			op = get_operator(reg, x; threaded)
			@test op isa Eye
			@test size(op) == (size(x), size(x))
			result = op * x
			@test result ≈ x
		end
		
		@testset "materialize" for threaded in [false, true]
			x = rand(6, 6)
			λ = 0.3
			reg = L1Image(λ)
			result = MriReconstructionToolbox.calculate(reg, x; threaded)
			manual_result = λ * norm(x, 1)
			@test result ≈ manual_result
		end
		
		@testset "materialize - complex data" for threaded in [false, true]
			x = randn(ComplexF64, 4, 4)
			λ = 0.5
			reg = L1Image(λ)
			result = MriReconstructionToolbox.calculate(reg, x; threaded)
			manual_result = λ * norm(x, 1)
			@test result ≈ manual_result
		end
	end
	
	@testset "L1Wavelet2D Regularization" begin
		@testset "constructor" begin
			reg = L1Wavelet2D(0.1; wavelet=WT.db4, levels=3)
			@test reg.λ == 0.1
			@test reg.wavelet == WT.db4
			@test reg.levels == 3
		end
		
		@testset "get_operator - no padding needed" for threaded in [false, true]
			x = rand(16, 16)  # divisible by 2^2 = 4
			reg = L1Wavelet2D(0.1; levels=2)
			op = get_operator(reg, x; threaded)
			@test op isa WaveletOp
			
			# Test operator application
			result = op * x
			manual_result = dwt(x, wavelet(WT.db2), 2)
			@test result == manual_result
			
			# Test inverse
			x_reconstructed = op' * result
			@test x_reconstructed ≈ x rtol=1e-10
		end
		
		@testset "get_operator - padding needed" for threaded in [false, true]
			x = rand(15, 15)  # not divisible by 2^2 = 4
			reg = L1Wavelet2D(0.1; levels=2)
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
			reg = L1Wavelet2D(0.1; levels=2)
			op = get_operator(reg, x; threaded)
			@test op isa BatchOp
			
			# Test operator application
			result = op * x
			@test length(result) == length(x)
			manual_result = zeros(16, 16, 5)
			for i in 1:5
				manual_result[:,:,i] .= dwt(x[:,:,i], wavelet(WT.db2), 2)
			end
			@test result == manual_result
		end
		
		@testset "get_operator - dimension check" for threaded in [false, true]
			x = rand(10)  # 1D
			reg = L1Wavelet2D(0.1)
			@test_throws AssertionError get_operator(reg, x; threaded)
		end
		
		@testset "materialize" for threaded in [false, true]
			x = rand(16, 16) # rand(16, 16)
			λ = 0.2
			reg = L1Wavelet2D(λ)
			result = MriReconstructionToolbox.calculate(reg, x; threaded)
			@test result isa Real
			@test result ≥ 0  # L1 norm is non-negative
			
			# Compare with manual wavelet transform
			op = get_operator(reg, x; threaded=false)
			wavelet_coeffs = op * x
			manual_wavelet_coeffs = dwt(x, wavelet(reg.wavelet), reg.levels)
			@test wavelet_coeffs ≈ manual_wavelet_coeffs
			manual_result = λ * norm(wavelet_coeffs, 1)
			@test result ≈ manual_result
		end
		
		@testset "materialize - different wavelets" for threaded in [false, true]
			x = rand(32, 32)
			λ = 0.1
			reg = L1Wavelet2D(λ; wavelet=WT.haar, levels=3)
			result = MriReconstructionToolbox.calculate(reg, x; threaded)
			manual_result = λ * sum(abs, dwt(x, wavelet(WT.haar), 3))
			@test result ≈ manual_result
		end
	end
	
	@testset "L1Wavelet3D Regularization" begin
		@testset "constructor" begin
			reg = L1Wavelet3D(0.15; wavelet=WT.haar, levels=1)
			@test reg.λ == 0.15
			@test reg.wavelet == WT.haar
			@test reg.levels == 1
		end

		@testset "get_operator - no padding needed" for threaded in [false, true]
			x = rand(8, 8, 8)  # divisible by 2^1 = 2
			reg = L1Wavelet3D(0.1; levels=1)
			op = get_operator(reg, x; threaded)
			@test op isa WaveletOp
			
			# Test operator application
			result = op * x
			@test length(result) == length(x)
			
			# Test inverse
			x_reconstructed = op' * result
			@test x_reconstructed ≈ x rtol=1e-10
		end

		@testset "get_operator - padding needed" for threaded in [false, true]
			x = rand(7, 9, 11)  # not all divisible by 2^1 = 2
			reg = L1Wavelet3D(0.1; levels=1)
			op = get_operator(reg, x; threaded)
			@test op isa Compose  # WaveletOp * ZeroPad
			
			# Test operator application
			result = op * x
			@test length(result) >= length(x)  # Due to padding
			# Check inverse with cropping via adjoint
			x_reconstructed = op' * result
			@test x_reconstructed ≈ x rtol=1e-10
		end

		@testset "get_operator - 4D input (batched)" for threaded in [false, true]
			x = rand(8, 8, 8, 3)
			reg = L1Wavelet3D(0.1; levels=1)
			op = get_operator(reg, x; threaded)
			@test op isa BatchOp
			
			# Test operator application
			result = op * x
			@test length(result) == length(x)
			# Inverse consistency per batch
			x_reconstructed = op' * result
			@test x_reconstructed ≈ x rtol=1e-10
		end

		@testset "get_operator - dimension check" for threaded in [false, true]
			x = rand(10, 10)  # 2D
			reg = L1Wavelet3D(0.1)
			@test_throws AssertionError get_operator(reg, x; threaded)
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
			for i in axes(x,1), j in axes(x,2)
				# x-direction (dim 1)
				if i == first(axes(x,1))
					manual[i,j,1] = x[i+1,j] - x[i,j]
				else
					manual[i,j,1] = x[i,j] - x[i-1,j]
				end
				# y-direction (dim 2)
				if j == first(axes(x,2))
					manual[i,j,2] = x[i,j+1] - x[i,j]
				else
					manual[i,j,2] = x[i,j] - x[i,j-1]
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
			for k in axes(x,3)
				for i in axes(x,1), j in axes(x,2)
					if i == first(axes(x,1))
						manual[i,j,k,1] = x[i+1,j,k] - x[i,j,k]
					else
						manual[i,j,k,1] = x[i,j,k] - x[i-1,j,k]
					end
					if j == first(axes(x,2))
						manual[i,j,k,2] = x[i,j+1,k] - x[i,j,k]
					else
						manual[i,j,k,2] = x[i,j,k] - x[i,j-1,k]
					end
				end
			end
			@test result == manual
		end
		
		@testset "get_operator - dimension check" for threaded in [false, true]
			x = rand(10)  # 1D
			reg = TotalVariation2D(0.1)
			@test_throws AssertionError get_operator(reg, x; threaded)
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
			manual_result = λ * sum(sqrt.(sum(abs2, grad; dims=3)))
			@test result ≈ manual_result
		end

		@testset "materialize - constant image" for threaded in [false, true]
			x = ones(Float64, 6, 6)  # Constant image
			reg = TotalVariation2D(0.5)
			result = MriReconstructionToolbox.calculate(reg, x; threaded)
			# TV of constant image should be zero (or very small due to boundary conditions)
			@test result ≈ 0.0 atol=1e-10
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
			for i in axes(x,1), j in axes(x,2), k in axes(x,3)
				# dim 1
				if i == first(axes(x,1))
					manual[i,j,k,1] = x[i+1,j,k] - x[i,j,k]
				else
					manual[i,j,k,1] = x[i,j,k] - x[i-1,j,k]
				end
				# dim 2
				if j == first(axes(x,2))
					manual[i,j,k,2] = x[i,j+1,k] - x[i,j,k]
				else
					manual[i,j,k,2] = x[i,j,k] - x[i,j-1,k]
				end
				# dim 3
				if k == first(axes(x,3))
					manual[i,j,k,3] = x[i,j,k+1] - x[i,j,k]
				else
					manual[i,j,k,3] = x[i,j,k] - x[i,j,k-1]
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
			for t in 1:size(x,4)
				for i in axes(x,1), j in axes(x,2), k in axes(x,3)
					# dim 1
					if i == first(axes(x,1))
						manual[i,j,k,t,1] = x[i+1,j,k,t] - x[i,j,k,t]
					else
						manual[i,j,k,t,1] = x[i,j,k,t] - x[i-1,j,k,t]
					end
					# dim 2
					if j == first(axes(x,2))
						manual[i,j,k,t,2] = x[i,j+1,k,t] - x[i,j,k,t]
					else
						manual[i,j,k,t,2] = x[i,j,k,t] - x[i,j-1,k,t]
					end
					# dim 3
					if k == first(axes(x,3))
						manual[i,j,k,t,3] = x[i,j,k+1,t] - x[i,j,k,t]
					else
						manual[i,j,k,t,3] = x[i,j,k,t] - x[i,j,k-1,t]
					end
				end
			end
			@test result == manual
		end
		
		@testset "get_operator - dimension check" for threaded in [false, true]
			x = rand(10, 10)  # 2D
			reg = TotalVariation3D(0.1)
			@test_throws AssertionError get_operator(reg, x; threaded)
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
			manual_result = λ * sum(sqrt.(sum(abs2, grad; dims=4)))
			@test result ≈ manual_result
		end

		@testset "materialize - constant volume" for threaded in [false, true]
			x = ones(Float64, 4, 4, 4)  # Constant volume
			reg = TotalVariation3D(0.7)
			result = MriReconstructionToolbox.calculate(reg, x; threaded)
			# TV of constant volume should be zero (or very small due to boundary conditions)
			@test result ≈ 0.0 atol=1e-10
		end
	end
	
	@testset "Threading Tests" begin
		@testset "L1Wavelet2D with threading" begin
			x = rand(16, 16, 4)
			reg = L1Wavelet2D(0.1)
			op_threaded = get_operator(reg, x; threaded=true)
			op_sequential = get_operator(reg, x; threaded=false)
			@test size(op_threaded) == size(op_sequential)
			
			# Test that results are the same
			result_threaded = op_threaded * x
			result_sequential = op_sequential * x
			@test result_threaded ≈ result_sequential
		end
		
		@testset "TotalVariation2D with threading" begin
			x = rand(10, 10, 3)
			reg = TotalVariation2D(0.1)
			op_threaded = get_operator(reg, x; threaded=true)
			op_sequential = get_operator(reg, x; threaded=false)
			@test size(op_threaded) == size(op_sequential)
			
			# Test that results are the same
			result_threaded = op_threaded * x
			result_sequential = op_sequential * x
			@test result_threaded ≈ result_sequential
		end
	end
	
	@testset "Type Stability Tests" begin
		@testset "Float32 compatibility" begin
			x = randn(Float32, 8, 8)
			reg = Tikhonov(0.1f0)
			result = MriReconstructionToolbox.calculate(reg, x; threaded=false)
			@test typeof(result) == Float32
		end
		
		@testset "Complex number compatibility" begin
			x = randn(ComplexF64, 8, 8)
			reg = L1Image(0.1)
			result = MriReconstructionToolbox.calculate(reg, x; threaded=false)
			@test result isa Real
		end
		
		@testset "Operator type consistency" begin
			x = randn(Float32, 8, 8)
			reg = L1Wavelet2D(0.1f0)
			op = get_operator(reg, x; threaded=false)
			result = op * x
			@test eltype(result) == Float32
		end
	end
	
	@testset "Edge Cases" begin
		@testset "Small arrays" begin
			x = rand(2, 2)
			reg = TotalVariation2D(0.1)
			op = get_operator(reg, x; threaded=false)
			result = op * x
			@test length(result) == 2 * length(x)
		end
		
		@testset "Zero regularization parameter" begin
			x = rand(5, 5)
			reg = Tikhonov(0.0)
			result = MriReconstructionToolbox.calculate(reg, x; threaded=false)
			@test result ≈ 0.0 atol=1e-15
		end
		
		@testset "High levels wavelet" begin
			x = rand(32, 32)
			reg = L1Wavelet2D(0.1; levels=4)  # High decomposition level
			op = get_operator(reg, x; threaded=false)
			result = op * x
			@test length(result) == length(x)
		end
	end

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
			reg = L1Wavelet2D(λ; levels=2)
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
			reg = L1Wavelet3D(λ; levels=1)
			op = get_operator(reg, x_named; threaded)
			coeffs = op * x_named
			x_rec = op' * coeffs
			@test Array(x_rec) ≈ x rtol=1e-10
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
			for i in axes(x,1), j in axes(x,2)
				manual[i,j,1] = (i == first(axes(x,1))) ? x[i+1,j] - x[i,j] : x[i,j] - x[i-1,j]
				manual[i,j,2] = (j == first(axes(x,2))) ? x[i,j+1] - x[i,j] : x[i,j] - x[i,j-1]
			end
			@test Array(g) == manual
			res_calc = MriReconstructionToolbox.calculate(reg, x_named; threaded)
			@test res_calc ≈ λ * sum(sqrt.(sum(abs2, manual; dims=3)))
		end

		@testset "TotalVariation3D NamedDims" for threaded in [false, true]
			x = rand(6, 6, 6)
			x_named = NamedDimsArray(x, (:x, :y, :z))
			λ = 0.4
			reg = TotalVariation3D(λ)
			op = get_operator(reg, x_named; threaded)
			g = op * x_named
			manual = similar(Array(g))
			for i in axes(x,1), j in axes(x,2), k in axes(x,3)
				manual[i,j,k,1] = (i == first(axes(x,1))) ? x[i+1,j,k] - x[i,j,k] : x[i,j,k] - x[i-1,j,k]
				manual[i,j,k,2] = (j == first(axes(x,2))) ? x[i,j+1,k] - x[i,j,k] : x[i,j,k] - x[i,j-1,k]
				manual[i,j,k,3] = (k == first(axes(x,3))) ? x[i,j,k+1] - x[i,j,k] : x[i,j,k] - x[i,j,k-1]
			end
			@test Array(g) == manual
			res_calc = MriReconstructionToolbox.calculate(reg, x_named; threaded)
			@test res_calc ≈ λ * sum(sqrt.(sum(abs2, manual; dims=4)))
		end
	end
end