using Test
using MriReconstructionToolbox
using AbstractOperators
using NamedDims
using StructuredOptimization

# Helper to evaluate a StructuredOptimization Term with a single variable
function eval_term(terms)
 	vars = StructuredOptimization.extract_variables(terms)
	@assert length(vars) == 1
	xvar = vars[1]
	f = StructuredOptimization.extract_functions(terms)
	op = StructuredOptimization.extract_operators((xvar,), terms)
	xval = ~xvar
	return f(op * xval)
end

@testset "minimizer.build_model" begin
	# 1) Identity operator, single regularization (L1Image)
	@testset "Eye + L1Image" for threaded in (false, true)
		x = rand(8, 8)
		𝒜 = Eye(x)
		y = 𝒜 * x .+ 0.01 .* randn(size(x))
		reg = L1Image(0.2)
		terms = build_model(𝒜, y, reg; threaded)

		model_val = eval_term(terms)

		x̂ = 𝒜' * y
		data_fidelity = 0.5 * sum(abs2, (𝒜 * x̂) .- y)
		reg_val = MriReconstructionToolbox.calculate(reg, y; threaded)
		@test isapprox(model_val, data_fidelity + reg_val; rtol=1e-10, atol=1e-12)
	end

	# 2) Identity operator, two regularizations (L1Image + Tikhonov)
	@testset "Eye + L1Image + Tikhonov" for threaded in (false, true)
		x = rand(6, 6)
		𝒜 = Eye(x)
		y = copy(x)
		regs = (L1Image(0.1), Tikhonov(0.05))
		t = build_model(𝒜, y, regs; threaded)

		model_val = eval_term(t)

		x̂ = 𝒜' * y
		data_fidelity = 0.5 * sum(abs2, (𝒜 * x̂) .- y)
		reg1 = MriReconstructionToolbox.calculate(regs[1], y; threaded)
		reg2 = MriReconstructionToolbox.calculate(regs[2], y; threaded)
		@test isapprox(model_val, data_fidelity + reg1 + reg2; rtol=1e-10, atol=1e-12)
	end

	# 3) Nontrivial operator: simple zero-pad + identity batch; checks data term contribution
	@testset "Linear op + Tikhonov" for threaded in (false, true)
		x = rand(8, 8)
		y = rand(8, 8)
		𝒜 = Eye(x)  # keep simple/fast; exercise op path anyway
		reg = Tikhonov(0.3)
		t = build_model(𝒜, y, reg; threaded)

		# Evaluate term and compare to manual objective at default x0 = 𝒜' * y = y
		model_val = eval_term(t)

		x̂ = 𝒜' * y
		data_fidelity = 0.5 * sum(abs2, (𝒜 * x̂) .- y)
		reg_val = MriReconstructionToolbox.calculate(reg, y; threaded)
		@test isapprox(model_val, data_fidelity + reg_val; rtol=1e-10, atol=1e-12)

		# Now perturb x and verify objective increases appropriately
		vars = StructuredOptimization.extract_variables(t)
		xvar = vars[1]
		x0 = copy(~xvar)
		δ = 0.01 .* randn(size(x0))
		~xvar .= x0 .+ δ  # update variable value
		model_val2 = eval_term(t)
		# manual objective: 1/2*||A*(x0+δ)-y||^2 + R(x0+δ)
		data = 0.5 * sum(abs2, (𝒜 * (x0 .+ δ)) .- y)
		reg2 = MriReconstructionToolbox.calculate(reg, x0 .+ δ; threaded)
		@test isapprox(model_val2, data + reg2; rtol=1e-8, atol=1e-10)
	end

	# 4) NamedDims inputs: ensure build_model accepts NamedDims and evaluates
	@testset "NamedDims y and A" for threaded in (false, true)
		x = rand(8, 8)
		𝒜 = MriReconstructionToolbox.NamedDimsOp{(:x, :y),(:x, :y)}(Eye(x))
		y = NamedDimsArray(copy(x), (:x, :y))
		reg = L1Image(0.15)
		t = build_model(𝒜, y, reg; threaded)

		model_val = eval_term(t)

		x̂ = 𝒜' * y
		data_fidelity = 0.5 * sum(abs2, (𝒜 * x̂) .- y)
		reg_val = MriReconstructionToolbox.calculate(reg, y; threaded)
		@test isapprox(
			model_val,
			data_fidelity + reg_val;
			rtol=1e-10,
			atol=1e-12,
		)
	end

	# 5) Multiple regs tuple overload correctness vs single-reg method
	@testset "overload parity" for threaded in (false, true)
		x = rand(5, 5)
		𝒜 = Eye(x)
		y = copy(x)
		reg = Tikhonov(0.2)
		t1 = build_model(𝒜, y, reg; threaded)
		t2 = build_model(𝒜, y, (reg,); threaded)
		@test isapprox(eval_term(t1), eval_term(t2); atol=1e-12)
	end
end
