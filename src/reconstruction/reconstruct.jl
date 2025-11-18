"""
	reconstruct(
		acq_data::AcquisitionInfo;
		regularization::Union{Regularization, Tuple{Vararg{Regularization}}}=(),
		algorithm=(CG(;maxit=20), FISTA(;maxit=50), ADMM(;maxit=100)),
		threaded::Bool=true)

Performs MRI reconstruction from k-space data using the specified regularization, and optimization algorithm.

# Arguments
- `acq_data::AcquisitionInfo`: The acquisition information containing k-space data, sensitivity maps, and other parameters.
- `regularization::Union{Regularization, Tuple{Vararg{Regularization}}}`: The regularization term(s) to use (default is no regularization).
- `algorithm`: The optimization algorithm(s) to use (default is a tuple of CG, FISTA, and ADMM with specified max iterations).
- `x₀::Union{Nothing,AbstractArray}=nothing`: Optional initial guess for the image (default is 𝒜' * y).
- `normalization::Normalization = BartScaling()`: scaling applied to operators/data
- `tol::Float64 = 1e-4`: stopping tolerance for iterative algorithms
- `maxit::Int = 100`: maximum iterations for the chosen solver
- `freq::Union{Nothing,Int} = nothing`: progress print frequency (iterations)
- `verbose::Bool = true`: enable/disable logging output
- `threaded::Bool = (Threads.nthreads() > 1)`: enable threaded execution when available
- `exact_opnorm::Bool = false`: use exact operator norm for stepsize estimation
- `decomposition_executor::Union{Nothing,ReconstructionExecutor} = nothing`: override executor for decomposition
- `disable_inverse_scale_output::Bool = false`: skip rescaling the final output
- `disable_normalop_optimization::Bool = false`: disable normal-operator optimizations
- `disable_problem_decomposition::Bool = false`: disable automatic problem decomposition
- `disable_operator_normalization::Bool = false`: disable operator normalization
- `printfunc::Function = println`: custom logging function

# Returns
- The reconstructed image (NamedDimsArray if input is NamedDimsArray, otherwise standard Array).

# Notes

## Normalop Optimization

- If `disable_normalop_optimization` is false, the function uses `normalop_ls` for efficiency when appropriate.
- It should not change the result of the reconstruction, but it changes the reported value of consistency term.
- To understand this, one can think of the optimization problem as minimizing
`|| 𝒜*x - y ||_2^2 + Σ R_i(x)`, where `|| 𝒜*x - y ||_2^2` is the consistency term, `R_i` are the regularization
terms, `𝒜` is the encoding operator, and `y` is the k-space data.
- When normalop optimization is disabled, the reported value of `f(x)` is exactly the consistency term
`|| 𝒜*x - y ||_2^2`. When enabled, it exploits the fact that `∇f(x) = 𝒜'*(𝒜*x - y) = 𝒜'*𝒜*x - 𝒜'*y`,
and usually there exists an optimized operator for `𝒜'*𝒜`. Therefore, it computes `f(x)` as `|| 𝒜'*𝒜*x - 𝒜'*y ||_2^2`, which leads to the same result, but with
potentially improved efficiency.

## Problem Decomposition

- If `disable_problem_decomposition` is false, the function automatically decomposes the reconstruction problem
over batch dimensions of the image that are not affected by the Fourier transform or regularization terms.
- E.g., for a 3D+t acquisition with no regularization, the reconstruction is decomposed over the time dimension,
and each 3D volume is reconstructed independently.
- This can significantly speed up the reconstruction when multiple CPU cores are available. Usually, this leads
to better resource utilization and faster overall reconstruction times, but is useful to disable for debug purposes.

"""
function reconstruct(
	acq_data::AcquisitionInfo,
	regularization::Union{Regularization,Tuple{Vararg{Regularization}}}=(),
	algorithm=(CG(), CGNR(), FISTA(), ADMM());
	x₀::Union{Nothing,AbstractArray}=nothing,
	kwargs...,
)
	config = construct_config(kwargs)
	t_start = time()
	regularization = ensure_tuple(regularization)
	decomposition_plan = get_problem_decomposition_plan(acq_data, regularization, config)
	if isnothing(decomposition_plan)
		@conditionally_enable_threading config.threaded begin
			x, _ = _reconstruct(acq_data, regularization, algorithm, x₀, config)
		end
	else
		x = execute(decomposition_plan, acq_data, config) do local_acq, local_conf
			_reconstruct(local_acq, regularization, algorithm, x₀, local_conf)
		end
	end
	t_end = time()
	config.verbose && config.printfunc("Total time: ", format_time(t_end - t_start))
	return x
end

function _reconstruct(acq_data, regularization, algorithm, x₀, config)
	# Construct encoding operator
	@step "Constructing encoding operator" config begin
		𝒜 = get_encoding_operator(
			acq_data; threaded=config.threaded, fast_planning=regularization == ()
		)
	end

	# Direct reconstruction
	x̂, scale = _direct_reconstruct(𝒜, acq_data, x₀, regularization, config)

	if regularization == ()
		# No regularization, return direct reconstruction
		if scale != 1 && config.disable_inverse_scale_output
			@step "Scaling image" config begin
				x̂ ./= scale
			end
		end
	else
		# Iterative reconstruction with regularization
		x̂ = _iterative_reconstruct(
			𝒜, acq_data, x̂, scale, regularization, algorithm, config
		)
	end

	return x̂, scale
end

function _direct_reconstruct(𝒜, acq_data, x₀, regularization, config)
	direct_recon_only = regularization == ()
	if !isnothing(x₀) && direct_recon_only
		config.verbose && config.printfunc(
			"Warning: Initial guess x₀ is ignored when no regularization is specified."
		)
		x₀ = nothing
	end
	if isnothing(x₀)
		@step (direct_recon_only ? "Reconstructing image" : "Getting initial estimate") config begin
			x₀ = 𝒜' * acq_data.kspace_data
		end
	end
	if config.normalization != NoScaling()
		@step "Computing scaling factor" config begin
			scale = get_scale(config.normalization, acq_data, x₀)
		end
		if scale == 0
			config.verbose &&
				config.printfunc("Warning: Computed scale is zero, defaulting to scale=1.0")
			scale = 1
		end
		config.verbose && @sprintf("Using scaling factor: %g\n", scale)
	else
		scale = 1
	end
	return x₀, real(eltype(x₀))(scale)
end

function _iterative_reconstruct(𝒜, acq_data, x₀, scale, regularization, algorithm, config)
	if scale != 1
		@step "Scaling k-space data" config begin
			acq_data = AcquisitionInfo(acq_data, kspace_data=acq_data.kspace_data ./ scale)
		end
	end
	if !config.disable_operator_normalization
		@step "Normalizing encoding operator" config begin
			𝒜 = normalize_op(𝒜, config.exact_opnorm)
		end
	end
	@step "Building optimization model" config begin
		model = build_model(
			unname(𝒜),
			unname(acq_data.kspace_data),
			regularization;
			threaded=config.threaded,
			x₀,
			disable_normalop_optimization=config.disable_normalop_optimization,
		)
	end
	@printing_step "Reconstructing image" config begin
		if isnothing(config.freq)
			freq = config.verbose ? get_reasonable_freq(config.maxit) : -1
		else
			freq = config.freq
		end
		ϵ = eps(real(eltype(x₀)))
		tol = config.tol == 0 ? 0 : max(ϵ*10, config.tol * maximum(abs, x₀))
		stop =
			(iter, state) -> ProximalAlgorithms.default_stopping_criterion(tol, iter, state)
		display =
			(it, alg, iter, state) ->
				ProximalAlgorithms.default_display(it, alg, iter, state, config.printfunc)
		algorithm = patch_algorithm_with_default_values(algorithm)
		verbose = freq != -1
		x_var, _ = solve(model, algorithm; stop, maxit=config.maxit, freq, verbose, display)
		x = ~x_var
	end
	if !config.disable_inverse_scale_output && scale != 1
		@step "Inverse scaling image" config begin
			x .*= scale
		end
	end
	if acq_data.kspace_data isa NamedDimsArray
		img_dimnames = dimnames(𝒜, 2)
		x = NamedDimsArray{img_dimnames}(x)
	end
	return x
end

function get_reasonable_freq(maxit)
	reasonable_freqs = [1, 5, 10, 20, 50, 100]
	freq_i = findfirst(x -> x >= maxit ÷ 20, reasonable_freqs)
	return isnothing(freq_i) ? 100 : reasonable_freqs[freq_i]
end

function patch_algorithm_with_default_values(
	algorithm::ProximalAlgorithms.IterativeAlgorithm{T}
) where {
	T<:Union{
		ProximalAlgorithms.ForwardBackwardIteration,
		ProximalAlgorithms.FastForwardBackwardIteration,
	},
}
	if :Lf ∉ keys(algorithm.kwargs)
		return ProximalAlgorithms.override_parameters(algorithm; Lf=1)
	else
		return algorithm
	end
end

function patch_algorithm_with_default_values(
	algorithm::ProximalAlgorithms.IterativeAlgorithm{ProximalAlgorithms.ADMMIteration}
)
	if :cg_tol ∉ keys(algorithm.kwargs) && :cg_maxit ∉ keys(algorithm.kwargs)
		return ProximalAlgorithms.override_parameters(algorithm; cg_tol=1e-3, cg_maxit=10)
	elseif :cg_tol ∉ keys(algorithm.kwargs)
		return ProximalAlgorithms.override_parameters(algorithm; cg_tol=1e-3)
	elseif :cg_maxit ∉ keys(algorithm.kwargs)
		return ProximalAlgorithms.override_parameters(algorithm; cg_maxit=10)
	else
		return algorithm
	end
end

function patch_algorithm_with_default_values(
	algorithm::ProximalAlgorithms.IterativeAlgorithm
)
	return algorithm
end

function patch_algorithm_with_default_values(algorithm::Tuple)
	return map(patch_algorithm_with_default_values, algorithm)
end
