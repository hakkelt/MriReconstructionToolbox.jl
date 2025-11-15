Base.@kwdef struct Config
	normalization::Normalization = BartScaling()
	tol::Float64 = 1e-4
	maxit::Int = 100
	freq::Union{Nothing,Int} = nothing
	verbose::Bool = true
	threaded::Bool = nthreads() > 1
	exact_opnorm::Bool = false
	decomposition_executor::Union{Nothing,ReconstructionExecutor} = nothing
	disable_inverse_scale_output::Bool = false
	disable_normalop_optimization::Bool = false
	disable_problem_decomposition::Bool = false
	disable_operator_normalization::Bool = false
	printfunc::Function = println
end

function Config(config::Config; kwargs...)
    new_kwargs = Dict{Symbol,Any}()
	for fn in fieldnames(Config)
		if haskey(kwargs, fn)
			new_kwargs[fn] = kwargs[fn]
		else
			new_kwargs[fn] = getfield(config, fn)
		end
	end
	return Config(; new_kwargs...)
end

function construct_config(kwargs)
	if haskey(kwargs, :config)
		config_to_extend = kwargs[:config]
		@argcheck config_to_extend isa Config "The provided config must be of type Config"
		check_kwargs(kwargs)
		return Config(config_to_extend; kwargs_without_config...)
	else
		check_kwargs(kwargs)
		return Config(; kwargs...)
	end
end

function check_kwargs(kwargs)
	for key in keys(kwargs)
		@argcheck hasfield(Config, key) "Unknown keyword argument: $key"
	end
end
