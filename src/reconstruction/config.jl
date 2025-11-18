"""
    Config(; kwargs...)

Global reconstruction configuration used by `reconstruct` and internal routines.
It centralizes iteration control, tolerances, normalization, threading, and
automatic problem decomposition behavior.

Fields (with defaults):
- normalization::Normalization = BartScaling() — scaling applied to operators/data
- tol::Float64 = 1e-4 — stopping tolerance for iterative algorithms
- maxit::Int = 100 — maximum iterations for the chosen solver
- freq::Union{Nothing,Int} = nothing — progress print frequency (iterations)
- verbose::Bool = true — enable/disable logging output
- threaded::Bool = (Threads.nthreads() > 1) — enable threaded execution when available
- exact_opnorm::Bool = false — use exact operator norm for stepsize estimation
- decomposition_executor::Union{Nothing,ReconstructionExecutor} = nothing — override executor for decomposition
- disable_inverse_scale_output::Bool = false — skip rescaling the final output
- disable_normalop_optimization::Bool = false — disable normal-operator optimizations
- disable_problem_decomposition::Bool = false — disable automatic problem decomposition
- disable_operator_normalization::Bool = false — disable operator normalization
- printfunc::Function = println — custom logging function

Constructors:
- `Config(; kwargs...)` — build from defaults, override selected fields
- `Config(config::Config; kwargs...)` — extend an existing config overriding selected fields

Examples
```julia
using MriReconstructionToolbox

# Default config
conf = Config()

# Custom tolerances and iterations
conf = Config(; tol=1e-5, maxit=200, verbose=false)

# Extend an existing config
conf2 = Config(conf; maxit=50, disable_problem_decomposition=true)

# Use with reconstruct (keywords still override config fields)
x̂ = reconstruct(acq; config=conf2, maxit=25)
```
"""
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
		kwargs_without_config = filter(kv -> kv[1] != :config, kwargs)
		check_kwargs(kwargs_without_config)
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
