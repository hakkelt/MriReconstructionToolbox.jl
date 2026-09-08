"""
    ReconstructionConfig(; kwargs...)

Method-independent run settings for `reconstruct`: scaling, output, threading and
automatic task splitting.

Anything that only a particular method can act on — iteration counts, tolerances, the solver
algorithm — belongs on that method's constructor instead, not here. `ReconstructionConfig` rejects such
keywords rather than silently ignoring them.

Fields (with defaults):
- `scaling::Scaling = BartScaling()` — scaling applied to operators/data (see also `NoScaling`, `MeasurementBasedScaling`, `FixedScaling`)
- `verbosity::Verbosity = Verbose()` — output mode: `Silent()`, `ProgressBar()` or `Verbose()`
- `threaded::Bool = (Threads.nthreads() > 1)` — enable threaded execution when available
- `task_executor::Union{Nothing,ReconstructionExecutor} = nothing` — override executor for task splitting
- `disable_inverse_scale_output::Bool = false` — skip rescaling the final output
- `disable_task_splitting::Bool = false` — disable automatic task splitting
- `slice_id::Union{Nothing, String} = nothing` — set by the task-splitting machinery to name the
  slab currently being solved, and surfaced to an `on_iteration` callback as its `slice` field.
  Not meant to be passed by hand.

`verbosity` also accepts `true`/`false` and the symbols `:verbose`, `:progress`, `:silent`,
which are normalized to the corresponding [`Verbosity`](@ref) via `as_verbosity`.

Constructors:
- `ReconstructionConfig(; kwargs...)` — build from defaults, override selected fields
- `ReconstructionConfig(config::ReconstructionConfig; kwargs...)` — extend an existing config overriding selected fields

Examples
```julia
using MriReconstructionToolbox

# Default config
conf = ReconstructionConfig()

# A progress bar instead of the textual log
conf = ReconstructionConfig(; verbosity = ProgressBar())

# Extend an existing config
conf2 = ReconstructionConfig(conf; disable_task_splitting = true)

# Iteration control belongs to the method, not the config
x̂ = reconstruct(acq, IterativeReconstruction(reg; maxit = 50, tol = 1e-6); config = conf2)
```
"""
struct ReconstructionConfig
    scaling::Scaling
    verbosity::Verbosity
    threaded::Bool
    task_executor::Union{Nothing, ReconstructionExecutor}
    disable_inverse_scale_output::Bool
    disable_task_splitting::Bool
    slice_id::Union{Nothing, String}

    function ReconstructionConfig(;
            scaling::Scaling = BartScaling(),
            verbosity = Verbose(),
            threaded::Bool = nthreads() > 1,
            task_executor::Union{Nothing, ReconstructionExecutor} = nothing,
            disable_inverse_scale_output::Bool = false,
            disable_task_splitting::Bool = false,
            slice_id::Union{Nothing, AbstractString} = nothing,
        )
        return new(
            scaling,
            as_verbosity(verbosity),
            threaded,
            task_executor,
            disable_inverse_scale_output,
            disable_task_splitting,
            isnothing(slice_id) ? nothing : String(slice_id),
        )
    end
end

function ReconstructionConfig(config::ReconstructionConfig; kwargs...)
    new_kwargs = Dict{Symbol, Any}()
    for fn in fieldnames(ReconstructionConfig)
        if haskey(kwargs, fn)
            new_kwargs[fn] = kwargs[fn]
        else
            new_kwargs[fn] = getfield(config, fn)
        end
    end
    return ReconstructionConfig(; new_kwargs...)
end

function construct_config(kwargs)
    if haskey(kwargs, :config)
        config_to_extend = kwargs[:config]
        @argcheck config_to_extend isa ReconstructionConfig "The provided config must be of type ReconstructionConfig"
        kwargs_without_config = filter(kv -> kv[1] != :config, kwargs)
        check_kwargs(kwargs_without_config)
        return ReconstructionConfig(config_to_extend; kwargs_without_config...)
    else
        check_kwargs(kwargs)
        return ReconstructionConfig(; kwargs...)
    end
end

# Keywords that used to live on `ReconstructionConfig` but are properties of a method, not of a run. Naming
# them explicitly turns what would be "unknown keyword" into a message that says where the
# parameter went.
const _METHOD_OWNED_KWARGS = Dict{Symbol, String}(
    :maxit => "Pass `maxit` to the reconstruction method instead, e.g. `IterativeReconstruction(reg; maxit = 50)` or `POCS(; maxit = 20)`.",
    :tol => "Pass `tol` to the reconstruction method instead, e.g. `IterativeReconstruction(reg; tol = 1e-6)`.",
    :algorithm => "Pass `algorithm` to `IterativeReconstruction`, e.g. `IterativeReconstruction(reg; algorithm = FISTA())`.",
    :verbose => "Use `verbosity` instead: `verbosity = Verbose()` / `Silent()` / `ProgressBar()`.",
    :printfunc => "Use `verbosity = Verbose(; printfunc = ...)` instead.",
    :freq => "Use `verbosity = Verbose(; freq = ...)` instead.",
    :on_iteration => "Pass `on_iteration` to `IterativeReconstruction`, e.g. " *
        "`IterativeReconstruction(reg; on_iteration = IterationTrace())`. Only an iterative " *
        "method has iterations to observe, so the callback is method-owned rather than a run setting.",
)

function check_kwargs(kwargs)
    for key in keys(kwargs)
        if haskey(_METHOD_OWNED_KWARGS, key) && !hasfield(ReconstructionConfig, key)
            throw(ArgumentError("`$key` belongs to the reconstruction method, not `ReconstructionConfig`. $(_METHOD_OWNED_KWARGS[key])"))
        end
        @argcheck hasfield(ReconstructionConfig, key) "Unknown keyword argument: $key"
    end
    return nothing
end
