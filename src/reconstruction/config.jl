"""
    ReconstructionConfig(; kwargs...)

Method-independent run settings for `reconstruct`: scaling, output, threading and
automatic task splitting.

Anything that only a particular method can act on — iteration counts, tolerances, the solver
algorithm — belongs on that method's constructor instead, not here. `ReconstructionConfig` rejects such
keywords rather than silently ignoring them.

Fields (with defaults):
- `scaling::Scaling = QuantileScaling()` — scaling applied to operators/data (see also `NoScaling`,
  `BartScaling`, `MaxScaling`, `StdScaling`, `NoiseLevelScaling`, `MeasurementBasedScaling`,
  `KSpaceNormScaling`, `SystemMatrixBasedScaling`, `FixedScaling`)
- `verbosity::Verbosity = Silent()` — output mode: `Silent()`, `ProgressBar()` or `Verbose()`
- `threaded::Bool = (Threads.nthreads() > 1)` — enable threaded execution when available. A
  reconstruction whose k-space is in device (GPU) memory never threads: the device kernels are
  the parallelism.
- `task_executor::Union{Nothing,ReconstructionExecutor} = nothing` — override executor for task
  splitting. `MultiThreadingExecutor` is rejected for a device reconstruction.
- `disable_inverse_scale_output::Bool = false` — skip rescaling the final output
- `disable_task_splitting::Union{Nothing,Bool} = nothing` — disable automatic task splitting.
  `nothing` picks the default for where the k-space lives: splitting on the host,
  [`DEVICE_DISABLES_TASK_SPLITTING`](@ref) on a device.
- `fft_planning::Symbol = :auto` — how carefully FFTW plans the encoding operator's FFTs:
  `:measure` times candidate algorithms and finds faster plans at a planning cost of about
  0.1–0.3 s per 2D transform and 1–1.5 s per 3D one; `:estimate` plans instantly from a
  heuristic whose plans can run several times slower; `:auto` weighs the one against the other
  for the reconstruction at hand (see "Performance & Threading" in the manual). A measured plan
  is remembered for the rest of the session and in the on-disk wisdom cache, so `:measure` also
  pays off when the same acquisition is reconstructed many times, even by short or direct
  reconstructions. Has no effect under FFTW.jl's `mkl` provider, which ignores planner flags.
- `slice_id::Union{Nothing, String} = nothing` — set by the task-splitting machinery to name the
  slab currently being solved, and surfaced to an `on_iteration` callback as its `slice` field.
  Not meant to be passed by hand.

`verbosity` also accepts the symbols `:verbose`, `:progress` and `:silent`, which are normalized
to the corresponding [`Verbosity`](@ref) via `as_verbosity`.

Constructors:
- `ReconstructionConfig(; kwargs...)` — build from defaults, override selected fields
- `ReconstructionConfig(config::ReconstructionConfig; kwargs...)` — extend an existing config overriding selected fields

Examples
```julia
using MriReconstructionToolbox

# Default config
conf = ReconstructionConfig()

# A reconstruction is silent by default; ask for a progress bar or the textual log
conf = ReconstructionConfig(; verbosity = ProgressBar())
conf = ReconstructionConfig(; verbosity = Verbose())

# Measure FFT plans once for a series of reconstructions of the same acquisition
conf = ReconstructionConfig(; fft_planning = :measure)

# Extend an existing config
conf2 = ReconstructionConfig(conf; disable_task_splitting = true)

# Iteration control belongs to the method, not the config
x̂ = reconstruct(acq, IterativeReconstruction(reg; maxit = 50, reltol = 1e-6); config = conf2)
```
"""
const FFT_PLANNING_MODES = (:auto, :estimate, :measure)

struct ReconstructionConfig
    scaling::Scaling
    verbosity::Verbosity
    threaded::Bool
    task_executor::Union{Nothing, ReconstructionExecutor}
    disable_inverse_scale_output::Bool
    disable_task_splitting::Union{Nothing, Bool}
    fft_planning::Symbol
    slice_id::Union{Nothing, String}

    function ReconstructionConfig(;
            scaling::Scaling = QuantileScaling(),
            verbosity = Silent(),
            threaded::Bool = nthreads() > 1,
            task_executor::Union{Nothing, ReconstructionExecutor} = nothing,
            disable_inverse_scale_output::Bool = false,
            disable_task_splitting::Union{Nothing, Bool} = nothing,
            fft_planning::Symbol = :auto,
            slice_id::Union{Nothing, AbstractString} = nothing,
        )
        @argcheck fft_planning in FFT_PLANNING_MODES "fft_planning must be one of $FFT_PLANNING_MODES"
        return new(
            scaling,
            as_verbosity(verbosity),
            threaded,
            task_executor,
            disable_inverse_scale_output,
            disable_task_splitting,
            fft_planning,
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

"""
    DEVICE_DISABLES_TASK_SPLITTING

The `disable_task_splitting` a device (GPU) reconstruction gets when the caller leaves it at
`nothing`. Slices run one after another on a device, so splitting only pays where a smaller
problem is cheaper per element, which a device kernel is not: each slice pays its own kernel
launches, operator build and scaling pass.
"""
const DEVICE_DISABLES_TASK_SPLITTING = true

"""
    resolve_config(config, acq_data) -> ReconstructionConfig

`config` with the settings that depend on where the k-space lives made concrete: on a device
`threaded` is off and `disable_task_splitting = nothing` becomes
[`DEVICE_DISABLES_TASK_SPLITTING`](@ref); on the host `nothing` becomes `false`.
"""
function resolve_config(config::ReconstructionConfig, acq_data)
    device = _is_device(acq_data)
    if device && config.task_executor isa MultiThreadingExecutor
        throw(
            ArgumentError(
                "MultiThreadingExecutor cannot run a reconstruction whose k-space is in device memory " *
                    "($(nameof(_array_type_of(acq_data)))); its slices would share one device. " *
                    "Use SequentialExecutor() or leave `task_executor` unset."
            )
        )
    end
    disable_task_splitting = _task_splitting_disabled(config, acq_data)
    threaded = config.threaded && !device
    disable_task_splitting === config.disable_task_splitting && threaded == config.threaded && return config
    return ReconstructionConfig(config; disable_task_splitting, threaded)
end

_task_splitting_disabled(config::ReconstructionConfig, acq_data) =
    something(config.disable_task_splitting, _is_device(acq_data) && DEVICE_DISABLES_TASK_SPLITTING)

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
    :tol => "The stopping tolerance is `reltol`, and it belongs to the reconstruction method, e.g. `IterativeReconstruction(reg; reltol = 1e-6)`.",
    :reltol => "Pass `reltol` to the reconstruction method instead, e.g. `IterativeReconstruction(reg; reltol = 1e-6)`.",
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
