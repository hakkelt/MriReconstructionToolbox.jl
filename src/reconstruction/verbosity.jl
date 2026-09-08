"""
    Verbosity

Abstract supertype of the three progress-reporting modes a reconstruction can run in. Pass one
as the `verbosity` field of [`ReconstructionConfig`](@ref) (or as the `verbosity` keyword of `reconstruct`):

- [`Silent`](@ref) — no output at all.
- [`ProgressBar`](@ref) — a single progress bar, nothing else.
- [`Verbose`](@ref) — the textual step/timing log plus periodic solver output.

The same three modes apply to direct and iterative methods alike.
"""
abstract type Verbosity end

"""
    Silent() <: Verbosity

Produce no output whatsoever.
"""
struct Silent <: Verbosity end

"""
    ProgressBar(; output::IO = stderr, dt::Float64 = 0.1) <: Verbosity

Show a single progress bar and no textual log.

Exactly one bar is opened per `reconstruct` call. Its granularity is decided by
`progress_total`:

- A split task gets a bar over slices (this outranks everything else).
- A method with a countable loop (`IterativeReconstruction`, `POCS`, `SPIRiT`,
  `PhaseConstrained`, `GRAPPA`) gets a determinate bar over that loop.
- Everything else (`DirectReconstruction`, `Homodyne`) gets an indeterminate stage indicator
  that advances through the same phases [`Verbose`](@ref) would print.

# Fields
- `output::IO = stderr` — where the bar is drawn.
- `dt::Float64 = 0.1` — minimum interval in seconds between redraws.
"""
Base.@kwdef struct ProgressBar <: Verbosity
    output::IO = stderr
    dt::Float64 = 0.1
end

"""
    Verbose(; printfunc = println, freq = nothing, timing = true) <: Verbosity

Print the textual reconstruction log: one line per phase with its timing and allocations, plus
the iterative solver's own periodic output.

# Fields
- `printfunc::Function = println` — the sink every line is written through.
- `freq::Union{Nothing, Int} = nothing` — solver output frequency in iterations. `nothing`
  picks a reasonable value from the method's `maxit`, `0` prints only the final summary, and
  `-1` disables solver output entirely.
- `timing::Bool = true` — print MRT's own phase/timing messages. `false` keeps the solver
  output but drops everything else (this is what per-slice output uses).
"""
Base.@kwdef struct Verbose <: Verbosity
    printfunc::Function = println
    freq::Union{Nothing, Int} = nothing
    timing::Bool = true
end

# Internal: a `ProgressBar` whose meter has actually been opened by `with_progress`. Carrying the
# live meter in the verbosity object (rather than in a scoped value or an extra `ReconstructionConfig` field)
# is what lets every output site keep going through one `config.verbosity`, and what makes the
# "a bar is already open" check a plain dispatch instead of dynamic state.
struct ActiveProgress{M} <: Verbosity
    settings::ProgressBar
    meter::M
    determinate::Bool
end

"""
    as_verbosity(x) -> Verbosity

Normalize the `verbosity` keyword. A `Verbosity` passes through; `true`/`false` map to
`Verbose()`/`Silent()`; `:silent`, `:progress` and `:verbose` map to the corresponding types.
"""
as_verbosity(v::Verbosity) = v
as_verbosity(b::Bool) = b ? Verbose() : Silent()
function as_verbosity(s::Symbol)
    s === :silent && return Silent()
    (s === :progress || s === :progressbar || s === :bar) && return ProgressBar()
    s === :verbose && return Verbose()
    throw(
        ArgumentError(
            "Unknown verbosity $(repr(s)); expected :silent, :progress or :verbose, or a Verbosity instance"
        )
    )
end
as_verbosity(x) = throw(ArgumentError("Cannot interpret $(repr(x)) as a Verbosity"))

# --- the seam every output site goes through -------------------------------------------------

"""
    log_message(v::Verbosity, args...)

Emit one textual line. Only [`Verbose`](@ref) with `timing = true` prints anything.
"""
log_message(::Verbosity, args...) = nothing
log_message(v::Verbose, args...) = (v.timing && v.printfunc(args...); nothing)

"""
    should_log_steps(v::Verbosity) -> Bool

Whether the `@step` / `@printing_step` brackets should time their body and print it.
"""
should_log_steps(::Verbosity) = false
should_log_steps(v::Verbose) = v.timing

"""
    report_step(v::Verbosity, step_name)

Announce entering a named phase without timing it. Only an indeterminate [`ProgressBar`](@ref)
uses this: it is what turns the existing `@step` brackets into the stage indicator shown for
methods that have no countable loop.
"""
report_step(::Verbosity, step_name) = nothing
function report_step(v::ActiveProgress, step_name)
    v.determinate && return nothing
    v.meter.desc = uppercasefirst(String(step_name)) * "... "
    ProgressMeter.next!(v.meter)
    return nothing
end

"""
    progress_tick(v::Verbosity) -> Union{Nothing, Function}

The callback a determinate loop calls once per iteration, or `nothing` when no determinate bar
is open. Methods take it as the `progress` keyword of `_direct_reconstruct`.
"""
progress_tick(::Verbosity) = nothing
function progress_tick(v::ActiveProgress)
    v.determinate || return nothing
    return () -> (ProgressMeter.next!(v.meter); nothing)
end

_noop_display(it, alg, iter, state) = nothing

"""
    solver_output(v::Verbosity, maxit) -> (verbose::Bool, freq::Int, display::Function)

The `(verbose, freq, display)` triple handed to `StructuredOptimization.solve`. `Silent`
disables it, `Verbose` reproduces the textual iteration log, and an open determinate
`ProgressBar` turns it into one bar tick per iteration.
"""
solver_output(::Silent, maxit) = (false, -1, _noop_display)
solver_output(::ProgressBar, maxit) = (false, -1, _noop_display)

function solver_output(v::ActiveProgress, maxit)
    v.determinate || return (false, -1, _noop_display)
    meter = v.meter
    display = (it, alg, iter, state) -> (ProgressMeter.next!(meter); nothing)
    return (true, 1, display)
end

function solver_output(v::Verbose, maxit)
    freq = isnothing(v.freq) ? get_reasonable_freq(maxit) : v.freq
    display =
        (it, alg, iter, state) ->
    ProximalAlgorithms.default_display(it, alg, iter, state, v.printfunc)
    return (freq != -1, freq, display)
end

"""
    slice_verbosity(v::Verbosity, id; freq = -1) -> Verbosity

The verbosity used *inside* one slice of a task-split reconstruction. `Verbose` keeps the
solver output at `freq` and prefixes every line with the slice id, but drops its own phase
messages; every other mode goes `Silent`, since the slice-level bar owns the display.
"""
slice_verbosity(::Verbosity, id; freq::Int = -1) = Silent()
function slice_verbosity(v::Verbose, id; freq::Int = -1)
    return Verbose(; printfunc = (s...) -> v.printfunc("[$id] ", s...), freq, timing = false)
end

"""
    with_progress(f, v::Verbosity, total; desc = "Reconstructing")

Run `f(verbosity)` with a progress meter open when `v` is a [`ProgressBar`](@ref) that is not
already active, and with `v` unchanged otherwise. `total === nothing` opens an indeterminate
stage indicator instead of a determinate bar. The meter is always finished on exit, so early
convergence leaves no half-drawn bar behind.
"""
with_progress(f::Function, v::Verbosity, total; desc::AbstractString = "Reconstructing") = f(v)

function with_progress(
        f::Function, v::ProgressBar, total; desc::AbstractString = "Reconstructing"
    )
    meter, determinate = if isnothing(total)
        ProgressMeter.ProgressUnknown(; desc, output = v.output, dt = v.dt, spinner = true), false
    else
        ProgressMeter.Progress(total; desc, output = v.output, dt = v.dt), true
    end
    return try
        f(ActiveProgress(v, meter, determinate))
    finally
        ProgressMeter.finish!(meter)
    end
end

"""
    progress_total(method, acq_data) -> Union{Nothing, Int}

Number of ticks a determinate progress bar for `method` should have, or `nothing` when the
method has no countable loop and should get an indeterminate stage indicator instead.
"""
progress_total(::ReconstructionMethod, acq_data) = nothing
