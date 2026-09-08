"""
    IterationTrace([reduction])

Collector for [`IterativeReconstruction`](@ref)'s `on_iteration` callback: records one entry per
solver iteration, so that a single `reconstruct` call yields both a convergence-vs-iteration and a
convergence-vs-wall-clock curve.

`reduction` is applied to the current image estimate and its result is what gets stored. The
default, `identity`, keeps every intermediate image — convenient but `maxit` images' worth of
memory, so for a convergence curve pass the metric instead:

```julia
trace = IterationTrace(x -> nrmse(x, reference))
```

# Fields
- `iterations::Vector{Int}` — 1-based index of each recorded iteration.
- `times::Vector{Float64}` — seconds since the solve started, from the monotonic `time_ns` clock.
  The clock starts after the encoding operator and the operator-norm estimate are built, so it
  times the solver and not the setup.
- `values::Vector` — `reduction(x)` per iteration. Concretely typed after the first entry.
- `metrics::Vector{NamedTuple}` — the algorithm-specific part of each callback payload
  (`objective`, `primal_residual`, `residual_norm`, …); see [`IterativeReconstruction`](@ref) for
  which algorithm supplies which.
- `slices::Vector{String}` — the slab each entry came from, empty unless the reconstruction was
  split into tasks.

# Thread safety

Under task splitting the slabs solve concurrently and the callback fires from several tasks at
once. Every entry is appended under the trace's own lock, so the vectors stay consistent, but the
*order* of entries is then the order the tasks happened to reach them: group by `slices` before
plotting a split run.

# Example
```julia
using MriReconstructionToolbox

nrmse(x, ref) = sqrt(sum(abs2, x .- ref) / sum(abs2, ref))

trace = IterationTrace(x -> nrmse(x, reference))
x̂ = reconstruct(
    acq, IterativeReconstruction(L1Wavelet2D(0.01); algorithm = FISTA(), maxit = 60, on_iteration = trace)
)

trace.iterations   # 1:60
trace.times        # seconds into the solve
trace.values       # NRMSE at each iteration
```
"""
mutable struct IterationTrace{F}
    const reduction::F
    const lock::ReentrantLock
    const iterations::Vector{Int}
    const times::Vector{Float64}
    values::Vector
    const metrics::Vector{NamedTuple}
    const slices::Vector{String}
end

function IterationTrace(reduction = identity)
    return IterationTrace(
        reduction, ReentrantLock(), Int[], Float64[], Any[], NamedTuple[], String[]
    )
end

# The keys the trace unpacks into its own columns; whatever else the payload carries is
# algorithm-specific and goes to `metrics` verbatim.
const _TRACE_OWN_KEYS = NamedTuple{(:iteration, :x, :elapsed_ns, :slice)}

function (trace::IterationTrace)(info::NamedTuple)
    # Outside the lock: an expensive reduction (an NRMSE over a whole volume, say) must not
    # serialise the slabs of a task-split run against each other.
    value = trace.reduction(info.x)
    elapsed = info.elapsed_ns / 1.0e9
    metrics = Base.structdiff(info, _TRACE_OWN_KEYS)
    @lock trace.lock begin
        # `values` starts out `Vector{Any}` because the reduction's return type is unknown until
        # it has run once; narrowing it here is what makes `trace.values` a concrete vector, ready
        # to plot, in the ordinary case of a scalar metric.
        if isempty(trace.values)
            trace.values = Vector{typeof(value)}()
        elseif !(value isa eltype(trace.values))
            trace.values = Any[trace.values...]
        end
        push!(trace.iterations, info.iteration)
        push!(trace.times, elapsed)
        push!(trace.values, value)
        push!(trace.metrics, metrics)
        haskey(info, :slice) && push!(trace.slices, info.slice)
    end
    return nothing
end

Base.length(trace::IterationTrace) = length(trace.iterations)
Base.isempty(trace::IterationTrace) = isempty(trace.iterations)

function Base.empty!(trace::IterationTrace)
    @lock trace.lock begin
        empty!(trace.iterations)
        empty!(trace.times)
        trace.values = Any[]
        empty!(trace.metrics)
        empty!(trace.slices)
    end
    return trace
end

function Base.show(io::IO, trace::IterationTrace)
    n = length(trace)
    print(io, "IterationTrace(", trace.reduction, "): ", n, " iteration", n == 1 ? "" : "s")
    isempty(trace.slices) || print(io, " over ", length(unique(trace.slices)), " slices")
    return
end
