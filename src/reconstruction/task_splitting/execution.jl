abstract type ReconstructionExecutor end
struct SequentialExecutor <: ReconstructionExecutor end
struct MultiThreadingExecutor <: ReconstructionExecutor end

function execute(f::Function, plan, acq_data, config)
    executor = suggest_executor(plan, acq_data, config)
    return execute(f, plan, acq_data, config, executor)
end

function execute(f::Function, plan, acq_data, config, executor::ReconstructionExecutor)
    maybe_print_task_splitting_info(plan, config)
    batch_sizes = plan.variable_size[collect(plan.variable_batch_dims)]
    slices = collect(get_slices(plan, acq_data))
    slice_threaded = slice_threading(config, executor)
    scales = Array{real(eltype(acq_data.kspace_data))}(undef, batch_sizes)

    # A split run's bar counts slices, not iterations: it is the only granularity that is
    # meaningful across every method, and `slice_verbosity` silences the slices so no inner bar
    # can open underneath it. `ProgressMeter.next!` is lock-guarded, so the threaded executor
    # ticking from several slices at once is safe.
    return with_progress(config.verbosity, length(slices); desc = "Slices ") do verbosity
        conf = ReconstructionConfig(config; verbosity)
        tick = progress_tick(verbosity)

        solved = map_items(slices, map(s -> s[2], slices), conf, executor; threaded = slice_threaded) do (idx, id, local_acq)
            r = execute_single_slice(f, idx, id, local_acq, conf; threaded = slice_threaded)
            isnothing(tick) || tick()
            r
        end
        results = Array{fieldtype(eltype(solved), 1)}(undef, batch_sizes)
        for ((idx, _, _), (r, s)) in zip(slices, solved)
            results[idx] = r
            scales[idx] = s
        end
        maybe_rescale_results!(results, scales, conf)
        stack_image_slices(results, plan, Val(conf.threaded))
    end
end


# Regularized task splitting: regularization strength (λ) is scale-dependent, so each slice
# must be normalized before the regularization term is applied. But if each slice used its own
# scale for the final output too, slice-to-slice intensity would vary with noisy per-slice scale
# estimates instead of the true (similar) signal levels. So: estimate each slice's own scale first
# (phase 1), then solve every slice using one shared `global_scale` for both k-space data and the
# final image - giving uniform output - while compensating λ per slice by `scale_i / global_scale`
# so the regularization behaves as if that slice had been normalized by its own scale (see
# `scale_regularization`).
function execute_regularized(plan, acq_data, config, method::IterativeReconstruction, x₀)
    prepare = function (idx, local_acq, local_conf)
        local_x₀ = isnothing(x₀) ? nothing : get_x₀_slice(x₀, plan, idx)
        # Planned for the iterative solve (`_fast_planning(method, …)`), not for the one adjoint
        # here, because this same operator is reused in phase 2 below -- otherwise phase 2 would
        # plan an equivalent operator again from scratch.
        𝒜 = build_encoding_operator(
            local_acq, method; threaded = local_conf.threaded,
            fast_planning = _fast_planning(method, local_acq, local_conf),
        )
        warm_start, scale, prior = _direct_reconstruct(𝒜, local_acq, local_x₀, method, local_conf)
        return warm_start, scale, 𝒜, prior
    end
    solve_slice = function (local_acq, warm_start, ratio, global_scale, local_conf, 𝒜, prior)
        local_reg = map(r -> scale_regularization(r, ratio), method.regularization)
        local_method = _with_regularization(method, local_reg)
        result, _ = _reconstruct(
            local_acq, local_method, warm_start, local_conf;
            scale_override = global_scale, 𝒜, prior,
        )
        _release_device_plans!(𝒜, _storage_template(local_acq))
        return result
    end
    return execute_two_phase(plan, acq_data, config, prepare, solve_slice)
end

# Same two-phase scheme as `execute_regularized`, for a component (multi-variable)
# reconstruction: phase 1 gets each slice's own scale from a plain direct estimate,
# phase 2 solves every slice under one shared `global_scale`, with each component's
# regularization compensated by `scale_i / global_scale` (`scale_regularization`).
function execute_regularized_components(plan, acq_data, config, method::IterativeReconstruction, x₀)
    prepare = function (idx, local_acq, local_conf)
        local_x₀ = isnothing(x₀) ? nothing : slice_x₀_components(x₀, plan, idx)
        𝒜 = build_encoding_operator(
            local_acq, method; threaded = local_conf.threaded,
            fast_planning = _fast_planning(method, local_acq, local_conf),
        )
        x̂, scale, prior = _direct_reconstruct_components(𝒜, local_acq, method, local_conf)
        return get_component_x0s(method.regularization, x̂, local_x₀), scale, 𝒜, prior
    end
    solve_slice = function (local_acq, x₀s, ratio, global_scale, local_conf, 𝒜, prior)
        local_components = map(c -> scale_regularization(c, ratio), method.regularization)
        local_method = _with_regularization(method, local_components)
        result, _ = _reconstruct_components(
            local_acq, local_method, nothing, local_conf;
            scale_override = global_scale, x₀s, 𝒜, prior,
        )
        _release_device_plans!(𝒜, _storage_template(local_acq))
        return result
    end
    return execute_two_phase(plan, acq_data, config, prepare, solve_slice)
end

# Shared skeleton of the two-phase scheme described above. `prepare(idx, local_acq, local_conf)` returns
# `(warm_start, scale, 𝒜, prior)` for one slice -- `𝒜` is the fully-planned encoding operator phase 1
# already had to build to get the warm start, cached here so phase 2 does not plan an equivalent
# one again; `prior` is what phase 1 computed while forming that warm start (`_warm_start_prior`:
# the operator-norm estimate, the curvature, the warm-start divisor), cached the same way so phase 2
# does not repeat it. `solve(local_acq, warm_start, ratio, global_scale, local_conf, 𝒜, prior)`
# solves that slice under the shared scale, with its regularization compensated by `ratio`,
# reusing both.
function execute_two_phase(plan, acq_data, config, prepare::Function, solve::Function)
    executor = suggest_executor(plan, acq_data, config)
    maybe_print_task_splitting_info(plan, config)
    batch_sizes = plan.variable_size[collect(plan.variable_batch_dims)]
    slices = collect(get_slices(plan, acq_data))

    slice_threaded = slice_threading(config, executor)

    # Both phases visit every slice, so the bar counts `2 * length(slices)` ticks.
    return with_progress(config.verbosity, 2 * length(slices); desc = "Slices ") do verbosity
        conf = ReconstructionConfig(config; verbosity)
        tick = progress_tick(verbosity)
        slice_config = (id) -> ReconstructionConfig(
            conf;
            verbosity = slice_verbosity(verbosity, id; freq = -1),
            threaded = slice_threaded,
            slice_id = id,
        )

        ids = map(s -> s[2], slices)
        prepared = map_items(slices, ids, conf, executor; threaded = slice_threaded) do (idx, id, local_acq)
            warm_start, scale, 𝒜, prior = prepare(idx, local_acq, slice_config(id))
            isnothing(tick) || tick()
            (id, local_acq, warm_start, scale, 𝒜, prior)
        end
        prelim = Array{eltype(prepared)}(undef, batch_sizes)
        for ((idx, _, _), p) in zip(slices, prepared)
            prelim[idx] = p
        end

        global_scale = robust_global_scale(vec(map(p -> p[4], prelim)))
        log_message(
            verbosity, @sprintf("Using shared scaling factor across slices: %g", global_scale)
        )

        indices = vec(collect(CartesianIndices(batch_sizes)))
        solved = map_items(indices, map(idx -> prelim[idx][1], indices), conf, executor; threaded = slice_threaded) do idx
            id, local_acq, warm_start, scale, 𝒜, prior = prelim[idx]
            ratio = safe_scale_ratio(scale, global_scale)
            r = solve(local_acq, warm_start, ratio, global_scale, slice_config(id), 𝒜, prior)
            isnothing(tick) || tick()
            r
        end
        results = reshape(solved, batch_sizes...)

        stack_image_slices(results, plan, Val(conf.threaded))
    end
end

# `threaded` is the *work item's* threading decision (`slice_threading`), not `config.threaded`:
# a loop whose slices run serially inside opens no pools.
function for_each_item!(
        f!::Function, items, config, ::SequentialExecutor; threaded = config.threaded
    )
    @conditionally_enable_threading threaded for item in items
        f!(item)
    end
    return nothing
end

# `threaded` is accepted for a uniform call site and ignored: here the slice loop itself is the
# parallelism, and `@budgeted_threads` decides its own budget.
function for_each_item!(
        f!::Function, items, config, ::MultiThreadingExecutor; threaded = false
    )
    @budgeted_threads for item in items
        f!(item)
    end
    return nothing
end

"""
    map_items(f, items, ids, config, executor; threaded) -> Vector

`f(item)` for every item, in a vector of the results' common concrete type; `ids` names each item
for the error a result of another type raises (see [`store_item!`](@ref)).

A slice's result type is not known until it runs (it depends on the acquisition and warm-start
array types). Under a [`SequentialExecutor`](@ref) the first item runs ahead of the rest to size the
vector, which costs nothing, since the slices run one at a time anyway. Under a
[`MultiThreadingExecutor`](@ref) a slice run ahead would run alone, with its operator built
unthreaded, while every other thread waited: with 8 slices on 2 threads that is 4.5 slice-times
for 4 slices' work. There every item runs in the loop instead, into a `Vector{Any}` that is
narrowed afterwards.
"""
function map_items(f::F, items, ids, config, executor::SequentialExecutor; threaded = config.threaded) where {F}
    first = run_first_item(@view(items[2:end]), config, executor; threaded) do
        f(items[1])
    end
    out = Vector{typeof(first)}(undef, length(items))
    out[1] = first
    for_each_item!(2:length(items), config, executor; threaded) do k
        store_item!(out, k, f(items[k]), ids[k])
    end
    return out
end

function map_items(f::F, items, ids, config, executor::MultiThreadingExecutor; threaded = false) where {F}
    raw = Vector{Any}(undef, length(items))
    for_each_item!(eachindex(items), config, executor; threaded) do k
        raw[k] = f(items[k])
    end
    out = Vector{typeof(raw[1])}(undef, length(items))
    for k in eachindex(raw)
        store_item!(out, k, raw[k], ids[k])
    end
    return out
end

# The first item of a sequential `map_items` runs outside the loop, so it must see the same
# restricted/full scope `for_each_item!` opens around the rest, or it runs unrestricted where the
# loop restricts every pool (or without NFFT's guarded pool where the loop enables it).
function run_first_item(f::Function, loop_items, config, ::SequentialExecutor; threaded = config.threaded)
    return @conditionally_enable_threading threaded f()
end

"""
    store_item!(dest, idx, value, id)

Write one item's result into the concretely-typed array [`map_items`](@ref) sized from its first
item. The array's element type is `typeof(first_result)`, so a later item producing a different
concrete type would otherwise surface as a bare `convert`/`MethodError`. Check it here so the error
names the slice and both types instead.
"""
function store_item!(dest::AbstractArray{T}, idx, value, id) where {T}
    value isa T || throw(
        ArgumentError(
            "slice $id produced a $(typeof(value)), but the first slice produced a $T. Task " *
                "splitting allocates its result array from the first slice's concrete type, so " *
                "every slice must agree; a slice-dependent k-space, warm-start or operator-norm " *
                "type is the usual cause."
        )
    )
    dest[idx] = value
    return nothing
end

function execute_single_slice(f::Function, idx, id, local_acq, config; kwargs...)
    v = config.verbosity
    # A slice keeps the solver's own periodic output (prefixed with the slice id) but not the
    # phase log; `freq = 0` is the "final summary only" default this path has always used.
    freq = v isa Verbose && !isnothing(v.freq) ? v.freq : 0
    local_conf = ReconstructionConfig(
        config;
        verbosity = slice_verbosity(v, id; freq),
        slice_id = id,
        disable_inverse_scale_output = true, kwargs...,
    )
    return f(idx, local_acq, local_conf)
end

"""
    slice_threading(config, executor) -> Bool

Whether the work *inside* one slice of a task-split reconstruction may thread.

There is one reason to say no, and it is about the threads, not about the size: a
[`MultiThreadingExecutor`](@ref) already occupies every thread with whole slices, so the work
inside a slice would be competing with its own siblings and must run sequentially.

A [`SequentialExecutor`](@ref) runs slices one at a time, leaving the threads free, so it passes
`config.threaded` through unchanged. It deliberately does **not** also apply a size gate: how
small is too small to thread is a per-kernel question that the operator answers with its own
input, through `AbstractOperators.threading_threshold` and `ProximalOperators.should_thread`. A
blanket gate here would override all of them at once, which is what it used to do — and what
kept the coil-fused encoding operator from ever being built on a 2-D slice. See `solve_core.jl`
for the measurements.
"""
slice_threading(config, executor::ReconstructionExecutor) =
    config.threaded && !(executor isa MultiThreadingExecutor)

"""
    slice_bytes(plan, acq_data) -> Int

Size in bytes of one slice's image variable under `plan`: `plan.variable_size` with every batch
dimension collapsed to one, times the size of a k-space element.
"""
function slice_bytes(plan, acq_data)
    per_slice = prod(
        ntuple(
            d -> d in plan.variable_batch_dims ? 1 : plan.variable_size[d],
            length(plan.variable_size),
        )
    )
    return per_slice * sizeof(eltype(acq_data.kspace_data))
end

"""
    suggest_executor(plan, acq_data, config) -> ReconstructionExecutor

Pick the executor for a task-split reconstruction, unless `config.task_executor` names one.

Slices are independent solves, so spreading them over threads is MRT's primary parallelism and
the only one that pays on the problem sizes this package sees: a slice has to reach
[`serial_blas_threshold_bytes`](@ref) before threading *inside* it returns anything, and a 2-D
slice of a clinical volume is two orders of magnitude below that (320² `ComplexF32` = 800 KiB
against a 16 MiB threshold). Two conditions, therefore:

  - **more than one slice**, since a single slice has nothing to spread; and
  - **slices too small to thread internally**, or at least as many slices as threads. Only a
    large slice can use the threads by itself, and then it is not obvious that `length(plan)`-way
    outer parallelism beats full inner parallelism, so the tie goes to the executor that keeps
    every thread busy.

Measured on the 3-D FSE knee (320×320 per slice, `ComplexF32`, 8 threads, exclusive test node),
1 thread → 8 threads, with the condition below (`after`) and with the `length(plan) > nthreads()`
it replaced (`before`):

| slices | CG-SENSE before | CG-SENSE after   | TV (20 it) before | TV (20 it) after   |
|--------|-----------------|------------------|-------------------|--------------------|
|  1     | 77.9 → 75.2 ms  | 83.1 → 78.7 ms   |   584 →  590 ms   |   650 →  585 ms    |
|  4     | 304 → 328 ms    | 331 → 231 ms     |  2535 → 2481 ms   |  2583 → 1627 ms    |
|  8     | 770 → 727 ms    | 832 → 404 ms     |  4926 → 4955 ms   |  5074 → 3073 ms    |
| 16     | 1564 → 682 ms   | 1338 → 752 ms    | 10128 → 4880 ms   |  9722 → 5666 ms    |
| 24     | 2164 → 917 ms   | 2147 → 1089 ms   | 14392 → 7149 ms   | 14406 → 8287 ms    |

The rows at 4 and 8 are the ones the old condition got wrong: 8 slices on 8 threads — a perfect
one-slice-per-thread split — fell through to the sequential executor and scaled 1.06x, and
everything at or below the thread count did the same. They now scale 2.06x and 1.65x. Rows 16 and
24 already took the threaded path and are unchanged within run-to-run noise.

The ceiling is ~2x rather than ~8x because a slice's solve is memory-bandwidth bound (the FFT and
the `SignAlternation` passes around it), so eight cores do not buy eight times the throughput.
"""
function suggest_executor(plan, acq_data, config)
    isnothing(config.task_executor) || return config.task_executor
    config.threaded && length(plan) > 1 || return SequentialExecutor()
    fits_in_one_thread = !_should_thread_work_item(config, slice_bytes(plan, acq_data))
    return (fits_in_one_thread || length(plan) >= nthreads()) ?
        MultiThreadingExecutor() : SequentialExecutor()
end
