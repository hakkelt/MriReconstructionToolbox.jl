abstract type ReconstructionExecutor end
struct SequentialExecutor <: ReconstructionExecutor end
struct MultiThreadingExecutor <: ReconstructionExecutor end

function execute(f::Function, plan, acq_data, config)
    executor = suggest_executor(plan, config)
    return execute(f, plan, acq_data, config, executor)
end

function execute(f::Function, plan, acq_data, config, executor::ReconstructionExecutor)
    maybe_print_task_splitting_info(plan, config)
    batch_sizes = plan.variable_size[collect(plan.variable_batch_dims)]
    slices = collect(get_slices(plan, acq_data))
    slice_threaded = slice_threading(plan, acq_data, config, executor)
    scales = Array{real(eltype(acq_data.kspace_data))}(undef, batch_sizes)

    # A split run's bar counts slices, not iterations: it is the only granularity that is
    # meaningful across every method, and `slice_verbosity` silences the slices so no inner bar
    # can open underneath it. `ProgressMeter.next!` is lock-guarded, so the threaded executor
    # ticking from several slices at once is safe.
    return with_progress(config.verbosity, length(slices); desc="Slices ") do verbosity
        conf = ReconstructionConfig(config; verbosity)
        tick = progress_tick(verbosity)

        # A slice's result type isn't known until `f` actually runs (it depends on the acquisition
        # and warm-start array types), so -- mirroring `execute_two_phase`'s own `prelim` idiom --
        # the first slice runs outside the (possibly threaded) loop to learn it, and `results` is
        # then allocated concretely instead of as `Array{AbstractArray}`.
        first_idx, first_id, first_local_acq = slices[1]
        first_r, first_s = run_first_item(@view(slices[2:end]), conf, executor; threaded=slice_threaded) do
            execute_single_slice(
                f, first_idx, first_id, first_local_acq, conf; threaded=slice_threaded
            )
        end
        isnothing(tick) || tick()
        results = Array{typeof(first_r)}(undef, batch_sizes)
        results[first_idx] = first_r
        scales[first_idx] = first_s

        run_slices!(
            results, scales, f, @view(slices[2:end]), conf, executor;
            threaded=slice_threaded, tick,
        )
        maybe_rescale_results!(results, scales, conf)
        stack_image_slices(results, plan, Val(conf.threaded))
    end
end

function run_slices!(
    results, scales, f, slices, config, executor::ReconstructionExecutor; threaded, tick=nothing
)
    for_each_item!(slices, config, executor; threaded) do (idx, id, local_acq)
        r, s = execute_single_slice(f, idx, id, local_acq, config; threaded)
        store_item!(results, idx, r, id)
        scales[idx] = s
        isnothing(tick) || tick()
    end
    return nothing
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
        # Planned properly (not `fast_planning`), because this same operator is reused for the
        # iterative solve in phase 2 below -- otherwise phase 2 would plan an equivalent operator
        # again from scratch.
        𝒜 = build_encoding_operator(local_acq, method; threaded=false, fast_planning=false)
        warm_start, scale, L = _direct_reconstruct(𝒜, local_acq, local_x₀, method, local_conf)
        return warm_start, scale, 𝒜, L
    end
    solve_slice = function (local_acq, warm_start, ratio, global_scale, local_conf, 𝒜, L)
        local_reg = map(r -> scale_regularization(r, ratio), method.regularization)
        local_method = _with_regularization(method, local_reg)
        result, _ = _reconstruct(
            local_acq, local_method, warm_start, local_conf;
            scale_override=global_scale, 𝒜, precomputed_L=L,
        )
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
        𝒜 = build_encoding_operator(local_acq, method; threaded=false, fast_planning=false)
        x̂, scale, L = _direct_reconstruct_components(𝒜, local_acq, method, local_conf)
        return get_component_x0s(method.regularization, x̂, local_x₀), scale, 𝒜, L
    end
    solve_slice = function (local_acq, x₀s, ratio, global_scale, local_conf, 𝒜, L)
        local_components = map(c -> scale_regularization(c, ratio), method.regularization)
        local_method = _with_regularization(method, local_components)
        result, _ = _reconstruct_components(
            local_acq, local_method, nothing, local_conf;
            scale_override=global_scale, x₀s, 𝒜, precomputed_L=L,
        )
        return result
    end
    return execute_two_phase(plan, acq_data, config, prepare, solve_slice)
end

# Shared skeleton of the two-phase scheme described above. `prepare(idx, local_acq, local_conf)` returns
# `(warm_start, scale, 𝒜, L)` for one slice -- `𝒜` is the fully-planned encoding operator phase 1
# already had to build to get the warm start, cached here so phase 2 does not plan an equivalent
# one again; `L` is the operator-norm estimate phase 1 used to scale-correct that warm start
# (`nothing` when not estimated), cached the same way so phase 2's step-size estimate does not
# repeat it. `solve(local_acq, warm_start, ratio, global_scale, local_conf, 𝒜, L)` solves that
# slice under the shared scale, with its regularization compensated by `ratio`, reusing both.
function execute_two_phase(plan, acq_data, config, prepare::Function, solve::Function)
    executor = suggest_executor(plan, config)
    maybe_print_task_splitting_info(plan, config)
    batch_sizes = plan.variable_size[collect(plan.variable_batch_dims)]
    slices = collect(get_slices(plan, acq_data))

    slice_threaded = slice_threading(plan, acq_data, config, executor)

    # Both phases visit every slice, so the bar counts `2 * length(slices)` ticks.
    return with_progress(config.verbosity, 2 * length(slices); desc="Slices ") do verbosity
        conf = ReconstructionConfig(config; verbosity)
        tick = progress_tick(verbosity)
        slice_config = (id) -> ReconstructionConfig(
            conf;
            verbosity=slice_verbosity(verbosity, id; freq=-1),
            threaded=slice_threaded,
            slice_id=id,
        )

        # `prelim`'s element type isn't known until `prepare` actually runs (it depends on the acquisition
        # and warm-start array types), so the first slice is run outside the (possibly threaded) loop to
        # learn it; `prelim` is then allocated concretely instead of as `Array{Any}`, keeping the phase-2
        # unpacking below type-stable.
        first_idx, first_id, first_local_acq = slices[1]
        first_warm_start, first_scale, first_𝒜, first_L =
            run_first_item(@view(slices[2:end]), conf, executor; threaded=slice_threaded) do
                prepare(first_idx, first_local_acq, slice_config(first_id))
            end
        isnothing(tick) || tick()
        first_prelim = (first_id, first_local_acq, first_warm_start, first_scale, first_𝒜, first_L)
        prelim = Array{typeof(first_prelim)}(undef, batch_sizes)
        prelim[first_idx] = first_prelim
        for_each_item!(@view(slices[2:end]), conf, executor; threaded=slice_threaded) do (idx, id, local_acq)
            warm_start, scale, 𝒜, L = prepare(idx, local_acq, slice_config(id))
            store_item!(prelim, idx, (id, local_acq, warm_start, scale, 𝒜, L), id)
            isnothing(tick) || tick()
        end

        global_scale = robust_global_scale(vec(map(p -> p[4], prelim)))
        log_message(
            verbosity, @sprintf("Using shared scaling factor across slices: %g", global_scale)
        )

        indices = vec(collect(CartesianIndices(batch_sizes)))
        first_result_idx = indices[1]
        first_res_id, first_res_acq, first_res_warm_start, first_res_scale, first_res_𝒜, first_res_L = prelim[first_result_idx]
        first_res_ratio = safe_scale_ratio(first_res_scale, global_scale)
        first_result = run_first_item(@view(indices[2:end]), conf, executor; threaded=slice_threaded) do
            solve(
                first_res_acq, first_res_warm_start, first_res_ratio, global_scale,
                slice_config(first_res_id), first_res_𝒜, first_res_L,
            )
        end
        isnothing(tick) || tick()
        results = Array{typeof(first_result)}(undef, batch_sizes)
        results[first_result_idx] = first_result
        for_each_item!(@view(indices[2:end]), conf, executor; threaded=slice_threaded) do idx
            id, local_acq, warm_start, scale, 𝒜, L = prelim[idx]
            ratio = safe_scale_ratio(scale, global_scale)
            store_item!(results, idx, solve(local_acq, warm_start, ratio, global_scale, slice_config(id), 𝒜, L), id)
            isnothing(tick) || tick()
        end

        stack_image_slices(results, plan, Val(conf.threaded))
    end
end

# `threaded` is the *work item's* threading decision (`slice_threading`), not `config.threaded`:
# opening every pool around a loop whose body was just gated serial is the dead weight this
# scope exists to avoid.
function for_each_item!(
    f!::Function, items, config, ::SequentialExecutor; threaded=config.threaded
)
    @conditionally_enable_threading threaded for item in items
        f!(item)
    end
    return nothing
end

# `threaded` is accepted for a uniform call site and ignored: here the slice loop itself is the
# parallelism, and `@budgeted_threads` decides its own budget.
function for_each_item!(
    f!::Function, items, config, ::MultiThreadingExecutor; threaded=false
)
    @budgeted_threads for item in items
        f!(item)
    end
    return nothing
end

# Both schemes below run the first item outside the loop to learn its concrete result type. That
# item must still see the threading scope `for_each_item!` opens around the rest, or it runs at a
# different budget from every other item: unrestricted where the sequential loop restricts every
# pool, or without NFFT's guarded pool where the loop enables it. `loop_items` is the collection
# the loop will iterate (the remaining items), because that is what `@budgeted_threads` derives
# its per-worker budget from; the first item's own work is `f()`.
function run_first_item(f::Function, loop_items, config, ::SequentialExecutor; threaded=config.threaded)
    return @conditionally_enable_threading threaded f()
end

function run_first_item(f::Function, loop_items, config, ::MultiThreadingExecutor; threaded=false)
    return with_thread_budget(f, budget_for(loop_items))
end

"""
    store_item!(dest, idx, value, id)

Write one item's result into the concretely-typed array the hoisted first item sized. The array's
element type is `typeof(first_result)`, so a later item producing a different concrete type would
otherwise surface as a bare `convert`/`MethodError` from inside a threaded loop. Check it here so
the error names the slice and both types instead.
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
        verbosity=slice_verbosity(v, id; freq),
        slice_id=id,
        disable_inverse_scale_output=true, kwargs...,
    )
    return f(idx, local_acq, local_conf)
end

"""
    slice_threading(plan, acq_data, config, executor) -> Bool

Whether the work *inside* one slice of a task-split reconstruction may thread.

Two independent reasons to say no:

  - A [`MultiThreadingExecutor`](@ref) already occupies every thread with whole slices, so the
    work inside a slice must run sequentially.
  - A [`SequentialExecutor`](@ref) runs slices one at a time, but a slice is by construction
    smaller than the whole problem, and below [`serial_blas_threshold_bytes`](@ref) threading a
    work item that small is a net loss — the same predicate
    `maybe_disable_unsplit_threading` applies to an unsplit problem, here applied per
    slice. Without this the outer scope is serial only around the *solve*
    (`with_restricted_threads` in `solve_core.jl`), leaving the per-slice operator build, the
    adjoint and the operator-norm estimate threaded over a work item too small to pay for it.

The per-slice byte count comes from `plan` (batch dimensions collapsed to one) and the k-space
element type, both known before any slice runs; nothing is measured at run time.
"""
function slice_threading(plan, acq_data, config, executor::ReconstructionExecutor)
    executor isa MultiThreadingExecutor && return false
    return _should_thread_work_item(config, slice_bytes(plan, acq_data))
end

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

function suggest_executor(plan, config)
    if !isnothing(config.task_executor)
        return config.task_executor
    elseif config.threaded && length(plan) > nthreads()
        return MultiThreadingExecutor()
    else
        return SequentialExecutor()
    end
end
