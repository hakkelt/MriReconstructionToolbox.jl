"""
	_iterative_reconstruct_core(𝒜, acq_data, x₀_or_x₀s, scale, method, config; build)

Shared driver behind the single-variable and `Component` iterative reconstructions: k-space/warm-start
scaling, the operator-norm step-size estimate, model building (via `build`), solver setup and
solve, solution extraction, and inverse scaling. `build(𝒜, y; x₀)` must return `(model, vars,
auxiliaries)` as `build_model_with_variables`/`build_model` do; `vars` is either a single `Variable`
(single-variable path) or a `Tuple` of them (component path), and every step below that differs by
shape dispatches on that (`_scale_x0`, `_inv_scale`, `_max_abs`, `_n_vars`, `_extract_solution`).

`present(x)` turns a raw solver iterate — already inverse-scaled here — into the value the caller
would have got back from `reconstruct`: it applies the signal model and the `NamedDimsArray` /
`DecomposedImage` wrapping that the two paths do differently. It is used only to build the
`on_iteration` callback's `x`, so it is never called at all when no callback was supplied.
"""
function _iterative_reconstruct_core(
    𝒜, acq_data, x₀_or_x₀s, scale, method::IterativeReconstruction, config;
    build::Function, present::Function=identity,
)
    if scale != 1
        @step "Scaling k-space data" config begin
            acq_data = AcquisitionInfo(acq_data; kspace_data=acq_data.kspace_data ./ scale)
            # Solver iterates in scaled units, so warm start and tolerance must match.
            x₀_or_x₀s = _scale_x0(x₀_or_x₀s, scale)
        end
    end
    # `‖𝒜‖` is wanted only as a step size: a proximal algorithm needs `Lf`, not a unit-norm
    # operator. Scaling `𝒜` by `1/L` instead did three things at once — supplied the step size,
    # multiplied the effective regularization weight by `L`, and returned the image `L` times too
    # large — and only the first was intended. See `docs/src/high-level/methods.md`, "Operator
    # norm, step size and λ".
    should_estimate_L = _should_estimate_operator_norm(method)
    L = should_estimate_L ? _operator_norm_for_stepsize(𝒜, method, config) : nothing
    # `@printing_step`, not `@step`: `@step`'s verbose path runs its body inside `@spawn`, so the
    # `model` / `vars` bindings would live only in that task's closure — the solve closures below
    # capture them, and neither inference (JET) nor a reader can then see they are defined.
    @printing_step "Building optimization model" config begin
        model, vars, _auxiliaries = build(𝒜, acq_data.kspace_data; x₀=x₀_or_x₀s)
    end
    @printing_step "Reconstructing image" config begin
        verbose, freq, display = solver_output(config.verbosity, something(method.maxit, 100))
        # `method.maxit` / `method.tol` are `nothing` when the caller wants the algorithm's own
        # values: the corresponding keyword is then left out of the `solve` call entirely, because
        # `ProximalAlgorithms.override_parameters` merges what is passed here *last* and would
        # otherwise silently overwrite e.g. `algorithm = FISTA(maxit = 500)`.
        ϵ = eps(real(eltype(_first_x0(x₀_or_x₀s))))
        solver_kwargs = (; freq, verbose, display)
        if !isnothing(method.maxit)
            solver_kwargs = (; solver_kwargs..., maxit=method.maxit)
        end
        if !isnothing(method.tol)
            # MRT's `tol` is relative to the initial estimate; ProximalAlgorithms' is absolute.
            tol = method.tol == 0 ? 0 : max(ϵ * 10, method.tol * _max_abs(x₀_or_x₀s))
            stop =
                (iter, state) -> ProximalAlgorithms.default_stopping_criterion(tol, iter, state)
            solver_kwargs = (; solver_kwargs..., stop)
        end
        # For n variables sharing the same operator 𝒜, the data term is ‖𝒜*(x₁+…+xₙ) - y‖², whose
        # gradient has Lipschitz constant ‖[𝒜 … 𝒜]‖² = n‖𝒜‖², since ‖[𝒜 … 𝒜]‖ = √n‖𝒜‖. When the
        # norm was not estimated (`disable_operator_normalization`), let the algorithm derive its
        # own step size instead of overriding it.
        R_type = real(eltype(_first_x0(x₀_or_x₀s)))
        Lf = should_estimate_L ? R_type(_n_vars(vars) * L^2) : nothing
        algorithm = patch_algorithm_with_default_values(method.algorithm, Lf; eltype_real=R_type)
        # Only add `hook` to the keyword set when a callback was actually supplied: leaving it out
        # keeps the algorithm's `hook` field `Nothing`-typed, and `ProximalAlgorithms._run_hook`
        # then compiles to nothing at all inside the iteration loop.
        if !isnothing(method.on_iteration)
            hook = _iteration_hook(
                method.on_iteration, present,
                (!config.disable_inverse_scale_output && scale != 1) ? scale : nothing,
                config.slice_id,
            )
            solver_kwargs = (; solver_kwargs..., hook)
        end
        try
            # For a small single-slab solve, threading every operator is a ~1.4x net loss: no one
            # layer dominates (FFT-plan threading is ≈neutral at 128², a threaded BLAS-1 CG loop
            # is ≈noise, Polyester on the gradient stencils actually helps a little) — it is the
            # accumulated fork/join + budget-enter/exit + `@spawn` overhead of a few hundred small
            # threaded ops per solve that adds up. So narrow *every* pool for the duration. A
            # low-rank prox is the exception: its level-3 SVDs thread 3.2x-3.9x, so those solves
            # keep the threaded budget. See `uses_blas3` / `with_serial_blas`.
            if uses_blas3(method.regularization)
                solve(model, algorithm; solver_kwargs...)
            elseif _work_item_bytes(_first_x0(x₀_or_x₀s)) < serial_blas_threshold_bytes()
                with_restricted_threads_if_needed() do
                    solve(model, algorithm; solver_kwargs...)
                end
            else
                with_serial_blas(_first_x0(x₀_or_x₀s)) do
                    solve(model, algorithm; solver_kwargs...)
                end
            end
        catch e
            if e isa ErrorException && occursin("cannot parse this problem for solver", e.msg)
                reg_types = map(typeof, ensure_tuple(method.regularization))
                throw(
                    ArgumentError(
                        "Cannot parse problem for algorithm $(typeof(algorithm)). " *
                            "Data fidelity: $(typeof(method.fidelity)), Regularization: $(reg_types). " *
                            "Check that the objective satisfies the solver assumptions."
                    )
                )
            else
                rethrow(e)
            end
        end
        # Read the solution from the image variable(s) themselves: once a regularization contributes
        # auxiliary variables (e.g. total generalized variation), the solver returns them alongside the
        # image and its ordering is not something to depend on.
        x = _extract_solution(vars)
    end
    if !config.disable_inverse_scale_output && scale != 1
        @step "Inverse scaling image" config begin
            x = _inv_scale(x, scale)
        end
    end
    return x
end

_scale_x0(x₀::AbstractArray, scale) = x₀ ./ scale
_scale_x0(x₀s::Tuple, scale) = map(x -> x ./ scale, x₀s)

_inv_scale(x::AbstractArray, scale) = x .* scale
_inv_scale(xs::Tuple, scale) = map(x -> x .* scale, xs)

_max_abs(x₀::AbstractArray) = maximum(abs, x₀)
_max_abs(x₀s::Tuple) = maximum(x -> maximum(abs, x), x₀s)

_first_x0(x₀::AbstractArray) = x₀
_first_x0(x₀s::Tuple) = x₀s[1]

_n_vars(::Variable) = 1
_n_vars(vars::Tuple) = length(vars)

# Solution extraction: defensive copy guarantees the returned reconstructed array is never
# aliased to internal solver buffers or caller-provided initial guesses (TODO 6: measured memory delta
# is ~0.21% of total solve allocation, well below the 2% threshold, protecting against aliasing bugs).
_extract_solution(x_var::Variable) = copy(~x_var)
_extract_solution(vars::Tuple) = map(v -> copy(~v), vars)

"""
    _iteration_hook(on_iteration, present, scale, slice_id) -> Function

The `hook(k, alg, iter, state)` handed to `ProximalAlgorithms`, wrapping the user's
`on_iteration` callback.

The iterate the solver holds is in the solver's own (scaled) units and is a bare array that the
solver keeps writing into, so it is copied, inverse-scaled and put through `present` before the
callback sees it — a callback that received the internal buffer could neither compare against a
reference image nor keep it. `scale === nothing` means the caller asked for no inverse scaling
(the `disable_inverse_scale_output` path), and then the copy comes from `present` alone.

The wall clock is `time_ns`, which is monotonic; `t₀` is read when the hook is built, immediately
before `solve`, so `elapsed_ns` measures solver time and excludes the operator build and the
operator-norm estimate.
"""
function _iteration_hook(on_iteration, present::Function, scale, slice_id)
    t₀ = time_ns()
    return function (k, alg, iter, state)
        raw = alg.solution(iter, state)
        x = present(isnothing(scale) ? _copy_iterate(raw) : _inv_scale(raw, scale))
        base = (; iteration=k, x=x, elapsed_ns=time_ns() - t₀)
        info = merge(base, _iteration_metrics(iter, state))
        on_iteration(isnothing(slice_id) ? info : merge(info, (; slice=slice_id)))
        return nothing
    end
end

_copy_iterate(x::AbstractArray) = copy(x)
_copy_iterate(xs::Tuple) = map(copy, xs)

"""
    _iteration_metrics(iter, state) -> NamedTuple

The algorithm-specific part of an `on_iteration` callback's payload, read off the solver state.
A field is present only where the algorithm computes the quantity: the generic fallback is the
empty `NamedTuple`, so an algorithm without convergence metrics simply contributes none rather
than a payload full of `nothing`s. The names mirror the columns `Verbose` prints, translated to
the vocabulary of `docs/src/high-level/algorithms.md`.
"""
_iteration_metrics(iter, state) = (;)

function _iteration_metrics(
    ::Union{
        ProximalAlgorithms.ForwardBackwardIteration,
        ProximalAlgorithms.FastForwardBackwardIteration,
    }, state,
)
    return (;
        objective=state.f_x + state.g_z,
        smooth_value=state.f_x,
        nonsmooth_value=state.g_z,
        stepsize=state.gamma,
        fixed_point_residual=norm(state.res, Inf) / state.gamma,
    )
end

function _iteration_metrics(iter::ProximalAlgorithms.DouglasRachfordIteration, state)
    return (;
        objective=state.f_y + state.g_z,
        smooth_value=state.f_y,
        nonsmooth_value=state.g_z,
        fixed_point_residual=norm(state.res, Inf) / iter.gamma,
    )
end

# `rᵏ_norm` / `sᵏ_norm` are per-block vectors (one entry per splitting block); reducing them with
# `maximum` keeps the payload's field types the same whatever the problem's block structure is,
# which is what lets a trace of these be collected into a concrete vector.
function _iteration_metrics(::ProximalAlgorithms.ADMMIteration, state)
    return (;
        primal_residual=maximum(state.rᵏ_norm),
        dual_residual=maximum(state.sᵏ_norm),
        iterate_change=state.Δx_norm,
    )
end

_iteration_metrics(::ProximalAlgorithms.AbstractCGIteration, state) = (; residual_norm=sqrt(state.r²))

function _iteration_metrics(::ProximalAlgorithms.POGMIteration, state)
    return (;
        objective=state.f_x + state.g_z,
        smooth_value=state.f_x,
        nonsmooth_value=state.g_z,
        stepsize=state.gamma,
        fixed_point_residual=norm(state.res, Inf) / state.gamma,
    )
end

function get_reasonable_freq(maxit)
    reasonable_freqs = [1, 5, 10, 20, 50, 100]
    freq_i = findfirst(x -> x >= maxit ÷ 20, reasonable_freqs)
    return isnothing(freq_i) ? 100 : reasonable_freqs[freq_i]
end

_is_krylov_solver(::ProximalAlgorithms.IterativeAlgorithm{<:Union{ProximalAlgorithms.CGIteration,ProximalAlgorithms.CGNRIteration}}) = true
_is_krylov_solver(::Union{Type{<:ProximalAlgorithms.CGIteration},Type{<:ProximalAlgorithms.CGNRIteration}}) = true
_is_krylov_solver(algs::Tuple) = all(_is_krylov_solver, algs)
_is_krylov_solver(::Any) = false

"""
    _should_estimate_operator_norm(method) -> Bool

Whether `‖𝒜‖` is worth computing for this method, i.e. whether the algorithm takes an `Lf`
step-size hint at all. A pure unregularized CG/CGNR solve does not: Krylov subspaces are scale
invariant, so it derives everything it needs itself.

Reads `method.disable_operator_normalization`, whose name predates the change that stopped this
rescaling the operator — it now suppresses the `Lf` estimate and nothing else.
"""
function _should_estimate_operator_norm(method::IterativeReconstruction)
    if !isnothing(method.disable_operator_normalization)
        return !method.disable_operator_normalization
    end
    is_pure_cg = isempty(method.regularization) && _is_krylov_solver(method.algorithm)
    return !is_pure_cg
end

# `‖𝒜‖`, for use as `Lf = n‖𝒜‖²`. `estimate_opnorm`'s power iteration converges from below, so
# this is a slight under-estimate of the true norm; `AbstractOperators.powerit`'s docstring
# records that, and `exact_opnorm = true` swaps in the converged `opnorm` for callers who mind.
function _operator_norm_for_stepsize(𝒜, method::IterativeReconstruction, config)
    local L
    # `@printing_step`, not `@step`: the latter runs its body in a `@spawn`, so `L` would be
    # bound only inside that task's closure.
    @printing_step "Estimating the operator norm" config begin
        L = method.exact_opnorm ? LinearAlgebra.opnorm(𝒜) : AbstractOperators.estimate_opnorm(𝒜)
    end
    @argcheck L != 0 "Cannot reconstruct with an encoding operator of zero norm"
    return L
end
