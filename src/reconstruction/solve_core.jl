"""
	_iterative_reconstruct_core(𝒜, acq_data, x₀_or_x₀s, scale, method, config; build)

Shared driver behind the single-variable and `Component` iterative reconstructions: k-space/warm-start
scaling, the operator-norm step-size estimate, model building (via `build`), solver setup and
solve, solution extraction, and inverse scaling. `build(𝒜, y; x₀)` must return `(model, vars,
auxiliaries)` as `build_model_with_variables`/`build_model` do; `vars` is either a single `Variable`
(single-variable path) or a `Tuple` of them (component path), and every step below that differs by
shape dispatches on that (`_scale_x0`, `_inv_scale`, `_max_abs`, `_n_vars`, `_extract_solution`).
"""
function _iterative_reconstruct_core(
        𝒜, acq_data, x₀_or_x₀s, scale, method::IterativeReconstruction, config; build::Function
    )
    if scale != 1
        @step "Scaling k-space data" config begin
            acq_data = AcquisitionInfo(acq_data; kspace_data = acq_data.kspace_data ./ scale)
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
        model, vars, _auxiliaries = build(𝒜, acq_data.kspace_data; x₀ = x₀_or_x₀s)
    end
    @printing_step "Reconstructing image" config begin
        if isnothing(config.freq)
            freq = config.verbose ? get_reasonable_freq(config.maxit) : -1
        else
            freq = config.freq
        end
        ϵ = eps(real(eltype(_first_x0(x₀_or_x₀s))))
        tol = config.tol == 0 ? 0 : max(ϵ * 10, config.tol * _max_abs(x₀_or_x₀s))
        stop =
            (iter, state) -> ProximalAlgorithms.default_stopping_criterion(tol, iter, state)
        display =
            (it, alg, iter, state) ->
        ProximalAlgorithms.default_display(it, alg, iter, state, config.printfunc)
        # For n variables sharing the same operator 𝒜, the data term is ‖𝒜*(x₁+…+xₙ) - y‖², whose
        # gradient has Lipschitz constant ‖[𝒜 … 𝒜]‖² = n‖𝒜‖², since ‖[𝒜 … 𝒜]‖ = √n‖𝒜‖. When the
        # norm was not estimated (`disable_operator_normalization`), let the algorithm derive its
        # own step size instead of overriding it.
        R_type = real(eltype(_first_x0(x₀_or_x₀s)))
        Lf = should_estimate_L ? R_type(_n_vars(vars) * L^2) : nothing
        algorithm = patch_algorithm_with_default_values(method.algorithm, Lf; eltype_real = R_type)
        verbose = freq != -1
        try
            # For a small single-slab solve, threading every operator is a ~1.4x net loss: no one
            # layer dominates (FFT-plan threading is ≈neutral at 128², a threaded BLAS-1 CG loop
            # is ≈noise, Polyester on the gradient stencils actually helps a little) — it is the
            # accumulated fork/join + budget-enter/exit + `@spawn` overhead of a few hundred small
            # threaded ops per solve that adds up. So narrow *every* pool for the duration. A
            # low-rank prox is the exception: its level-3 SVDs thread 3.2x-3.9x, so those solves
            # keep the threaded budget. See `uses_blas3` / `with_serial_blas`.
            if uses_blas3(method.regularization)
                solve(model, algorithm; stop, maxit = config.maxit, freq, verbose, display)
            elseif _work_item_bytes(_first_x0(x₀_or_x₀s)) < SERIAL_BLAS_THRESHOLD_BYTES
                with_restricted_threads_if_needed() do
                    solve(model, algorithm; stop, maxit = config.maxit, freq, verbose, display)
                end
            else
                with_serial_blas(_first_x0(x₀_or_x₀s)) do
                    solve(model, algorithm; stop, maxit = config.maxit, freq, verbose, display)
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

function get_reasonable_freq(maxit)
    reasonable_freqs = [1, 5, 10, 20, 50, 100]
    freq_i = findfirst(x -> x >= maxit ÷ 20, reasonable_freqs)
    return isnothing(freq_i) ? 100 : reasonable_freqs[freq_i]
end

_is_krylov_solver(::ProximalAlgorithms.IterativeAlgorithm{<:Union{ProximalAlgorithms.CGIteration, ProximalAlgorithms.CGNRIteration}}) = true
_is_krylov_solver(::Union{Type{<:ProximalAlgorithms.CGIteration}, Type{<:ProximalAlgorithms.CGNRIteration}}) = true
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
