"""
	_iterative_reconstruct_core(𝒜, acq_data, x₀_or_x₀s, scale, method, config; build)

Shared driver behind the single-variable and `Component` iterative reconstructions: k-space/warm-start
scaling, the `disable_operator_normalization` guard, model building (via `build`), solver setup and
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
    if !method.disable_operator_normalization
        @step "Normalizing encoding operator" config begin
            𝒜 = normalize_op(𝒜, method.exact_opnorm)
        end
    end
    @step "Building optimization model" config begin
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
        # For n variables sharing the same operator 𝒜, the data term is ‖𝒜*(x₁+…+xₙ) - y‖²; its
        # Lipschitz constant is n (not 1) when 𝒜 is normalized to unit norm, since
        # ‖[𝒜 … 𝒜]‖ = √n‖𝒜‖. When 𝒜 was left at its natural norm (disabled normalization), this
        # estimate no longer holds; let the algorithm derive its own instead of overriding it.
        Lf = method.disable_operator_normalization ? nothing : _n_vars(vars)
        algorithm = patch_algorithm_with_default_values(method.algorithm, Lf)
        verbose = freq != -1
        solve(model, algorithm; stop, maxit = config.maxit, freq, verbose, display)
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
