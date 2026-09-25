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
        build::Function, present::Function = identity, precomputed_L = nothing,
    )
    if scale != 1
        @step "Scaling k-space data" config begin
            acq_data = AcquisitionInfo(acq_data; kspace_data = _scale_kspace(acq_data.kspace_data, scale))
            # Solver iterates in scaled units, so warm start and tolerance must match.
            x₀_or_x₀s = _scale_x0(x₀_or_x₀s, scale)
        end
    end
    # `‖𝒜‖` is wanted only as a step size: a proximal algorithm needs `Lf`, not a unit-norm
    # operator. Scaling `𝒜` by `1/L` instead did three things at once — supplied the step size,
    # multiplied the effective regularization weight by `L`, and returned the image `L` times too
    # large — and only the first was intended. See `docs/src/high-level/methods.md`, "Operator
    # norm, step size and λ".
    # `@printing_step`, not `@step`: `@step`'s verbose path runs its body inside `@spawn`, so the
    # `model` / `vars` bindings would live only in that task's closure — the solve closures below
    # capture them, and neither inference (JET) nor a reader can then see they are defined.
    @printing_step "Building optimization model" config begin
        model, vars, _auxiliaries = build(𝒜, _measurement(acq_data.kspace_data); x₀ = x₀_or_x₀s)
    end
    # A tuple of algorithms is resolved to the one `solve` would run before anything is estimated
    # for it: whether `‖𝒜‖` is needed depends on that algorithm alone.
    selected_algorithm = _select_algorithm(model, method.algorithm)
    should_estimate_L = _should_estimate_operator_norm(method, selected_algorithm)
    # `precomputed_L` lets a caller that already estimated `‖𝒜‖` for the warm start
    # (`_direct_reconstruct`/`_direct_reconstruct_components`) hand it in instead of paying for
    # `estimate_opnorm` a second time here.
    L = should_estimate_L ?
        (isnothing(precomputed_L) ? _operator_norm_for_stepsize(𝒜, method, config) : precomputed_L) :
        nothing
    @printing_step "Reconstructing image" config begin
        verbose, freq, display = solver_output(config.verbosity, something(method.maxit, 100))
        # `method.maxit` / `method.reltol` are `nothing` when the caller wants the algorithm's own
        # values: the corresponding keyword is then left out of the `solve` call entirely, because
        # `ProximalAlgorithms.override_parameters` merges what is passed here *last* and would
        # otherwise silently overwrite e.g. `algorithm = FISTA(maxit = 500)`.
        ϵ = eps(real(eltype(_first_x0(x₀_or_x₀s))))
        solver_kwargs = (; freq, verbose, display)
        if !isnothing(method.maxit)
            solver_kwargs = (; solver_kwargs..., maxit = method.maxit)
        end
        if !isnothing(method.reltol)
            # MRT's `reltol` is relative to the initial estimate; ProximalAlgorithms' `tol` is
            # absolute, and this is where the one is turned into the other.
            tol = method.reltol == 0 ? 0 : max(ϵ * 10, method.reltol * _max_abs(x₀_or_x₀s))
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
        algorithm = patch_algorithm_with_default_values(selected_algorithm, Lf; eltype_real = R_type)
        algorithm = _scale_admm_penalty(
            algorithm, 𝒜, x₀_or_x₀s, acq_data, L, method, config; eltype_real = R_type,
        )
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
            # BLAS is narrowed for the duration; nothing else is.
            #
            # Whether a *kernel* should thread is the operator's own call, made per input by
            # `AbstractOperators.threading_threshold` and `ProximalOperators.should_thread`, so
            # this scope leaves FFTW, NFFT and Polyester alone and lets them decide. BLAS is the
            # one pool with no such policy: an iterative solve drives it almost entirely through
            # level-1 calls on one work item, which are memory-bandwidth-bound and gain nothing
            # from a thread team, while each call still pays to start one.
            #
            # A solve used to run every sub-16-MiB problem with *every* pool pinned instead, which
            # silently serialised each `@batch` kernel and stopped
            # `_coil_fused_encoding_operator` from being built at all (it gates on `threaded`).
            # Dropping the size gate but keeping BLAS narrow is what the measurements support.
            # Real data, 256x256 single-coil, TV-ADMM 20 it, 8 threads, min of 3:
            #
            # | scope around the solve        |   ms |
            # |-------------------------------|------|
            # | nothing narrowed              | 584.0 |
            # | every pool pinned             | 247.4 |
            # | counted pools pinned          | 243.1 |
            # | BLAS/MKL pinned (this)        | 233.3 |
            # | everything but Polyester      | 231.4 |
            #
            # i.e. BLAS alone accounts for the whole 2.5x, and narrowing anything further buys
            # nothing. `threaded = false` for the same case is 285.8 ms, so building the operators
            # threaded and pinning BLAS beats turning threading off wholesale.
            #
            # The narrowing is a soft default, not a hard limit, so the calls that are worth
            # threading take BLAS back for themselves: a large factorization
            # (`ProximalOperators.with_factorization_threads`, and the per-block SVDs of the
            # low-rank family through it), a large `gemm` (`AbstractOperators.BLAS3_THREAD_WORK`)
            # and a large CG step (`ProximalAlgorithms.CG_BLAS_THREAD_BYTES`). A low-rank solve
            # used to skip the narrowing altogether for its SVDs, leaving every level-1 call
            # threaded as well: OpenBLAS, 8 threads, locally-low-rank cine, 533 ms against 294 ms
            # with the narrowing on.
            with_serial_blas() do
                solve(model, algorithm; solver_kwargs...)
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
        base = (; iteration = k, x = x, elapsed_ns = time_ns() - t₀)
        info = merge(base, _iteration_metrics(iter, state))
        on_iteration(isnothing(slice_id) ? info : merge(info, (; slice = slice_id)))
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
            ProximalAlgorithms.POGMIteration,
        }, state,
    )
    return (;
        objective = state.f_x + state.g_z,
        smooth_value = state.f_x,
        nonsmooth_value = state.g_z,
        stepsize = state.gamma,
        fixed_point_residual = norm(state.res, Inf) / state.gamma,
    )
end

function _iteration_metrics(iter::ProximalAlgorithms.DouglasRachfordIteration, state)
    return (;
        objective = state.f_y + state.g_z,
        smooth_value = state.f_y,
        nonsmooth_value = state.g_z,
        fixed_point_residual = norm(state.res, Inf) / iter.gamma,
    )
end

# `rᵏ_norm` / `sᵏ_norm` are per-block vectors (one entry per splitting block); reducing them with
# `maximum` keeps the payload's field types the same whatever the problem's block structure is,
# which is what lets a trace of these be collected into a concrete vector.
function _iteration_metrics(::ProximalAlgorithms.ADMMIteration, state)
    return (;
        primal_residual = maximum(state.rᵏ_norm),
        dual_residual = maximum(state.sᵏ_norm),
        iterate_change = state.Δx_norm,
    )
end

_iteration_metrics(::ProximalAlgorithms.AbstractCGIteration, state) = (; residual_norm = sqrt(state.r²))

function get_reasonable_freq(maxit)
    reasonable_freqs = [1, 5, 10, 20, 50, 100]
    freq_i = findfirst(x -> x >= maxit ÷ 20, reasonable_freqs)
    return isnothing(freq_i) ? 100 : reasonable_freqs[freq_i]
end

# `AbstractCGIteration`, not the two concrete unpreconditioned types: `PCGIteration` and
# `PCGNRIteration` (`CG(; P)` / `CGNR(; P)`) are Krylov methods too, and naming only the plain
# pair silently sent every preconditioned solve down the proximal path -- paying `estimate_opnorm`
# for an `Lf` hint that a Krylov subspace never reads.
_is_krylov_solver(::ProximalAlgorithms.IterativeAlgorithm{<:ProximalAlgorithms.AbstractCGIteration}) = true
_is_krylov_solver(::Type{<:ProximalAlgorithms.AbstractCGIteration}) = true
_is_krylov_solver(algs::Tuple) = all(_is_krylov_solver, algs)
_is_krylov_solver(::Any) = false

"""
    _should_estimate_operator_norm(method) -> Bool

Whether `‖𝒜‖` is worth computing for this method, i.e. whether the algorithm takes an `Lf`
step-size hint at all. A pure unregularized CG/CGNR solve does not: Krylov subspaces are scale
invariant, so it derives everything it needs itself.

Neither does ADMM, and that is [`consumes_lf`](@ref)'s job to say. This used to ask only whether
the solve was a *pure* Krylov one, so every regularized solve paid `estimate_opnorm` — including
every ADMM solve, whose `patch_algorithm_with_default_values` method has always thrown the `Lf` it
was handed away. The estimate's only remaining consumer there was the warm-start scale, which
[`_warm_start_scale_proxy`](@ref) supplies for one operator application instead of twenty.

Reads `method.disable_operator_normalization`, whose name predates the change that stopped this
rescaling the operator — it now suppresses the `Lf` estimate and nothing else.

`algorithm` is the algorithm that will actually run. For a tuple of candidates that is the one
[`_select_algorithm`](@ref) resolves once the model exists; asked about the tuple itself, the
answer is `true` whenever *any* candidate takes `Lf`, which is what the default tuple always
says (it contains `POGM`) even when `CGNR` or `ADMM` is the one selected.
"""
function _should_estimate_operator_norm(method::IterativeReconstruction, algorithm = method.algorithm)
    if !isnothing(method.disable_operator_normalization)
        return !method.disable_operator_normalization
    end
    is_pure_cg = isempty(method.regularization) && _is_krylov_solver(algorithm)
    return !is_pure_cg && consumes_lf(algorithm)
end

"""
    _select_algorithm(model, algorithm)

The algorithm `solve(model, algorithm)` runs: `algorithm` itself, or for a tuple of candidates
the first that the model parses into (`StructuredOptimization.select_solver`, which builds
nothing to decide). A tuple none of whose members fits is returned unchanged, so `solve` still
reports why.
"""
_select_algorithm(model, algorithm) = algorithm
_select_algorithm(model, algorithms::Tuple) = something(select_solver(model, algorithms), algorithms)

"""
    _warm_start_needs_operator_norm(method) -> Bool

Whether the default warm start `𝒜'y` needs the `‖𝒜‖²` correction (`_direct_reconstruct` /
`_direct_reconstruct_components`) to be on the image's scale. Unlike
[`_should_estimate_operator_norm`](@ref), this does **not** auto-skip pure Krylov solvers: a
Krylov method derives its own step size regardless of warm-start scale, but a badly-scaled warm
start still costs it iterations before `maxit`/`reltol` are reached (the CG-SENSE case this fixes).
Only an explicit `disable_operator_normalization = true` skips it, preserving today's behavior for
callers who deliberately opted out of the operator-norm estimate altogether.
"""
_warm_start_needs_operator_norm(method::IterativeReconstruction) = method.disable_operator_normalization !== true

"""
	_scale_default_warm_start(𝒜, x̂, method, config) -> (x̂, L_or_nothing)

Put the default warm start `x̂ = 𝒜'y` on the image's scale, and return the operator norm if one
was computed. `𝒜'y` is only on the image's scale when `𝒜'𝒜 ≈ I`, which a raw FFT/NFFT is not, so
the warm start is divided by `ρ(𝒜'𝒜)` — one Landweber step.

This is the one place that decision is made, for both the single-variable and the component path
(`_direct_reconstruct`, `_direct_reconstruct_components`). The returned `L` is non-`nothing`
exactly when the algorithm needs it as its own step-size hint, in which case the caller threads it
on as `precomputed_L` so `_iterative_reconstruct_core` does not estimate it twice; when only the
warm start needed a scale, [`_warm_start_scale_proxy`](@ref) supplies it for one operator
application and there is no `L` to carry.

A tuple of algorithms is decided here as a whole, because which of them runs is only known once
the model exists (see [`_select_algorithm`](@ref)): if any of them takes `Lf`, `L` is estimated
and scales the warm start. The proxy would be cheaper, but it lands below `ρ(𝒜'𝒜)` where the
estimate lands above, and CG-SENSE with the default tuple converges measurably slower from it.
"""
function _scale_default_warm_start(𝒜, x̂, method::IterativeReconstruction, config)
    _warm_start_needs_operator_norm(method) || return x̂, nothing
    if _should_estimate_operator_norm(method)
        L = _operator_norm_for_stepsize(𝒜, method, config)
        return _scale_x0(x̂, L^2), L
    end
    return _scale_x0(x̂, _warm_start_scale_proxy(𝒜, x̂, config)), nothing
end

"""
	_warm_start_scale_proxy(𝒜, x̂, config) -> Real

A one-application stand-in for `‖𝒜‖²`, used to put the default warm start `x̂ = 𝒜'y` on the
image's scale when **nothing else in the solve needs the operator norm** — a pure Krylov solve,
where [`_should_estimate_operator_norm`](@ref) is `false` but
[`_warm_start_needs_operator_norm`](@ref) is `true`.

It is the Rayleigh quotient of the normal operator at the warm start,
`⟨x̂, 𝒜'𝒜 x̂⟩ / ⟨x̂, x̂⟩`, which estimates the same `ρ(𝒜'𝒜)` that `estimate_opnorm`'s power method
converges to — from below, as that does. One application of `𝒜` and one of `𝒜'` replace twenty;
they are applied in turn rather than through `𝒜' * 𝒜`, whose normal operator would be built
for this one product and then thrown away. It
works *because* the vector is `𝒜'y`: that already lies in the operator's dominant subspace, so a
single quotient is close. Measured on a 192²×8 acquisition:

| | power method | this proxy | proxy vs. power |
|---|---|---|---|
| Cartesian, R = 3 | 118 ms | 11 ms | 0.4 % under |
| radial, 80 spokes | 812 ms | 62 ms | 3.2 % under |

A few per cent is immaterial here: the correction exists to remove an order-of-magnitude scale
mismatch from the warm start, not to set a step size. Where `L` *is* the step size, the power
method still runs — an `Lf` hint that is too small costs convergence.
"""
function _warm_start_scale_proxy(𝒜, x̂::AbstractArray, config)
    local ρ
    # `@printing_step`, not `@step`, for the same reason as `_operator_norm_for_stepsize`.
    @printing_step "Estimating the warm-start scale" config begin
        R = real(eltype(x̂))
        denom = real(dot(x̂, x̂))
        num = denom > 0 ? real(dot(x̂, 𝒜' * (𝒜 * x̂))) : zero(denom)
        # A zero (or numerically degenerate) warm start needs no correction.
        ρ = num > 0 ? R(num / denom) : one(R)
    end
    return ρ
end

"""
	_scale_admm_penalty(algorithm, 𝒜, x₀, acq_data, L, method, config; eltype_real) -> algorithm

Make a penalty `ρ` given to ADMM relative to the curvature `‖𝒜‖²` of the data term.

ADMM's `x`-update solves `(𝒜'𝒜 + ρ B'B) x = …`, so `ρ` only means something next to `‖𝒜‖²`.
That is about 1 for a Cartesian encoding and about 2·10⁶ for a radial NFFT one, and a penalty
that is fine for the first is then seven orders of magnitude too small for the second: `ρ/‖𝒜‖²`
falls below `Float32` rounding and the proximal steps never reach `x`, so the result stops
depending on `λ` at all. Multiplying the penalty by `‖𝒜‖²` makes a given `rho` mean the same
thing for every encoding.

Solving the normalized problem `𝒜/‖𝒜‖`, `y/‖𝒜‖` instead is the same thing in other coordinates:
with `BartScaling` recomputed on the normalized data it has the same effective `λ`, and its
fixed-`ρ` iterates are these. Measured on radial cine (low rank, locally low rank; 17, 34 and 68
spokes) and on 2D radial and Cartesian TV, 20 iterations, the two agree to four digits of NRMSE
at every `λ`, so the cheaper of the two is the one kept.

Both a fixed `rho` and the initial `rho` of a `penalty_sequence` are scaled. ADMM's default
adaptive sequence is left to start from 1: it reaches the scale of the problem on its own, and
starting it from `‖𝒜‖²` measured no better on the same cases.

`‖𝒜‖²` is `L²` when the operator norm was estimated, else the Rayleigh quotient of `𝒜'𝒜` at the
warm start ([`_warm_start_scale_proxy`](@ref)), or at `𝒜'y` when the warm start is zero. An
explicit `disable_operator_normalization = true` leaves the penalty as given.
"""
_scale_admm_penalty(algorithm, 𝒜, x₀, acq_data, L, method, config; eltype_real) = algorithm

function _scale_admm_penalty(
        algorithm::ProximalAlgorithms.IterativeAlgorithm{ProximalAlgorithms.ADMMIteration},
        𝒜, x₀, acq_data, L, method::IterativeReconstruction, config; eltype_real,
    )
    method.disable_operator_normalization === true && return algorithm
    kwargs = algorithm.kwargs
    given = haskey(kwargs, :rho) ||
        (haskey(kwargs, :penalty_sequence) && !isnothing(kwargs[:penalty_sequence].rho))
    given || return algorithm
    s = eltype_real(isnothing(L) ? _admm_curvature(𝒜, x₀, acq_data, config) : L^2)
    if haskey(kwargs, :rho)
        return ProximalAlgorithms.override_parameters(algorithm; rho = kwargs[:rho] .* s)
    end
    ps = kwargs[:penalty_sequence]
    scaled = ProximalAlgorithms.reinstantiate_penalty_sequence(ps, eltype_real, ps.rho .* s)
    return ProximalAlgorithms.override_parameters(algorithm; penalty_sequence = scaled)
end

function _admm_curvature(𝒜, x₀::AbstractArray, acq_data, config)
    v = iszero(x₀) ? 𝒜' * _measurement(acq_data.kspace_data) : x₀
    return _warm_start_scale_proxy(𝒜, v, config)
end
_admm_curvature(𝒜, x₀s::Tuple, acq_data, config) =
    _warm_start_scale_proxy(𝒜, 𝒜' * _measurement(acq_data.kspace_data), config)

# `‖𝒜‖`, for use as `Lf = n‖𝒜‖²` and/or to scale-correct the default warm start.
#
# `estimate_opnorm` returns a value at or above `‖𝒜‖`: it pairs the power iteration, which
# converges from below, with the closed-form `opnorm_bound`, which is above, and returns the bound
# whenever it is finite. That is the direction a step size needs, since `gamma = 1/Lf` is fixed and
# no backtracking runs to catch a value that came out too low. How much overshoot to accept comes
# from the algorithm, via `opnorm_rel_margin`. `exact_opnorm = true` still swaps in the converged
# `opnorm` for callers who want the number itself.
function _operator_norm_for_stepsize(𝒜, method::IterativeReconstruction, config)
    local L
    # `@printing_step`, not `@step`: the latter runs its body in a `@spawn`, so `L` would be
    # bound only inside that task's closure.
    @printing_step "Estimating the operator norm" config begin
        L = method.exact_opnorm ? LinearAlgebra.opnorm(𝒜) :
            AbstractOperators.estimate_opnorm(
                𝒜; rel_margin = opnorm_rel_margin(method.algorithm)
            )
    end
    @argcheck L != 0 "Cannot reconstruct with an encoding operator of zero norm"
    return L
end
