"""
	IterativeReconstruction{R, A, F<:DataFidelity, M} <: AbstractIterativeMethod

Configures an iterative reconstruction problem with regularization terms, solver algorithms,
data fidelity, and signal modeling options.

# Fields
- `regularization::R`: Tuple of regularization terms (`Regularization` or `Component`).
- `algorithm::A`: Solver algorithm or tuple of candidate algorithms.
- `fidelity::F`: Data fidelity term (default `L2Loss()`).
- `signal_model::M`: Signal model mapping the optimization variable to the image (default `nothing`);
  e.g. `TemporalBasis` for subspace reconstruction or `KSpaceToImage` for a k-space-domain solve.
- `exact_opnorm::Bool`: Compute `‖𝒜‖` with a fully converged power iteration rather than the
  20-iteration estimate, which converges from below and so under-estimates slightly (default `false`).
- `disable_operator_normalization::Union{Nothing, Bool}`: Skip the `‖𝒜‖` estimate and let the
  algorithm derive its own step size (default `nothing` for auto-detection: skipped for pure
  unregularized CG/CGNR, which is scale invariant, run for proximal algorithms). The name predates
  the change that stopped `‖𝒜‖` being used to rescale `𝒜`; it is a step-size switch only.

`𝒜` is left at its natural norm and `Lf = n‖𝒜‖²` is passed to the algorithm instead, so the problem
solved is `½‖𝒜x - y‖² + R(x)`: `λ` weights the regularizer in the data's own units and the result
comes back in them. See "Operator norm, step size and λ" in `docs/src/high-level/methods.md` for
what changed and how to migrate a `λ` tuned against the previous behaviour.
- `disable_normalop_optimization::Bool`: Disable normal operator optimization (default `false`).
- `maxit::Union{Nothing, Int}`: Maximum solver iterations (default `100`). `nothing` defers to the
  `algorithm`'s own `maxit`, which is how an `algorithm = FISTA(maxit = 500)` is honoured.
- `tol::Union{Nothing, Float64}`: Stopping tolerance (default `1e-4`), **relative**: the absolute
  threshold handed to the solver is `max(10*eps, tol * maximum(abs, x₀))`, which differs from
  `ProximalAlgorithms`' absolute `tol`. `0` disables the tolerance test and `nothing` defers to the
  `algorithm`'s own stopping criterion.

`maxit` and `tol` are keyword-only on every constructor; regularization terms are the only
positional arguments.
"""
struct IterativeReconstruction{R <: Tuple, A, F <: DataFidelity, M} <: AbstractIterativeMethod
    regularization::R
    algorithm::A
    fidelity::F
    signal_model::M
    exact_opnorm::Bool
    disable_operator_normalization::Union{Nothing, Bool}
    disable_normalop_optimization::Bool
    maxit::Union{Nothing, Int}
    tol::Union{Nothing, Float64}

    function IterativeReconstruction(
            regularization::Tuple,
            algorithm,
            fidelity::F,
            signal_model::M,
            exact_opnorm::Bool,
            disable_operator_normalization::Union{Nothing, Bool},
            disable_normalop_optimization::Bool;
            maxit::Union{Nothing, Integer} = 100,
            tol::Union{Nothing, Real} = 1.0e-4,
        ) where {F <: DataFidelity, M}
        _validate_regularization(regularization)
        return new{typeof(regularization), typeof(algorithm), F, M}(
            regularization,
            algorithm,
            fidelity,
            signal_model,
            exact_opnorm,
            disable_operator_normalization,
            disable_normalop_optimization,
            isnothing(maxit) ? nothing : Int(maxit),
            isnothing(tol) ? nothing : Float64(tol),
        )
    end
end

function _validate_regularization(regs::Tuple)
    has_comp = any(r -> r isa Component, regs)
    has_bare = any(r -> r isa Regularization, regs)
    if has_comp && has_bare
        throw(ArgumentError("Cannot mix bare regularization terms with `Component`s; wrap loose regularization terms in a `Component`."))
    end
    return regs
end

# Keyword constructor
function IterativeReconstruction(;
        regularization = (),
        algorithm = DEFAULT_ALGORITHMS,
        fidelity::DataFidelity = L2Loss(),
        signal_model = nothing,
        exact_opnorm::Bool = false,
        disable_operator_normalization::Union{Nothing, Bool} = nothing,
        disable_normalop_optimization::Bool = false,
        maxit::Union{Nothing, Integer} = 100,
        tol::Union{Nothing, Real} = 1.0e-4,
    )
    regs_tuple = ensure_tuple(regularization)
    return IterativeReconstruction(
        regs_tuple,
        algorithm,
        fidelity,
        signal_model,
        exact_opnorm,
        disable_operator_normalization,
        disable_normalop_optimization;
        maxit,
        tol,
    )
end

# Positional vararg constructor (requires at least 1 term to prevent collision with keyword constructor)
function IterativeReconstruction(
        reg::Union{Regularization, Component},
        more_regs::Union{Regularization, Component}...;
        algorithm = DEFAULT_ALGORITHMS,
        fidelity::DataFidelity = L2Loss(),
        signal_model = nothing,
        exact_opnorm::Bool = false,
        disable_operator_normalization::Union{Nothing, Bool} = nothing,
        disable_normalop_optimization::Bool = false,
        maxit::Union{Nothing, Integer} = 100,
        tol::Union{Nothing, Real} = 1.0e-4,
    )
    regs = (reg, more_regs...)
    return IterativeReconstruction(;
        regularization = regs,
        algorithm,
        fidelity,
        signal_model,
        exact_opnorm,
        disable_operator_normalization,
        disable_normalop_optimization,
        maxit,
        tol,
    )
end

# Rebuild `method` with a different regularization tuple, every other field carried over. The
# decomposition path needs this per slice (λ is scale-compensated there); going through the
# keyword constructor keeps that call site from having to be edited every time a field is added.
function _with_regularization(method::IterativeReconstruction, regs::Tuple)
    return IterativeReconstruction(;
        regularization = regs,
        algorithm = method.algorithm,
        fidelity = method.fidelity,
        signal_model = method.signal_model,
        exact_opnorm = method.exact_opnorm,
        disable_operator_normalization = method.disable_operator_normalization,
        disable_normalop_optimization = method.disable_normalop_optimization,
        maxit = method.maxit,
        tol = method.tol,
    )
end

progress_total(method::IterativeReconstruction, acq_data) = method.maxit
