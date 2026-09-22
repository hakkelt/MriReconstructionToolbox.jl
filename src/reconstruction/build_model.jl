"""
    build_model(𝒜::AbstractOperator, y::AbstractArray, reg::Regularization; threaded::Bool)
    build_model(𝒜::AbstractOperator, y::AbstractArray, regs::Tuple; threaded::Bool)

Builds a StructuredOptimization.jl model from the encoding operator, the measured data and one or more regularizations.

# Arguments
- `𝒜::AbstractOperator`: The encoding operator.
- `y::AbstractArray`: The measured data.
- `reg::Regularization`: The regularization term.
- `threaded::Bool`: Whether to use threading.
- `x₀::Union{Nothing,AbstractArray}`: An optional initial guess for the variable (default is 𝒜' * y).

# Returns
- `terms::Tuple`: The terms defining the optimization problem.

# Example
```julia
julia> using MriReconstructionToolbox, StructuredOptimization
julia> x = rand(8, 8)
julia> 𝒜 = Eye(x)
julia> y = 𝒜 * x .+ 0.01 .* rand
julia> reg = L1Image(0.2)
julia> terms = build_model(𝒜, y, reg; threaded=false)
```
"""
function build_model(𝒜::AbstractOperator, y::AbstractArray, reg::Regularization; threaded::Bool = true, x₀::Union{Nothing, AbstractArray} = nothing)
    return build_model(𝒜, y, (reg,); threaded, x₀)
end

"""
	patch_algorithm_with_default_values(algorithm, Lf::Union{Nothing,Real}=nothing)

Fill in algorithm defaults that depend on the model just built. `Lf`, when not `nothing`, is the
Lipschitz-constant hint appropriate for the (possibly multi-variable) data term against the actual
operator used to build the model -- `nothing` when the operator was left at its natural, non-unit norm
(`disable_operator_normalization = true`), in which case the algorithm's own estimate against that
operator is correct and no override is applied.
"""
function patch_algorithm_with_default_values(
        algorithm::ProximalAlgorithms.IterativeAlgorithm{T}, Lf::Union{Nothing, Real} = nothing;
        eltype_real::Type{<:Real} = Float64,
    ) where {
        T <: Union{
            ProximalAlgorithms.ForwardBackwardIteration,
            ProximalAlgorithms.FastForwardBackwardIteration,
            ProximalAlgorithms.POGMIteration,
        },
    }
    if Lf !== nothing && :Lf ∉ keys(algorithm.kwargs)
        return ProximalAlgorithms.override_parameters(algorithm; Lf)
    else
        return algorithm
    end
end

function patch_algorithm_with_default_values(
        algorithm::ProximalAlgorithms.IterativeAlgorithm{ProximalAlgorithms.ADMMIteration}, Lf::Union{Nothing, Real} = nothing;
        eltype_real::Type{<:Real} = Float64,
    )
    # `cg_maxit` is capped well below ADMM's own default of 100 because the inner CG is warm-started
    # from the previous outer iterate, so a short solve per outer step is enough.
    #
    # Neither `rho` nor `cg_tol` is defaulted here, and both omissions are deliberate.
    #
    # `cg_tol` is derived by `ADMM` as `min(1e-2, tol * 100)`, keeping the inner solve tighter than
    # the outer stopping tolerance. Pinning any constant would break that coupling and cap the
    # reachable accuracy whenever a caller tightens `tol`.
    #
    # `rho` is left to ADMM's adaptive `SpectralRadiusApproximationPenalty`, which converges far
    # faster than any fixed penalty on the problems this package builds: on a 2x-undersampled
    # L1Wavelet + TV reconstruction the adaptive penalty reaches 0.035 relative error within 100
    # iterations, while a fixed `rho = 1` needs ~2000 iterations to match it. Terms whose ADMM
    # behaviour is sensitive to the penalty should document a tuned `rho` of their own rather than
    # have one imposed on every caller here.
    :cg_maxit ∈ keys(algorithm.kwargs) && return algorithm
    return ProximalAlgorithms.override_parameters(algorithm; cg_maxit = 10)
end

function patch_algorithm_with_default_values(
        algorithm::ProximalAlgorithms.IterativeAlgorithm{ProximalAlgorithms.DouglasRachfordIteration}, Lf::Union{Nothing, Real} = nothing;
        eltype_real::Type{<:Real} = Float64,
    )
    if :gamma ∉ keys(algorithm.kwargs)
        # `Lf = n‖𝒜‖²` for the data term, so the default step is `1/Lf`.
        # When two indicator / constraint terms are present, the problem is scale-free and gamma sets the rate.
        gamma = Lf !== nothing ? eltype_real(1 / max(Lf, eps(eltype_real))) : eltype_real(1)
        return ProximalAlgorithms.override_parameters(algorithm; gamma = gamma)
    else
        g = algorithm.kwargs[:gamma]
        if g isa Real && typeof(g) != eltype_real
            return ProximalAlgorithms.override_parameters(algorithm; gamma = eltype_real(g))
        end
    end
    return algorithm
end

function patch_algorithm_with_default_values(algorithm::ProximalAlgorithms.IterativeAlgorithm, Lf::Union{Nothing, Real} = nothing; eltype_real::Type{<:Real} = Float64)
    return algorithm
end

function patch_algorithm_with_default_values(algorithm::Tuple, Lf::Union{Nothing, Real} = nothing; eltype_real::Type{<:Real} = Float64)
    return map(a -> patch_algorithm_with_default_values(a, Lf; eltype_real), algorithm)
end

"""
	consumes_lf(algorithm) -> Bool

Whether this algorithm actually reads the `Lf` hint that
[`patch_algorithm_with_default_values`](@ref) is handed — i.e. whether estimating `‖𝒜‖` buys the
solve anything.

It is deliberately a mirror of the dispatch above, and lives beside it so the two cannot drift.
Three of those methods use `Lf`: the forward-backward family takes it as a step size, and
Douglas-Rachford derives `gamma = 1/Lf` from it. The `ADMMIteration` method does not, and neither
does the generic fallback — ADMM with a fixed or adaptive penalty and an inner CG `x`-update has
no Lipschitz step at all (`ProximalAlgorithms`' `admm.jl` has no `Lf` field), so the power
iteration that produced the number was pure cost.

Measured, AMD EPYC 7352, 1 thread, `OPENBLAS_NUM_THREADS=1`, 2026-09-21: `estimate_opnorm` is
48.4 ms of a 444 ms ADMM low-rank solve (10.9%) and 50.2 ms of a 165 ms solve on the
128²×8-coil sparsity case. A FISTA row still pays it, and should: there the number is the step
size, and an `Lf` that is too small costs convergence.
"""
consumes_lf(
    ::ProximalAlgorithms.IterativeAlgorithm{
        <:Union{
            ProximalAlgorithms.ForwardBackwardIteration,
            ProximalAlgorithms.FastForwardBackwardIteration,
            ProximalAlgorithms.POGMIteration,
            ProximalAlgorithms.DouglasRachfordIteration,
        },
    }
) = true
consumes_lf(algorithms::Tuple) = any(consumes_lf, algorithms)
consumes_lf(::Any) = false

"""
	opnorm_rel_margin(algorithm) -> Real

How far above `‖𝒜‖` the estimate handed to this algorithm as `Lf` may sit, as a relative margin
for `AbstractOperators.estimate_opnorm`.

Every algorithm that reads `Lf` here takes the step size to be exactly `1 / Lf`
(`forward_backward.jl:37`, `fast_forward_backward.jl:44`, `pogm.jl:50`) and none of them
backtracks while `Lf` is supplied, so nothing corrects a value that came out too low. The estimate
is therefore always asked for from above — that is `estimate_opnorm`'s `side = :upper` default —
and this margin only says how much overshoot to accept before spending more iterations closing the
gap. Overshoot costs convergence rate; a value below `‖𝒜‖` costs convergence itself.

POGM gets the tightest margin because it is the least forgiving: an estimate 1.2% low has been
observed to diverge it, where the forward-backward family only slows down. Where the closed-form
[`AbstractOperators.opnorm_bound`](@ref) is finite and tight — every Cartesian SENSE operator —
the margin is met on iteration zero and costs nothing either way.
"""
opnorm_rel_margin(
    ::ProximalAlgorithms.IterativeAlgorithm{<:ProximalAlgorithms.POGMIteration}
) = 1.0e-3
opnorm_rel_margin(algorithms::Tuple) = minimum(opnorm_rel_margin, algorithms)
opnorm_rel_margin(::Any) = 0.01

function build_model(
        𝒜::AbstractOperator, y::AbstractArray, regs::Tuple;
        threaded::Bool = true, x₀::Union{Nothing, AbstractArray} = nothing,
        fidelity::DataFidelity = L2Loss(),
    )
    terms, _, _ = build_model_with_variables(𝒜, y, regs; threaded, x₀, fidelity)
    return terms
end

"""
	build_model_with_variables(𝒜, y, regs; threaded, x₀, fidelity)

Same as [`build_model`](@ref), but also returns the variables the model was built from:
`(terms, x, auxiliary_variables)`, where `x` is the image variable and `auxiliary_variables` is a tuple of
the extra variables the regularizations introduced (see [`materialize_with_auxiliaries`](@ref)).

Callers that need the solution must read it from `x` rather than from the solver's returned variable tuple:
once a regularization contributes auxiliary variables, the position of the image variable in that tuple is
an implementation detail of `extract_variables`, not something to rely on.
"""
function build_model_with_variables(
        𝒜::AbstractOperator, y::AbstractArray, regs::Tuple;
        threaded::Bool = true, x₀::Union{Nothing, AbstractArray} = nothing,
        fidelity::DataFidelity = L2Loss(),
    )
    x₀ = isnothing(x₀) ? 𝒜' * y : copy(x₀)
    regs = bind_dimensions(regs, dims_of(x₀))
    x = Variable(unname(x₀))
    𝒜 = unname(𝒜)
    y = unname(y)
    reg_term_list, auxiliaries = materialize_all(regs, x; threaded)

    terms = if fidelity isa L2Loss
        # A plain least-squares term: it keeps `𝒜` in the expression, and StructuredOptimization
        # decides at parse time whether to differentiate through the fused normal operator `𝒜'𝒜`
        # instead. That decision belongs there and nowhere else, because only the parser sees the
        # operator expanded over the problem's full variable tuple — image alone, or
        # (image, auxiliary...) once a regularization such as total generalized variation adds one.
        @term ls(𝒜 * x - y)
    elseif fidelity isa HardConsistency
        prox = hard_consistency_prox(𝒜, y, fidelity.maxit, fidelity.tol)
        StructuredOptimization.Term(1, prox, identity_operator(unname(~x)) * x, "HardConsistency(𝒜x=y)")
    elseif fidelity isa NoFidelity
        if isempty(reg_term_list)
            throw(ArgumentError("No terms in optimization problem: NoFidelity specified with no regularizations."))
        end
        nothing
    else
        throw(ArgumentError("Unsupported data fidelity: $fidelity"))
    end

    if terms === nothing
        terms = first(reg_term_list)
        for reg_terms in reg_term_list[2:end]
            terms += reg_terms
        end
    else
        for reg_terms in reg_term_list
            terms += reg_terms
        end
    end
    return terms, x, auxiliaries
end

"""
    build_model(𝒜::AbstractOperator, y::AbstractArray, components::Tuple{Vararg{Component}}; threaded, x₀s, fidelity)

Builds a multi-variable StructuredOptimization.jl model for image decomposition: one
`Variable` per component, with data fidelity term (default `‖𝒜*(x₁ + x₂ + …) - y‖²`).

The data term applies `𝒜` to the *sum* of the component variables
(`𝒜 * (x₁ + x₂ + …)`, i.e. `Compose(𝒜, HCAT(Eye, …))`) rather than summing
`𝒜*x₁ + 𝒜*x₂ + …`, so `𝒜` is applied once per iteration instead of once per
component (requires `Compose`'s `getindex`/`permute` to distribute over a
multi-domain inner factor, upstream AbstractOperators fix).

The data term is the same plain least-squares term the single-variable path builds; whether its
gradient goes through the fused normal operator is decided when the problem is parsed, against
the joint domain of every component variable.

# Returns
- `(terms, vars, auxiliaries)`: `terms::StructuredOptimization.TermSet`, `vars::NTuple{n,Variable}` in
  component order, and the tuple of auxiliary variables the regularizations introduced (see
  [`materialize_with_auxiliaries`](@ref)) — usually empty.
"""
function build_model(
        𝒜::AbstractOperator, y::AbstractArray, components::Tuple{Component, Vararg{Component}};
        threaded::Bool = true, x₀s,
        fidelity::DataFidelity = L2Loss(),
    )
    check_components(components)
    components = bind_dimensions(components, dims_of(first(x₀s)))
    𝒜 = unname(𝒜)
    y = unname(y)
    # `Variable` stores the array by reference and `solve` writes the solution back through it, so
    # the caller's `x₀` must be copied here -- exactly as the single-variable path above does. In
    # the task-splitting path `x₀s` are `@view`s into the caller's full array, which would otherwise
    # be written through as well.
    vars = Tuple(Variable(copy(unname(x₀))) for x₀ in x₀s)
    ex = 𝒜 * reduce(+, vars)

    terms = if fidelity isa L2Loss
        StructuredOptimization.ls(ex - y)
    elseif fidelity isa HardConsistency
        prox = hard_consistency_prox(𝒜, y, fidelity.maxit, fidelity.tol)
        StructuredOptimization.Term(1, prox, reduce(+, vars), "HardConsistency(𝒜(x₁+…)=y)")
    elseif fidelity isa NoFidelity
        nothing
    else
        throw(ArgumentError("Unsupported data fidelity: $fidelity"))
    end

    auxiliaries = ()
    first_term = terms === nothing
    for (component, x) in zip(components, vars)
        component_terms, component_auxiliaries = materialize_with_auxiliaries(component, x; threaded)
        if first_term
            terms = component_terms
            first_term = false
        else
            terms += component_terms
        end
        auxiliaries = (auxiliaries..., component_auxiliaries...)
    end
    if terms === nothing
        throw(ArgumentError("No terms in optimization problem: NoFidelity specified with no regularizations."))
    end
    return terms, vars, auxiliaries
end
