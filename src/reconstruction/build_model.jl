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
function build_model(𝒜::AbstractOperator, y::AbstractArray, reg::Regularization; threaded::Bool = true, x₀::Union{Nothing, AbstractArray} = nothing, disable_normalop_optimization::Bool = false)
    return build_model(𝒜, y, (reg,); threaded, x₀, disable_normalop_optimization)
end

function build_model(𝒜::AbstractOperator, y::AbstractArray, regs::Tuple; threaded::Bool = true, x₀::Union{Nothing, AbstractArray} = nothing, disable_normalop_optimization::Bool = false)
    terms, _, _ = build_model_with_variables(
        𝒜, y, regs; threaded, x₀, disable_normalop_optimization
    )
    return terms
end

"""
	build_model_with_variables(𝒜, y, regs; threaded, x₀, disable_normalop_optimization)

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
        disable_normalop_optimization::Bool = false,
    )
    x₀ = isnothing(x₀) ? 𝒜' * y : copy(x₀)
    x = Variable(unname(x₀))
    𝒜 = unname(𝒜)
    y = unname(y)
    # The regularizations are materialized first because whether any of them introduces an auxiliary
    # variable decides which form the data term may take.
    reg_term_list = ()
    auxiliaries = ()
    for reg in regs
        @argcheck reg isa Regularization "All regularization terms must be of type Regularization."
        reg_terms, reg_auxiliaries = materialize_with_auxiliaries(reg, x; threaded)
        reg_term_list = (reg_term_list..., reg_terms)
        auxiliaries = (auxiliaries..., reg_auxiliaries...)
    end
    # `normalop_ls` precomputes 𝒜'𝒜 and stores it inside the term, but that operator spans the image
    # variable alone. Once a regularization adds an auxiliary variable (total generalized variation),
    # the solver's `x0` spans (image, auxiliary...) while the stored operator still does not, and ADMM
    # rejects the pair with "A'b must have the same size as x0". The plain `ls` form is assembled
    # against the full variable tuple by `extract_operators`, so it lifts correctly.
    use_normalop = !disable_normalop_optimization && isempty(auxiliaries)
    terms = use_normalop ? (@term normalop_ls(𝒜 * x - y)) : (@term ls(𝒜 * x - y))
    @assert terms isa StructuredOptimization.Term
    for reg_terms in reg_term_list
        terms += reg_terms
    end
    return terms, x, auxiliaries
end

"""
    build_model(𝒜::AbstractOperator, y::AbstractArray, components::Tuple{Vararg{Component}}; threaded, x₀s)

Builds a multi-variable StructuredOptimization.jl model for image decomposition: one
`Variable` per component, with data term `‖𝒜*(x₁ + x₂ + …) - y‖²`.

The data term applies `𝒜` to the *sum* of the component variables
(`𝒜 * (x₁ + x₂ + …)`, i.e. `Compose(𝒜, HCAT(Eye, …))`) rather than summing
`𝒜*x₁ + 𝒜*x₂ + …`, so `𝒜` is applied once per iteration instead of once per
component (requires `Compose`'s `getindex`/`permute` to distribute over a
multi-domain inner factor, upstream AbstractOperators fix).

`normalop_ls` fusion does not apply here (see `disable_normalop_optimization` docs);
plain `ls` is always used, since the fast normal-operator path for a sum of shared
operators requires the upstream `HCAT` normal-op fusion.

# Returns
- `(terms, vars, auxiliaries)`: `terms::StructuredOptimization.TermSet`, `vars::NTuple{n,Variable}` in
  component order, and the tuple of auxiliary variables the regularizations introduced (see
  [`materialize_with_auxiliaries`](@ref)) — usually empty.
"""
function build_model(
        𝒜::AbstractOperator, y::AbstractArray, components::Tuple{Vararg{Component}};
        threaded::Bool = true, x₀s,
    )
    check_components(components)
    𝒜 = unname(𝒜)
    y = unname(y)
    # `Variable` stores the array by reference and `solve` writes the solution back through it, so
    # the caller's `x₀` must be copied here -- exactly as the single-variable path above does. In
    # the decomposition path `x₀s` are `@view`s into the caller's full array, which would otherwise
    # be written through as well.
    vars = Tuple(Variable(copy(unname(x₀))) for x₀ in x₀s)
    ex = 𝒜 * reduce(+, vars)
    terms = StructuredOptimization.ls(ex - y)
    auxiliaries = ()
    for (component, x) in zip(components, vars)
        component_terms, component_auxiliaries = materialize_with_auxiliaries(component, x; threaded)
        terms += component_terms
        auxiliaries = (auxiliaries..., component_auxiliaries...)
    end
    return terms, vars, auxiliaries
end
