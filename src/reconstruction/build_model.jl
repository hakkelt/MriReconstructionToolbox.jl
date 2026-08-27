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
    x₀ = isnothing(x₀) ? 𝒜' * y : copy(x₀)
    x = Variable(unname(x₀))
    𝒜 = unname(𝒜)
    y = unname(y)
    if disable_normalop_optimization
        terms = @term ls(𝒜 * x - y)
    else
        terms = @term normalop_ls(𝒜 * x - y)
    end
    @assert terms isa StructuredOptimization.Term
    for reg in regs
        @argcheck reg isa Regularization "All regularization terms must be of type Regularization."
        terms += materialize(reg, x; threaded)
    end
    return terms
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
- `(terms, vars)`: `terms::StructuredOptimization.TermSet`, `vars::NTuple{n,Variable}` in component order.
"""
function build_model(
        𝒜::AbstractOperator, y::AbstractArray, components::Tuple{Vararg{Component}};
        threaded::Bool = true, x₀s,
    )
    check_components(components)
    𝒜 = unname(𝒜)
    y = unname(y)
    vars = Tuple(Variable(unname(x₀)) for x₀ in x₀s)
    ex = 𝒜 * reduce(+, vars)
    terms = StructuredOptimization.ls(ex - y)
    for (component, x) in zip(components, vars)
        terms += materialize(component, x; threaded)
    end
    return terms, vars
end
