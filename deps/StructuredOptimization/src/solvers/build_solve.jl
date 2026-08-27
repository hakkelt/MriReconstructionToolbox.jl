export suggest_algorithm

"""
	parse_problem(terms::TermSet, solver::IterativeAlgorithm)

Takes as input a TermSet containing the terms defining the problem and the solver.

Returns a TermSet containing the optimization variables and the problem terms
to be fed into the solver.

# Example

```julia
julia> x = Variable(4)
Variable(Float64, (4,))

julia> A, b = randn(10,4), randn(10);

julia> p = problem( ls(A*x - b ) , norm(x) <= 1 );

julia> StructuredOptimization.parse_problem(p, PANOCplus());
```
"""
function parse_problem(terms::Union{Term,TermSet}, algorithm::T, return_partial::Bool = false) where {T <: IterativeAlgorithm}
    terms = terms isa TermSet ? terms : TermSet(terms)
    assumptions = ProximalAlgorithms.get_assumptions(algorithm)
    variables = extract_variables(terms)
    remaining_terms = terms
    kwargs = Dict{Symbol, Any}()
    for assumption in assumptions
        for term_selection in reverse(collect(powerset(remaining_terms, 1)))
            term_selection = TermSet(term_selection...)
            preparation_result = prepare(term_selection, assumption, variables)
            if preparation_result !== nothing
                term_selection = collect(term_selection)
                remaining_terms = setdiff(remaining_terms, term_selection)
                push!(kwargs, preparation_result...)
                break
            end
        end
        if isempty(remaining_terms)
            if return_partial
                return (kwargs, remaining_terms)
            end
            return algorithm, kwargs, variables
        end
    end
    return return_partial ? (kwargs, remaining_terms) : nothing
end

function print_diagnostics(terms::Union{Term,TermSet}, algorithm::T) where {T <: IterativeAlgorithm}
    terms = terms isa TermSet ? terms : TermSet(terms)
    kwargs, remaining_terms = parse_problem(terms, algorithm, true)
    print("The algorithm $(typeof(algorithm).name.name) assumes problem of form: ")
    show(ProximalAlgorithms.get_assumptions(algorithm))
    println()
    if !isempty(kwargs)
        println("Successfully prepared the following terms:")
        for (key, value) in kwargs
            println(" - $key: $(typeof(value))")
        end
    end
    println("The following terms could not be prepared:")
    for term in remaining_terms
        println(" - $term")
    end
end

function parse_problem(terms::Union{Term,TermSet})
    terms = terms isa TermSet ? terms : TermSet(terms)
    for algorithm in ProximalAlgorithms.get_algorithms()
        result = parse_problem(terms, algorithm)
        if result !== nothing
            return result
        end
    end
    return nothing
end

function suggest_algorithm(terms::Union{Term,TermSet}, algorithms = ProximalAlgorithms.get_algorithms())
    terms = terms isa TermSet ? terms : TermSet(terms)
    suitable_algs = []
    for algorithm in algorithms
        result = parse_problem(terms, algorithm)
        if result !== nothing
            push!(suitable_algs, algorithm)
        end
    end
    return suitable_algs
end

function print_diagnostics(terms::Union{Term,TermSet})
    terms = terms isa TermSet ? terms : TermSet(terms)
    best_algorithm, best_algorithm_remaining_terms = nothing, Inf
    for algorithm in ProximalAlgorithms.get_algorithms()
        _, remaining_terms = parse_problem(terms, algorithm, true)
        if length(remaining_terms) < best_algorithm_remaining_terms
            best_algorithm_remaining_terms = length(remaining_terms)
            best_algorithm = algorithm
        end
    end
	println("The closest algorithm to the problem is $best_algorithm")
    print_diagnostics(terms, best_algorithm)
end

export solve

"""
    solve(terms::Union{Term,TermSet}; kwargs...)
	solve(terms::Union{Term,TermSet}, solver::IterativeAlgorithm; kwargs...)
	solve(terms::Union{Term,TermSet}, solvers::Union{AbstractVector,Tuple}; kwargs...)

Takes as input a Term/TermSet containing the terms defining the problem and the solver options.

Solves the problem returning a tuple containing the iterations taken and the build solver.

# Example

```julia
julia> x = Variable(4)
Variable(Float64, (4,))

julia> A, b = randn(10,4), randn(10);

julia> p = problem(ls(A*x - b ), norm(x) <= 1);

julia> solve(p, PANOCplus(); maxiter=10);

julia> ~x
```
"""
function solve(terms::Union{Term,TermSet}, solvers::Union{<:AbstractVector{IterativeAlgorithm},<:Tuple{Vararg{IterativeAlgorithm}}}; kwargs...)
    terms = terms isa TermSet ? terms : TermSet(terms)
    for solver in solvers
        result = parse_problem(terms, solver)
        if result isa Nothing
            continue
        end
        _, term_kwargs, x = result
        solver = override_parameters(solver; kwargs...)
        x_star, it = solver(; x0 = ~x, term_kwargs...)
        ~x .= x_star isa Tuple ? x_star[1] : x_star
        return x, it
    end
    if length(solvers) == 1
        print_diagnostics(terms, solvers[1])
        error("Sorry, I cannot parse this problem for solver of type $(typeof(solvers[1]).parameters[1])")
    else
        print_diagnostics(terms)
        error("Sorry, I cannot parse this problem for any of the provided solvers")
    end
end

function solve(terms::Union{Term,TermSet}, solver::IterativeAlgorithm; kwargs...)
    terms = terms isa TermSet ? terms : TermSet(terms)
	result = parse_problem(terms, solver)
	if result === nothing
		print_diagnostics(terms, solver)
		error("Sorry, I cannot parse this problem for solver of type $(typeof(solver).parameters[1])")
	end
	_, term_kwargs, x = result
    solver = override_parameters(solver; kwargs...)
    x_star, it = solver(; x0 = ~x, term_kwargs...)
	~x .= x_star isa Tuple ? x_star[1] : x_star
	return x, it
end

function solve(terms::Union{Term,TermSet}; kwargs...)
    terms = terms isa TermSet ? terms : TermSet(terms)
	result = parse_problem(terms)
	if result === nothing
		print_diagnostics(terms)
		error("Sorry, I cannot find a suitable solver for this problem")
	end
	solver, term_kwargs, x = result
    solver = override_parameters(solver; kwargs...)
    x_star, it = solver(; x0 = ~x, term_kwargs...)
	~x .= x_star
	return x, it
end
