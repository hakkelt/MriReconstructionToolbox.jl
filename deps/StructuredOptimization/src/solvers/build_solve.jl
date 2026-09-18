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
# Candidate term-subsets for one assumption, in the order they are tried.
#
# The selection preference is: absorb as many terms as possible into a single
# assumption (largest subsets first). This is a deterministic score — subsets are
# ranked by `(size, powerset-position)` — so a fixed problem always parses the same
# way regardless of external iteration order. Enumerating `powerset` largest-first
# reproduces the historical `reverse(collect(powerset(...)))` order exactly, keeping
# `parse_problem`/`suggest_algorithm`/`print_diagnostics` behavior stable.
candidate_term_subsets(remaining_terms) = reverse(collect(powerset(remaining_terms, 1)))

# Try to consume some subset of `remaining_terms` with `assumption`, most-preferred
# subset first. Returns `(preparation_result, matched_terms)` on the first success,
# or `nothing` if no subset satisfies the assumption.
function match_assumption(assumption, remaining_terms, variables)
    for term_selection in candidate_term_subsets(remaining_terms)
        preparation_result = prepare(TermSet(term_selection...), assumption, variables)
        if preparation_result !== nothing
            return preparation_result, term_selection
        end
    end
    return nothing
end

function parse_problem(terms::Union{Term, TermSet}, algorithm::T, return_partial::Bool = false) where {T <: IterativeAlgorithm}
    terms = terms isa TermSet ? terms : TermSet(terms)
    assumptions = ProximalAlgorithms.get_assumptions(algorithm)
    variables = extract_variables(terms)
    remaining_terms = terms
    kwargs = Dict{Symbol, Any}()
    for assumption in assumptions
        match = match_assumption(assumption, remaining_terms, variables)
        if match !== nothing
            preparation_result, matched_terms = match
            remaining_terms = setdiff(remaining_terms, matched_terms)
            push!(kwargs, preparation_result...)
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

"""
    print_diagnostics(terms::Union{Term,TermSet}[, algorithm])

Explain how a problem matches (or fails to match) a solver's assumptions. With an
`algorithm`, print the assumed problem form, the terms that were successfully
prepared, and — for each term that could not be prepared — the unsatisfied property
(e.g. `is_convex`, `is_proximable`) that blocked it. Without an `algorithm`, report
the closest-matching algorithm and diagnose against it.

This is the tool to reach for when [`solve`](@ref) or [`@minimize`](@ref) errors with
"cannot parse this problem": it names the DCP-style property the problem violates.
"""
function print_diagnostics(terms::Union{Term, TermSet}, algorithm::T) where {T <: IterativeAlgorithm}
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
    variables = extract_variables(terms)
    assumptions = ProximalAlgorithms.get_assumptions(algorithm)
    for term in remaining_terms
        reasons = unsatisfied_reasons(term, assumptions)
        if isempty(reasons)
            println(" - $term")
        else
            # Phase 2.4: surface *why* the term was rejected (the DCP-style failed
            # property), so a solver mismatch fails legibly instead of silently.
            println(" - $term (unsatisfied: $(join(reasons, "; ")))")
        end
    end
    return
end

# Function-side predicate list of an assumption, or `nothing` if it has none
# (e.g. LeastSquaresTerm / OperatorTermWithInfimalConvolution).
_assumption_func(assumption) = hasproperty(assumption, :func) ? assumption.func : nothing

# Compact, deduplicated list of "<role>: <unmet properties>" strings explaining why
# `term` fails each of `assumptions`' function-side predicate sets.
function unsatisfied_reasons(term, assumptions)
    reasons = String[]
    for assumption in assumptions
        item = _assumption_func(assumption)
        item === nothing && continue
        unmet = unsatisfied_properties(term, item)
        if !isempty(unmet)
            reason = "$(item.first) requires $(join((nameof(p) for p in unmet), ", "))"
            reason in reasons || push!(reasons, reason)
        end
    end
    return reasons
end

function parse_problem(terms::Union{Term, TermSet})
    terms = terms isa TermSet ? terms : TermSet(terms)
    for algorithm in ProximalAlgorithms.get_algorithms()
        result = parse_problem(terms, algorithm)
        if result !== nothing
            return result
        end
    end
    return nothing
end

"""
    suggest_algorithm(terms::Union{Term,TermSet}[, algorithms])

Return the list of algorithms (from `algorithms`, defaulting to every algorithm
`ProximalAlgorithms` advertises) whose assumptions the problem `terms` can be parsed
into. An empty result means no available algorithm matches the problem structure; use
[`print_diagnostics`](@ref) to see why.

# Example

```julia
julia> x = Variable(4); A, b = randn(10, 4), randn(10);

julia> suggest_algorithm(problem(ls(A*x - b) + 1e-2*norm(x, 1)))
```
"""
function suggest_algorithm(terms::Union{Term, TermSet}, algorithms = ProximalAlgorithms.get_algorithms())
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

function print_diagnostics(terms::Union{Term, TermSet})
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
    return print_diagnostics(terms, best_algorithm)
end

export solve

# Run a solver on an already-parsed problem, apply kwarg overrides, and write the
# minimizer back into the variable. `x_star` may be a Tuple for multi-variable
# problems; take its first block in that case (the shared write-back convention).
#
# Every function `prepare` placed in `term_kwargs` (`:f`, `:g`, ...) is called by the
# solver once per iteration with an `x0`-shaped input (`extract_operators` always
# builds its operator over the full `variables` tuple, so every term's domain is the
# same combined space `x0` lives in). `preallocate` is called once here, before the
# iteration starts, so any scratch space those calls need is allocated once instead of
# on every iteration; values with nothing to preallocate come back unchanged.
function _run_solver(solver, term_kwargs, x; kwargs...)
    solver = override_parameters(solver; kwargs...)
    x0 = ~x
    term_kwargs = Dict(key => preallocate(value, x0) for (key, value) in term_kwargs)
    x_star, it = solver(; x0 = x0, term_kwargs...)
    ~x .= x_star isa Tuple ? x_star[1] : x_star
    return x, it
end

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

julia> solve(p, PANOCplus(); maxit=10);

julia> ~x
```
"""
function solve(terms::Union{Term, TermSet}, solvers::Union{<:AbstractVector{<:IterativeAlgorithm}, <:Tuple{Vararg{IterativeAlgorithm}}}; kwargs...)
    terms = terms isa TermSet ? terms : TermSet(terms)
    for solver in solvers
        result = parse_problem(terms, solver)
        if result isa Nothing
            continue
        end
        _, term_kwargs, x = result
        return _run_solver(solver, term_kwargs, x; kwargs...)
    end
    return if length(solvers) == 1
        print_diagnostics(terms, solvers[1])
        error("Sorry, I cannot parse this problem for solver of type $(typeof(solvers[1]).parameters[1])")
    else
        print_diagnostics(terms)
        error("Sorry, I cannot parse this problem for any of the provided solvers")
    end
end

function solve(terms::Union{Term, TermSet}, solver::IterativeAlgorithm; kwargs...)
    terms = terms isa TermSet ? terms : TermSet(terms)
    result = parse_problem(terms, solver)
    if result === nothing
        print_diagnostics(terms, solver)
        error("Sorry, I cannot parse this problem for solver of type $(typeof(solver).parameters[1])")
    end
    _, term_kwargs, x = result
    return _run_solver(solver, term_kwargs, x; kwargs...)
end

function solve(terms::Union{Term, TermSet}; kwargs...)
    terms = terms isa TermSet ? terms : TermSet(terms)
    result = parse_problem(terms)
    if result === nothing
        print_diagnostics(terms)
        error("Sorry, I cannot find a suitable solver for this problem")
    end
    solver, term_kwargs, x = result
    return _run_solver(solver, term_kwargs, x; kwargs...)
end
