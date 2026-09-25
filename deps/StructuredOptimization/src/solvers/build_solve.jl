export suggest_algorithm, select_solver

# A dry run parses a problem only to learn whether (and at what cost) it parses, so it skips
# what `prepare` would build that the answer does not depend on: normal operators and
# factorizations (see `merge_function_with_operator`). It is task-local, so concurrent
# parses on other tasks are unaffected.
_is_dry_run() = get(task_local_storage(), :StructuredOptimization_dry_run, false)::Bool
_dry_run(f) = task_local_storage(f, :StructuredOptimization_dry_run, true)

"""
	parse_problem(terms::TermSet, solver::IterativeAlgorithm)

Takes as input a TermSet containing the terms defining the problem and the solver.

Returns a TermSet containing the optimization variables and the problem terms
to be fed into the solver.

# Example

```jldoctest
julia> x = Variable(4);

julia> A, b = randn(10, 4), randn(10);

julia> p = problem(ls(A * x - b), norm(x) <= 1);

julia> _, kwargs, _ = StructuredOptimization.parse_problem(p, ProximalAlgorithms.PANOCplus());

julia> sort(collect(keys(kwargs)))
3-element Vector{Symbol}:
 :A
 :f
 :g
```

The keys are the slots the algorithm's assumptions declare: here a smooth `f` of a linear
`A`, plus a proximable `g`. A problem that cannot fill them returns `nothing`.
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

# What a term subset costs an assumption, in the units of `best_formulation`: the sum over
# its terms of the cheapest formulation the assumption can actually use.
#
# This is the scoring half of the two-layer search. It reads only the *unexpanded* term
# operator — a field access — and the trait predicates, so it neither builds an operator nor
# touches an array; the whole score of a problem costs a few dozen type queries against the
# thousands of operator applications of the optimization pass it selects.
function selection_cost(assumption, term_selection)
    needs = needs_prox(assumption) ? :prox : :any
    total = 0.0
    for term in term_selection
        _, cost = best_formulation(operator(term), term.f, displacement(term), term.lambda, needs)
        total += isfinite(cost) ? cost : 0.0
    end
    return total
end

"""
    match_assumption(assumption, remaining_terms, variables)

Consume a subset of `remaining_terms` with `assumption`, returning
`(preparation_result, matched_terms)` or `nothing` when no subset satisfies it.

Every subset that prepares is scored and the best one is taken, rather than the first one
that happens to work. The key is

    (-length(subset), selection_cost(assumption, subset), position in the powerset)

so the primary preference is still "absorb as many terms as possible into one assumption",
the formulation cost decides between subsets of equal size, and the historical
powerset position breaks a remaining tie — which makes the result deterministic and
reproduces the previous first-match choice wherever the costs tie.

Scoring the two layers together is the point: a cheaper formulation is only better if the
algorithm that gets selected can use it, which is why `selection_cost` asks `assumption`
what it needs rather than ranking formulations on their own.

Enumeration is pruned rather than exhaustive. `candidate_term_subsets` yields subsets
largest-first, so `-length(subset)` is non-decreasing: once a subset of size `k` has
prepared, no smaller subset can beat it and the search stops at the end of that size class.
"""
function match_assumption(assumption, remaining_terms, variables)
    best, best_key = nothing, nothing
    for (position, term_selection) in enumerate(candidate_term_subsets(remaining_terms))
        # Prune: sizes are non-increasing, so nothing from here on can beat the incumbent.
        best_key !== nothing && -length(term_selection) > best_key[1] && break
        preparation_result = prepare(TermSet(term_selection...), assumption, variables)
        preparation_result === nothing && continue
        key = (-length(term_selection), selection_cost(assumption, term_selection), position)
        if best_key === nothing || key < best_key
            best, best_key = (preparation_result, term_selection), key
        end
    end
    return best
end

# The parse of `terms` under `algorithm`, as `(kwargs, remaining_terms, cost)`. `cost` is
# the summed formulation cost of everything that was consumed, and is what ranks algorithms
# against each other in `parse_problem(terms)`.
function parse_terms(terms::TermSet, algorithm)
    assumptions = ProximalAlgorithms.get_assumptions(algorithm)
    variables = extract_variables(terms)
    remaining_terms = terms
    kwargs = Dict{Symbol, Any}()
    cost = 0.0
    for assumption in assumptions
        match = match_assumption(assumption, remaining_terms, variables)
        if match !== nothing
            preparation_result, matched_terms = match
            remaining_terms = setdiff(remaining_terms, matched_terms)
            cost += selection_cost(assumption, matched_terms)
            push!(kwargs, preparation_result...)
        end
        isempty(remaining_terms) && break
    end
    return kwargs, remaining_terms, cost
end

function parse_problem(terms::Union{Term, TermSet}, algorithm::T, return_partial::Bool = false) where {T <: IterativeAlgorithm}
    terms = terms isa TermSet ? terms : TermSet(terms)
    kwargs, remaining_terms, _ = parse_terms(terms, algorithm)
    if return_partial
        return (kwargs, remaining_terms)
    end
    isempty(remaining_terms) || return nothing
    return algorithm, kwargs, extract_variables(terms)
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

# The term's `repr` if it has one, its `show` form otherwise — what a user wrote, as
# opposed to the desugared operator graph.
_term_repr(term::Term) = term.repr !== nothing ? term.repr : string(term)
_term_repr(term) = string(term)

"""
    parse_failure_message(terms, what) -> String

Why `terms` could not be parsed for `what` (a solver type name, or a phrase describing a
set of solvers), naming each unparseable term and the property that blocked it.

`solve` prints the full `print_diagnostics` report before failing, but the report goes to
stdout and is lost to a caller that catches the error. PLAN.md 2.4 asks for a *rejecting*
ruleset, so the message itself has to carry the term's `repr` and the failed DCP-style
property — that is the difference between a caught error a program can act on and one it
can only re-raise.
"""
function parse_failure_message(terms::TermSet, what::AbstractString, algorithm = closest_algorithm(terms))
    lines = ["Sorry, I cannot parse this problem for $what."]
    if algorithm !== nothing
        _, remaining_terms = parse_problem(terms, algorithm, true)
        assumptions = ProximalAlgorithms.get_assumptions(algorithm)
        for term in remaining_terms
            reasons = unsatisfied_reasons(term, assumptions)
            entry = isempty(reasons) ?
                "  - $(_term_repr(term)): no assumption of $(typeof(algorithm).name.name) accepts its structure" :
                "  - $(_term_repr(term)): $(join(reasons, "; "))"
            entry in lines || push!(lines, entry)
        end
    end
    push!(lines, "Call print_diagnostics(problem) for the full report.")
    return join(lines, "\n")
end

# The algorithm that leaves the fewest terms unparsed, or `nothing` if there are none to
# choose from. This is the same "closest match" `print_diagnostics(terms)` reports.
function closest_algorithm(terms::TermSet, algorithms = ProximalAlgorithms.get_algorithms())
    best, fewest = nothing, nothing
    _dry_run() do
        for algorithm in algorithms
            _, remaining_terms = parse_problem(terms, algorithm, true)
            if fewest === nothing || length(remaining_terms) < fewest
                best, fewest = algorithm, length(remaining_terms)
            end
        end
    end
    return best
end

# Auto-selection: the algorithm whose *complete* parse is cheapest, by the same cost model
# the formulation layer uses, with the order `get_algorithms` advertises breaking ties. The
# two layers are scored jointly here: an algorithm that asks less of a term (a gradient
# rather than a prox, say) may let that term take a cheaper formulation, and that shows up
# in this total.
#
# The ranking runs as a dry run, and only the winner is then prepared for real: the cost of a
# parse never depends on what preparing builds, so every losing algorithm would otherwise
# pay for a normal operator nobody uses.
function parse_problem(terms::Union{Term, TermSet})
    terms = terms isa TermSet ? terms : TermSet(terms)
    best, best_key = nothing, nothing
    _dry_run() do
        for (position, algorithm) in enumerate(ProximalAlgorithms.get_algorithms())
            _, remaining_terms, cost = parse_terms(terms, algorithm)
            isempty(remaining_terms) || continue
            key = (cost, position)
            if best_key === nothing || key < best_key
                best, best_key = algorithm, key
            end
        end
    end
    best === nothing && return nothing
    return parse_problem(terms, best)
end

"""
    select_solver(terms::Union{Term,TermSet}, solvers)

The solver [`solve`](@ref)`(terms, solvers)` runs: the first of `solvers` that the problem
parses into, or `nothing` if none does. Nothing is built to decide this -- no normal
operator, no factorization -- so it is cheap enough to ask before committing to a solver,
e.g. to know whether a step size is worth estimating for it.
"""
function select_solver(terms::Union{Term, TermSet}, solvers)
    terms = terms isa TermSet ? terms : TermSet(terms)
    return _dry_run() do
        for solver in solvers
            _, remaining_terms, _ = parse_terms(terms, solver)
            isempty(remaining_terms) && return solver
        end
        return nothing
    end
end

"""
    suggest_algorithm(terms::Union{Term,TermSet}[, algorithms])

Return the list of algorithms (from `algorithms`, defaulting to every algorithm
`ProximalAlgorithms` advertises) whose assumptions the problem `terms` can be parsed
into. An empty result means no available algorithm matches the problem structure; use
[`print_diagnostics`](@ref) to see why.

# Example

```jldoctest
julia> x = Variable(4); A, b = randn(10, 4), randn(10);

julia> isempty(suggest_algorithm(problem(ls(A * x - b) + 1.0e-2 * norm(x, 1))))
false

julia> p = problem(norm(A * x, 1));  # the term is not proximable: prox does not compose

julia> isempty(suggest_algorithm(p))  # but algorithms with an operator slot still take it
false

julia> ProximalAlgorithms.FastForwardBackward() in suggest_algorithm(p)
false
```
"""
function suggest_algorithm(terms::Union{Term, TermSet}, algorithms = ProximalAlgorithms.get_algorithms())
    terms = terms isa TermSet ? terms : TermSet(terms)
    suitable_algs = []
    _dry_run() do
        for algorithm in algorithms
            _, remaining_terms, _ = parse_terms(terms, algorithm)
            isempty(remaining_terms) && push!(suitable_algs, algorithm)
        end
    end
    return suitable_algs
end

function print_diagnostics(terms::Union{Term, TermSet})
    terms = terms isa TermSet ? terms : TermSet(terms)
    best_algorithm = closest_algorithm(terms)
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

```jldoctest
julia> x = Variable(4);

julia> A, b = randn(10, 4), randn(10);

julia> ~x .= 0.0;

julia> p = problem(ls(A * x - b), norm(x) <= 1);

julia> vars, it = solve(p, ProximalAlgorithms.PANOCplus(); maxit = 200);

julia> norm(~x) <= 1 + 1.0e-6  # the constraint holds at the returned point
true

julia> it > 0
true
```

The minimizer is written back into the variables, so `~x` is the answer; the returned tuple
is `(variables, iterations)`.
"""
function solve(terms::Union{Term, TermSet}, solvers::Union{<:AbstractVector{<:IterativeAlgorithm}, <:Tuple{Vararg{IterativeAlgorithm}}}; kwargs...)
    terms = terms isa TermSet ? terms : TermSet(terms)
    solver = select_solver(terms, solvers)
    if solver !== nothing
        _, term_kwargs, x = parse_problem(terms, solver)
        return _run_solver(solver, term_kwargs, x; kwargs...)
    end
    return if length(solvers) == 1
        print_diagnostics(terms, solvers[1])
        error(parse_failure_message(terms, "solver of type $(typeof(solvers[1]).parameters[1])", solvers[1]))
    else
        print_diagnostics(terms)
        error(parse_failure_message(terms, "any of the provided solvers", closest_algorithm(terms, solvers)))
    end
end

function solve(terms::Union{Term, TermSet}, solver::IterativeAlgorithm; kwargs...)
    terms = terms isa TermSet ? terms : TermSet(terms)
    result = parse_problem(terms, solver)
    if result === nothing
        print_diagnostics(terms, solver)
        error(parse_failure_message(terms, "solver of type $(typeof(solver).parameters[1])", solver))
    end
    _, term_kwargs, x = result
    return _run_solver(solver, term_kwargs, x; kwargs...)
end

function solve(terms::Union{Term, TermSet}; kwargs...)
    terms = terms isa TermSet ? terms : TermSet(terms)
    result = parse_problem(terms)
    if result === nothing
        print_diagnostics(terms)
        error(parse_failure_message(terms, "any available solver"))
    end
    solver, term_kwargs, x = result
    return _run_solver(solver, term_kwargs, x; kwargs...)
end
