export problem, @minimize, @term

"""
	problem(terms...)

Constructs a problem.

# Example

```jldoctest
julia> x = Variable(4)
Variable(Float64, (4,), "x")

julia> A, b = randn(10, 4), randn(10);

julia> p = problem(ls(A * x - b), norm(x) <= 1);

julia> length(p)
2
```

Arguments are flattened, so a `TermSet` built with `+` and a list of separate terms give the
same problem:

```jldoctest
julia> x = Variable(4); A, b = randn(10, 4), randn(10);

julia> length(problem(ls(A * x - b) + 1.0e-2 * norm(x, 1)))
2
```

"""
problem(terms...) = begin
    flattened_terms = Term[]
    for t in terms
        if t isa TermSet
            append!(flattened_terms, t.terms)
        elseif t isa Term
            push!(flattened_terms, t)
        else
            error("All arguments must be of type Term or TermSet")
        end
    end
    TermSet(flattened_terms...)
end

function expand_terms_with_repr(expr)
    if expr isa Expr && expr.head == :call && expr.args[1] == :+
        return Tuple(map(t -> :(Term($(esc(t)), $(string(t)))), expr.args[2:end]))
    elseif expr isa Symbol
        return (esc(expr),)
    elseif expr isa Expr && expr.head == :tuple
        return Tuple(first.(expand_terms_with_repr.(expr.args)))
    else
        return (:(Term($(esc(expr)), $(string(expr)))),)
    end
end

"""
    @term expr

Records the code representation of the term. Useful if later we want to print the term, e.g. when debugging.
"""
macro term(expr)
    terms = expand_terms_with_repr(expr)
    return Expr(:block, terms...)
end

"""
    @minimize cost [st ctr] [with slv_opt]

Minimize a given problem with cost function `cost`, constraints `ctr` and solver options `slv_opt`.

# Example

```jldoctest
julia> A, b, x = randn(10, 4), randn(10), Variable(4);

julia> @minimize ls(A * x - b) + 0.5 * norm(x);

julia> length(~x)  # `~x` is the array holding the solution
4

julia> @minimize ls(A * x - b) st x >= 0.0;

julia> all(~x .>= -1.0e-9)  # the constraint is satisfied at the returned point
true

julia> _, it = @minimize ls(A * x - b) st norm(x) <= 2.0 with ProximalAlgorithms.PANOCplus();

julia> norm(~x) <= 2.0 + 1.0e-6
true
```

Note that the variables are *not* reset between these calls: `~x` still holds the previous
solution when the next `@minimize` starts, which warm-starts it. Write `~x .= 0.0` first to
opt out.

Returns as output a tuple containing the optimization variables and the number
of iterations spent by the solver algorithm.
"""
macro minimize(cf::Union{Expr, Symbol})
    cost = expand_terms_with_repr(cf)
    problem_expr = Expr(:call, :problem, cost...)
    return :(solve($problem_expr))
end

macro minimize(cf::Union{Expr, Symbol}, s::Symbol, cstr::Union{Expr, Symbol})
    cost = expand_terms_with_repr(cf)
    if s == :st
        constraints = expand_terms_with_repr(cstr)
        terms = (cost..., constraints...)
        problem_expr = Expr(:call, :problem, terms...)
        return :(solve($problem_expr))
    elseif s == :with
        solver = esc(cstr)
        problem_expr = Expr(:call, :problem, cost...)
        return :(solve($problem_expr, $solver))
    else
        error("wrong symbol after cost function! use `st` or `with`")
    end
end

macro minimize(cf::Union{Expr, Symbol}, s::Symbol, cstr::Union{Expr, Symbol}, w::Symbol, slv::Union{Expr, Symbol})
    cost = expand_terms_with_repr(cf)
    s != :st && error("wrong symbol after cost function! use `st`")
    constraints = expand_terms_with_repr(cstr)
    w != :with && error("wrong symbol after constraints! use `with`")
    solver = esc(slv)
    terms = (cost..., constraints...)
    problem_expr = Expr(:call, :problem, terms...)
    return :(solve($problem_expr, $solver))
end
