# returns all variables of a cost function, in terms of appearance
extract_variables(t::TermOrExpr) = variables(t)

function extract_variables(t::Union{Tuple, TermSet})
    var_tuples = variables.(t)
    vars = collect(Base.Iterators.flatten(var_tuples))
    return tuple(unique(vars)...)
end

# The term's function with its weight λ applied, and nothing else.
#
# This is the one extraction convention in the package: a term is `λ · f(A·x + d)`, the
# displacement `d` is carried by the affine operator (`extract_affines`/`affine`), and λ is
# applied exactly once, here. Anything that folds the operator or the displacement into the
# function is an *absorption* and belongs in `merge_function_with_operator`, which is the
# only place that knows what the selected algorithm will ask of the term.
weighted_function(t::Term) = t.lambda == 1 ? t.f : Postcompose(t.f, t.lambda)
weighted_function(t::TermSet) = SeparableSum(weighted_function.(t)...)

# Extract the linear operators (`accessor = operator`) or the affine operators
# keeping displacement (`accessor = affine`) from a term/expression, ordered to match
# `xAll`. The two families are identical apart from which accessor they use, so they
# share one implementation.

#single term, single variable (split by type so the single-variable case stays
# strictly more specific than the multi-variable `Term` method below — no ambiguity)
_extract(accessor, ::Tuple{Variable}, t::AbstractExpression) = accessor(t)
_extract(accessor, ::Tuple{Variable}, t::Term) = accessor(t)
_extract(accessor, xAll::NTuple{N, Variable}, t::AbstractExpression) where {N} =
    _sort_and_extract(accessor, xAll, expand(xAll, t))
_extract(accessor, xAll::NTuple{N, Variable}, t::Term) where {N} =
    _extract(accessor, xAll, TermSet(t))

#multiple terms, multiple variables
function _extract(accessor, xAll::NTuple{N, Variable}, t::TermSet) where {N}
    ops = ()
    for ti in t
        tex = expand(xAll, ti)
        ops = (ops..., _sort_and_extract(accessor, xAll, tex))
    end
    return vcat(ops...)
end

_sort_and_extract(accessor, ::Tuple{Variable}, t::TermOrExpr) = accessor(t)

function _sort_and_extract(accessor, xAll::NTuple{N, Variable}, t::TermOrExpr) where {N}
    p = zeros(Int, N)
    xL = variables(t)
    for i in eachindex(xAll)
        p[i] = findfirst(xi -> xi == xAll[i], xL)
    end
    return accessor(t)[p]
end

# returns all operators with an order dictated by xAll
extract_operators(xAll, t) = _extract(operator, xAll, t)
# returns all affines (operators keeping displacement) with an order dictated by xAll
extract_affines(xAll, t) = _extract(affine, xAll, t)

# Expand a term/expression to the problem's full domain: every variable of `xAll` the
# term does not mention gets a `Zeros` block, so all terms share one domain and their
# operators can be stacked.
#
# The padding rule itself lives in `add_missing_vars` (addition_tricky_part.jl), which
# does the same job at the operator level for `Usum_op`. Going through it keeps a single
# rule for what a padded block looks like; here it is only wrapped back up as an
# `Expression` over the widened variable tuple.
function expand(xAll::NTuple{N, Variable}, ex::AbstractExpression) where {N}
    ex = convert(Expression, ex)
    new_vars, new_op = add_missing_vars(ex.x, ex.L, xAll)
    return new_vars === ex.x ? ex : Expression(new_vars, new_op)
end

# Preserve λ, f and the term's repr (so diagnostics stay readable after expansion).
expand(xAll::NTuple{N, Variable}, t::Term) where {N} =
    Term(t.lambda, t.f, expand(xAll, t.A), t.repr)
