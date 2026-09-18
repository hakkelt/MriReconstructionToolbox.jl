# returns all variables of a cost function, in terms of appearance
extract_variables(t::TermOrExpr) = variables(t)

function extract_variables(t::Union{Tuple, TermSet})
    var_tuples = variables.(t)
    vars = collect(Base.Iterators.flatten(var_tuples))
    return tuple(unique(vars)...)
end

# extract functions from terms
function extract_functions(t::Term)
    disp = displacement(t)
    f = disp == 0 ? t.f : PrecomposeDiagonal(t.f, one(t.lambda), disp) #for now I keep this
    f = t.lambda == 1 ? f : Postcompose(f, t.lambda)                                  #for now I keep this
    #TODO change this
    return f
end
extract_functions(t::TermSet) = SeparableSum(extract_functions.(t)...)

# extract functions from terms without displacement
function extract_functions_nodisp(t::Term)
    f = t.lambda == 1 ? t.f : Postcompose(t.f, t.lambda)
    return f
end
extract_functions_nodisp(t::TermSet) = SeparableSum(extract_functions_nodisp.(t)...)

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

# expand term domain dimensions
function expand(xAll::NTuple{N, Variable}, t::Term) where {N}
    C = codomain_type(operator(t))
    size_out = size(operator(t), 1)
    ex = t.A

    for x in xAll
        if !(x in variables(t))
            ex += Zeros(eltype(~x), size(x), C, size_out) * x
        end
    end
    # Preserve the term's repr so diagnostics stay readable after expansion.
    return Term(t.lambda, t.f, ex, t.repr)
end

function expand(xAll::NTuple{N, Variable}, ex::AbstractExpression) where {N}
    ex = convert(Expression, ex)
    C = codomain_type(operator(ex))
    size_out = size(operator(ex), 1)

    for x in xAll
        if !(x in variables(ex))
            ex += Zeros(eltype(~x), size(x), C, size_out) * x
        end
    end
    return ex
end
