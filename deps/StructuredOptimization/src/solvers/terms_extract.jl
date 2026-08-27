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
extract_functions(t::TermSet) = SeparableSum(extract_functions.(t))

# extract functions from terms without displacement
function extract_functions_nodisp(t::Term)
  f = t.lambda == 1 ? t.f : Postcompose(t.f, t.lambda)
  return f
end
extract_functions_nodisp(t::TermSet) = SeparableSum(extract_functions_nodisp.(t))

# extract operators from terms

# returns all operators with an order dictated by xAll

#single term, single variable
extract_operators(::Tuple{Variable}, t::AbstractExpression)  = operator(t)
extract_operators(::Tuple{Variable}, t::Term)  = operator(t)
extract_operators(xAll::NTuple{N,Variable}, t::AbstractExpression) where {N} = sort_and_extract_operators(xAll, expand(xAll, t))
extract_operators(xAll::NTuple{N,Variable}, t::Term) where {N} = extract_operators(xAll, TermSet(t,))

#multiple terms, multiple variables
function extract_operators(xAll::NTuple{N,Variable}, t::TermSet) where {N}
  ops = ()
  for ti in t
    tex = expand(xAll,ti)
    ops = (ops...,sort_and_extract_operators(xAll,tex))
  end
  return vcat(ops...)
end

sort_and_extract_operators(::Tuple{Variable}, t::TermOrExpr) = operator(t)

function sort_and_extract_operators(xAll::NTuple{N,Variable}, t::TermOrExpr) where {N}
  p = zeros(Int,N)
  xL = variables(t)
  for i in eachindex(xAll)
    p[i] = findfirst( xi -> xi == xAll[i], xL)
  end
  return operator(t)[p]
end

# extract affines from terms

# returns all affines with an order dictated by xAll

#single term, single variable
extract_affines(::Tuple{Variable}, t::AbstractExpression)  = affine(t)
extract_affines(::Tuple{Variable}, t::Term)  = affine(t)
extract_affines(xAll::NTuple{N,Variable}, t::AbstractExpression) where {N} = sort_and_extract_affines(xAll, expand(xAll, t))
extract_affines(xAll::NTuple{N,Variable}, t::Term) where {N} = extract_affines(xAll, TermSet(t,))

#multiple terms, multiple variables
function extract_affines(xAll::NTuple{N,Variable}, t::TermSet) where {N}
  ops = ()
  for ti in t
    tex = expand(xAll,ti)
    ops = (ops...,sort_and_extract_affines(xAll,tex))
  end
  return vcat(ops...)
end

sort_and_extract_affines(::Tuple{Variable}, t::TermOrExpr) = affine(t)

function sort_and_extract_affines(xAll::NTuple{N,Variable}, t::TermOrExpr) where {N}
  p = zeros(Int,N)
  xL = variables(t)
  for i in eachindex(xAll)
    p[i] = findfirst( xi -> xi == xAll[i], xL)
  end
  return affine(t)[p]
end

# expand term domain dimensions
function expand(xAll::NTuple{N,Variable}, t::Term) where {N}
  xt   = variables(t)
  C    = codomain_type(operator(t))
  size_out = size(operator(t),1)
  ex = t.A

  for x in xAll
    if !( x in variables(t) ) 
      ex += Zeros(eltype(~x),size(x),C,size_out)*x
    end
  end
  return Term(t.lambda, t.f, ex)
end

function expand(xAll::NTuple{N,Variable}, ex::AbstractExpression) where {N}
  ex = convert(Expression,ex)
  xt   = variables(ex)
  C    = codomain_type(operator(ex))
  size_out = size(operator(ex),1)

  for x in xAll
    if !( x in variables(ex) ) 
      ex += Zeros(eltype(~x),size(x),C,size_out)*x
    end
  end
  return ex
end
