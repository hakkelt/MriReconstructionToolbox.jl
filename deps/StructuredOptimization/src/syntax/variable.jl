import Base: convert, size, eltype, ~
export Variable, get_name

struct Variable{T, N, A <: AbstractArray{T,N}} <: AbstractExpression
	x::A
  name::String
  function Variable(x::AbstractArray{T,N}; name::String="x") where {T,N}
    A = typeof(x)
    new{T,N,A}(x, name)
  end
end

# constructors
"""
	Variable([T::Type,] dims...; name::String="x")
  Variable(x::AbstractArray; name::String="x")

Creates an optimization variable of type `T` and dimensions `dims...`, or from the provided array `x`.
The optional `name` argument allows to specify a name for the variable, which is useful for display purposes.

"""
function Variable(T::Type, args::Int...; name::String="x")
  Variable(zeros(T, args...); name)
end

function Variable(args::Int...; name::String="x")
  Variable(zeros(args...); name)
end

# Utils

function Base.show(io::IO, x::Variable)
  print(io, "Variable($(eltype(x.x)), $(size(x.x)), \"$(x.name)\")")
end

"""
	~(x::Variable)

Returns the `Array` of the variable `x`
"""
~(x::Variable) = x.x
~(x::Tuple{Variable}) = (~)(x[1])
~(x::NTuple{N,Variable}) where {N} = ArrayPartition((~).(x))

"""
size(x::Variable, [dim...])

Like `size(A::AbstractArray, [dims...])` returns the tuple containing the dimensions of the variable `x`.
"""
size(x::Variable) = size(x.x)
size(x::Variable, dim::Integer) = size(x.x, dim)

"""
eltype(x::Variable)

Like `eltype(x::AbstractArray)` returns the type of the elements of `x`.
"""
eltype(x::Variable) = eltype(x.x)

"""
get_name(x::Variable)

Returns the name of the variable `x`. If no name was provided at construction, returns `"x"`.
"""
get_name(x::Variable) = x.name
