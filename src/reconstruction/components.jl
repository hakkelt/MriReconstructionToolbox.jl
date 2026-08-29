"""
    Component(name::Symbol, regs::Regularization...)

One additive component of an image decomposition (e.g. the low-rank part of an
`L+S` reconstruction). `name` is mandatory and must be unique among the components
passed to `reconstruct`. At least one regularization is required.

# Example
```julia
julia> using MriReconstructionToolbox
julia> Component(:lowrank, LowRank(0.05; time_dim = :time), TemporalFourier(0.01; time_dim = :time))
Component(:lowrank, LowRank(0.05), TemporalFourier(0.01))
```
"""
struct Component{R <: Tuple{Vararg{Regularization}}}
    name::Symbol
    regularizations::R
    function Component(name::Symbol, regs::Regularization...)
        @argcheck !isempty(regs) "Component `$name` needs at least one regularization."
        return new{typeof(regs)}(name, regs)
    end
end

function Base.show(io::IO, c::Component)
    print(io, "Component(:", c.name)
    for reg in c.regularizations
        print(io, ", ", reg)
    end
    return print(io, ")")
end

"""
    check_components(components::Tuple{Vararg{Component}})

Validate a tuple of `Component`s: at least two components, all unique names.
Throws `ArgumentError` otherwise.
"""
function check_components(components::Tuple{Vararg{Component}})
    @argcheck length(components) >= 2 "Image decomposition needs at least two components; use the plain regularization API for a single component."
    names = map(c -> c.name, components)
    @argcheck length(unique(names)) == length(names) "Component names must be unique, got $names."
    return nothing
end

function get_affected_dims(c::Component, acq_info::Union{Nothing, AcquisitionInfo}, image_dims)
    dims = Any[]
    for reg in c.regularizations
        append!(dims, get_affected_dims(reg, acq_info, image_dims))
    end
    return unique(dims)
end

function scale_regularization(c::Component, factor::Real)
    return Component(c.name, map(reg -> scale_regularization(reg, factor), c.regularizations)...)
end

function bind_dimensions(c::Component, image_dims)
    return Component(c.name, map(reg -> bind_dimensions(reg, image_dims), c.regularizations)...)
end
bind_dimensions(c::Component, ::Nothing) = c

function materialize(c::Component, x::Variable; threaded::Bool)
    terms, _ = materialize_with_auxiliaries(c, x; threaded)
    return terms
end

function materialize_with_auxiliaries(c::Component, x::Variable; threaded::Bool)
    # The constructor guarantees at least one regularization, so the reduction needs no seed.
    term_list, auxiliaries = materialize_all(c.regularizations, x; threaded)
    return reduce(+, term_list), auxiliaries
end

"""
    DecomposedImage(total, components::NamedTuple) <: AbstractArray

Result of an image decomposition reconstruction. Behaves as an `AbstractArray`
equal to the sum of its components (`total`); the individual components remain
accessible through `.components` (or the `components` function).

# Example
```julia
img[10, 10, 3]          # sum of components at that index
img.components.lowrank  # low-rank part
Array(img)               # plain Array of the sum
```
"""
struct DecomposedImage{T, N, A <: AbstractArray{T, N}, C <: NamedTuple} <: AbstractArray{T, N}
    total::A
    components::C
end

Base.size(img::DecomposedImage) = size(img.total)
Base.getindex(img::DecomposedImage, I...) = getindex(img.total, I...)
Base.IndexStyle(::Type{<:DecomposedImage{T, N, A}}) where {T, N, A} = IndexStyle(A)
Base.similar(img::DecomposedImage, ::Type{S}, dims::Dims) where {S} = similar(img.total, S, dims)

function Base.setindex!(::DecomposedImage, args...)
    throw(ErrorException("DecomposedImage is read-only; use `Array(img)` for a mutable copy."))
end

function Base.getproperty(img::DecomposedImage, name::Symbol)
    if name === :total || name === :components
        return getfield(img, name)
    else
        return getproperty(getfield(img, :components), name)
    end
end

Base.Array(img::DecomposedImage) = Array(unname(img.total))
Base.convert(::Type{Array}, img::DecomposedImage) = Array(img)
NamedDims.NamedDimsArray(img::DecomposedImage) = img.total isa NamedDimsArray ? img.total : throw(ArgumentError("DecomposedImage has no dimension names."))

"""
    total(img::DecomposedImage)

Return the stored sum array (named or not) of an image-decomposition result.
"""
total(img::DecomposedImage) = img.total

"""
    components(img::DecomposedImage)

Return the `NamedTuple` of the individual components of an image-decomposition result.
"""
components(img::DecomposedImage) = img.components

unname(img::DecomposedImage) = unname(img.total)
dimnames(img::DecomposedImage) = dimnames(img.total)

function rescale!(img::DecomposedImage, factor)
    unname(img.total) .*= factor
    for c in img.components
        unname(c) .*= factor
    end
    return img
end

function Base.show(io::IO, ::MIME"text/plain", img::DecomposedImage)
    print(io, "DecomposedImage{", eltype(img), "} of size ", size(img), " with components ")
    return print(io, join(keys(img.components), ", "))
end
