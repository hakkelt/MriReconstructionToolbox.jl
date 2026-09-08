"""
    Component(name::Symbol, regs::Regularization...)

One additive component of an image decomposition (e.g. the low-rank part of an
`L+S` reconstruction). `name` is mandatory and must be unique among the components
passed to `reconstruct`. At least one regularization is required.

# Example
```julia
julia> using MriReconstructionToolbox
julia> Component(:lowrank, LowRank(0.05; time_dim = :time), L1TemporalFourier(0.01; time_dim = :time))
Component(:lowrank, LowRank(0.05), L1TemporalFourier(0.01))
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
    collisions = filter(in(_DECOMPOSED_IMAGE_RESERVED_NAMES), names)
    @argcheck isempty(collisions) "Component name(s) $collisions collide with DecomposedImage's own field(s) $_DECOMPOSED_IMAGE_RESERVED_NAMES; rename the component(s) (reserved names: $_DECOMPOSED_IMAGE_RESERVED_NAMES)."
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
accessible either through `.components` (or the `components` function), or,
more concisely, directly as a property: `img.lowrank` is shorthand for
`img.components.lowrank`. The struct's own fields (`total`, `components`)
always win over a component name, and `check_components`/the constructor
reject a component named `total` or `components`, since such a name would
otherwise be unreachable through dot access.

# Example
```julia
img[10, 10, 3]          # sum of components at that index
img.lowrank              # low-rank part (shorthand)
img.components.lowrank  # low-rank part (equivalent long form)
Array(img)               # plain Array of the sum
```
"""
# `total` and `components` are the struct's real fields; a component sharing either name would
# otherwise be shadowed by `getproperty` below (or shadow the field itself), so both
# `check_components` and the constructor below reject the collision up front.
const _DECOMPOSED_IMAGE_RESERVED_NAMES = (:total, :components)

struct DecomposedImage{T, N, A <: AbstractArray{T, N}, C <: NamedTuple} <: AbstractArray{T, N}
    total::A
    components::C
    function DecomposedImage(total::A, components::C) where {T, N, A <: AbstractArray{T, N}, C <: NamedTuple}
        collisions = filter(in(_DECOMPOSED_IMAGE_RESERVED_NAMES), keys(components))
        @argcheck isempty(collisions) "Component name(s) $collisions collide with DecomposedImage's own field(s) $_DECOMPOSED_IMAGE_RESERVED_NAMES; rename the component(s)."
        return new{T, N, A, C}(total, components)
    end
end

Base.size(img::DecomposedImage) = size(getfield(img, :total))
Base.getindex(img::DecomposedImage, I...) = getindex(getfield(img, :total), I...)
Base.IndexStyle(::Type{<:DecomposedImage{T, N, A}}) where {T, N, A} = IndexStyle(A)
Base.similar(img::DecomposedImage, ::Type{S}, dims::Dims) where {S} = similar(getfield(img, :total), S, dims)

function Base.setindex!(::DecomposedImage, args...)
    throw(ErrorException("DecomposedImage is read-only; use `Array(img)` for a mutable copy."))
end

"""
    Base.getproperty(img::DecomposedImage, name::Symbol)

Real struct fields (`total`, `components`) resolve first; any other `name` is looked up in the
components `NamedTuple`, so `img.lowrank` is shorthand for `img.components.lowrank`. A `name` that
is neither a field nor a component name throws an `ArgumentError` listing both.
"""
function Base.getproperty(img::DecomposedImage, name::Symbol)
    if name === :total || name === :components
        return getfield(img, name)
    end
    comps = getfield(img, :components)
    haskey(comps, name) && return getfield(comps, name)
    throw(ArgumentError("DecomposedImage has no property `$name`; available properties: $(join(propertynames(img), ", "))."))
end

function Base.propertynames(img::DecomposedImage, ::Bool = false)
    return (_DECOMPOSED_IMAGE_RESERVED_NAMES..., keys(getfield(img, :components))...)
end

Base.Array(img::DecomposedImage) = Array(unname(getfield(img, :total)))
Base.convert(::Type{Array}, img::DecomposedImage) = Array(img)
NamedDims.NamedDimsArray(img::DecomposedImage) = getfield(img, :total) isa NamedDimsArray ? getfield(img, :total) : throw(ArgumentError("DecomposedImage has no dimension names."))

"""
    total_image(img::DecomposedImage)

Return the stored sum array (named or not) of an image-decomposition result.
"""
total_image(img::DecomposedImage) = getfield(img, :total)

"""
    components(img::DecomposedImage)

Return the `NamedTuple` of the individual components of an image-decomposition result.
"""
components(img::DecomposedImage) = getfield(img, :components)

unname(img::DecomposedImage) = unname(getfield(img, :total))
dimnames(img::DecomposedImage) = dimnames(getfield(img, :total))

function rescale!(img::DecomposedImage, factor)
    unname(getfield(img, :total)) .*= factor
    for c in getfield(img, :components)
        unname(c) .*= factor
    end
    return img
end

function Base.show(io::IO, ::MIME"text/plain", img::DecomposedImage)
    print(io, "DecomposedImage{", eltype(img), "} of size ", size(img), " with components ")
    return print(io, join(keys(getfield(img, :components)), ", "))
end

# A component is a named bundle of regularizations, so it inherits the trait from them.
# Without this method the component reconstruction path hits a `MethodError` in
# `_iterative_reconstruct_core`, since `method.regularization` is a tuple of `Component`s
# there rather than of `Regularization`s.
uses_blas3(c::Component) = uses_blas3(c.regularizations)
