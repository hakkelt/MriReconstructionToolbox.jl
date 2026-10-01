# Moving an acquisition between host and device storage with `Adapt.adapt`.
#
# The k-space, the sensitivity maps and the density compensation move to the target storage;
# the subsampling pattern and the trajectory stay on the host, because the operators built from
# them (`GetIndex`, the NFFT plan) take host indices and a host trajectory on every storage, and
# move to the device what they apply there.

_adapt_array(to, ::Nothing) = nothing
_adapt_array(to, x::NamedDimsArray{L}) where {L} = NamedDimsArray{L}(Adapt.adapt(to, parent(x)))
_adapt_array(to, x::AbstractArray) = Adapt.adapt(to, x)

Adapt.adapt_structure(to, p::PartitionedKSpace) =
    PartitionedKSpace(map(part -> _adapt_array(to, part), p.parts), p.ragged_dim, p.dimnames)

_adapt_kspace(to, ksp) = _adapt_array(to, ksp)
_adapt_kspace(to, ksp::PartitionedKSpace) = Adapt.adapt_structure(to, ksp)

Adapt.adapt_structure(to, info::CartesianAcquisitionInfo) = CartesianAcquisitionInfo(
    info;
    kspace_data = _adapt_kspace(to, info.kspace_data),
    sensitivity_maps = _adapt_array(to, info.sensitivity_maps),
)

Adapt.adapt_structure(to, info::NonCartesianAcquisitionInfo) = NonCartesianAcquisitionInfo(
    info;
    kspace_data = _adapt_kspace(to, info.kspace_data),
    sensitivity_maps = _adapt_array(to, info.sensitivity_maps),
    dcf = _adapt_array(to, info.dcf),
)

"""
    _is_device(x) -> Bool

Whether `x` is held in device (GPU) memory. Wrappers (`NamedDimsArray`, views, reshapes) are
looked through. `false` for everything the GPU extension does not claim, including `nothing`.
"""
_is_device(x) = false
_is_device(x::NamedDimsArray) = _is_device(parent(x))
_is_device(x::SubArray) = _is_device(parent(x))
_is_device(x::Base.ReshapedArray) = _is_device(parent(x))
_is_device(x::Base.ReinterpretArray) = _is_device(parent(x))
_is_device(x::PartitionedKSpace) = _is_device(first(x.parts))
_is_device(info::AcquisitionInfo) = _is_device(info.kspace_data)

"""
    _release_device_plans!(op, storage)

Hand the device FFT plans inside `op` back to their library's plan cache, so the next operator of
the same shape is planned from it. `op` must not be applied afterwards. `storage` is the array
`op` was built for (see [`_storage_template`](@ref)); nothing happens unless a device backend's
extension adds a method for its array type: CUDA's plans otherwise return to the cache only when
the garbage collector finalizes them, and until then every reconstruction plans its FFTs anew,
2.2 ms per 128×128 plan against 5 µs from the cache (Quadro RTX 6000).
"""
_release_device_plans!(op, storage) = nothing

"""
    _storage_template(x)

The unwrapped array whose storage `x` lives in, for `similar` and `adapt`: the parent of a
`NamedDimsArray`, the first part of a `PartitionedKSpace`.
"""
_storage_template(x::AbstractArray) = x
_storage_template(x::Union{NamedDimsArray, SubArray, Base.ReshapedArray}) = _storage_template(parent(x))
_storage_template(x::PartitionedKSpace) = _storage_template(first(x.parts))
_storage_template(info::AcquisitionInfo) = _storage_template(info.kspace_data)

# `x` on the host: itself if it is there already, a host copy otherwise.
_to_host(x) = _is_device(x) ? Array(x) : x

# The array type operators built for `x` allocate with (their `array_type` keyword).
_array_type_of(x) = typeof(_storage_template(x))

# `adapt` that keeps `NamedDimsArray` names and recurses into tuples; everything else goes
# through `Adapt.adapt`, so an `AcquisitionInfo` uses the methods above.
_adapt_any(to, x) = Adapt.adapt(to, x)
_adapt_any(to, x::NamedDimsArray) = _adapt_array(to, x)
_adapt_any(to, x::Tuple) = map(xi -> _adapt_any(to, xi), x)
_adapt_any(to, x::NamedTuple) = map(xi -> _adapt_any(to, xi), x)

"""
    _to_storage_of(template, x)

`x` moved to the storage `template` lives in: unchanged on the host, `adapt`ed to the device
otherwise.
"""
_to_storage_of(template, x) = _is_device(template) ? _adapt_any(_storage_adaptor(template), x) : x

# What `adapt` converts to for `template`'s storage: the KernelAbstractions backend of a device
# array (defined by the GPU extension), `Array` otherwise.
_storage_adaptor(template) = _device_adaptor(_storage_template(template))
_device_adaptor(::AbstractArray) = Array

"""
    _on_host(f, template, args...)

Run `f` on host copies of `args` and move the result back to the storage `template` lives in.
Used by the preprocessing and direct methods whose kernels are scalar loops: on the host nothing
is copied.
"""
function _on_host(f, template, args...)
    _is_device(template) || return f(args...)
    return _to_storage_of(template, f(_adapt_any(Array, args)...))
end
