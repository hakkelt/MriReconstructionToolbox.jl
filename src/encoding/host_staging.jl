# Operators with no device kernels (wavelets, contourlets) run on the host inside a device
# reconstruction: they are built for a host array of the same shape and wrapped so that each
# apply copies its input to the host and its output back.

"""
    _host_template(x)

A host array of `x`'s element type, shape and dimension names, to build a host-only operator for.
"""
_host_template(x::NamedDimsArray{L}) where {L} = NamedDimsArray{L}(_host_template(parent(x)))
_host_template(x::AbstractArray{T}) where {T} = Array{T}(undef, size(x))

"""
    _host_staged(build, x)

`build(template)` for an operator acting on arrays like `x`. On the host that is `build(x)`; on a
device the operator is built for a host template and wrapped in an `OperatorWrapper` whose
domain and codomain are `x`'s device storage.
"""
function _host_staged(build, x::AbstractArray)
    _is_device(x) || return build(x)
    return _wrap_on_host(build(_host_template(x)), _array_type_of(x))
end

_wrap_on_host(op::NamedDimsOp{D, C}, array_type) where {D, C} =
    NamedDimsOp{D, C}(_wrap_on_host(op.L, array_type))
_wrap_on_host(op::AbstractOperators.AbstractOperator, array_type) =
    OperatorWrapper(op; array_type)
