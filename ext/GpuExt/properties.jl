array_type_display_string(::Type{<:AbstractGPUArray}) = "ᵍᵖᵘ"

_should_thread(::AbstractGPUArray) = false
_should_thread(::Type{<:AbstractGPUArray}) = false

# A `dot` of two views of a device matrix reaches BLAS with device pointers on a backend that
# defines no `dot` of its own (OpenCL), so the sample is one `k × k` Gram product on the
# device instead. `isdiag` indexes scalars, so it runs on a host copy of the (small) product.
function _gram_sample_is_diagonal(A::AbstractGPUMatrix, d::Int)
    k = min(size(A, d), AbstractOperators._GRAM_SAMPLE)
    B = d == 1 ? A[1:k, :] : A[:, 1:k]
    return isdiag(Array(d == 1 ? B * B' : B' * B))
end
_isdiag(G::AbstractGPUArray) = isdiag(Array(G))
