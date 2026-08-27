# Variation: KernelAbstractions @kernel implementations.

# JLBackend (JLArrays) is synchronous — KA provides no synchronize method for it.
# Use this guard to avoid a MethodError on JLBackend while still synchronizing
# asynchronous GPU backends (CUDA, AMDGPU).
_ka_synchronize(backend) =
    hasmethod(KernelAbstractions.synchronize, (typeof(backend),)) &&
    KernelAbstractions.synchronize(backend)

@kernel function _var_fwd_dim1_kernel!(y_col, @Const(b_flat), dim_stride)
    cnt = @index(Global, Linear)
    n = length(b_flat)
    i1 = (cnt - 1) % dim_stride + 1
    if i1 == 1
        next = cnt + 1 <= n ? cnt + 1 : cnt
        y_col[cnt] = b_flat[next] - b_flat[cnt]
    else
        y_col[cnt] = b_flat[cnt] - b_flat[cnt - 1]
    end
end

@kernel function _var_fwd_dimd_kernel!(y_col, @Const(b_flat), dim_stride, nd)
    cnt = @index(Global, Linear)
    k = (cnt - 1) ÷ dim_stride
    pos = k % nd
    if pos == 0
        y_col[cnt] = b_flat[cnt + dim_stride] - b_flat[cnt]
    else
        y_col[cnt] = b_flat[cnt] - b_flat[cnt - dim_stride]
    end
end

@kernel function _var_adj_kernel!(y_flat, @Const(b), @Const(dim_in_ntuple))
    cnt = @index(Global, Linear)
    dim_in = dim_in_ntuple
    N = length(dim_in)
    T = eltype(y_flat)
    acc = zero(T)
    dim_stride = 1
    for d in 1:N
        nd = dim_in[d]
        i_d = ((cnt - 1) ÷ dim_stride) % nd + 1
        acc += if i_d == 1
            -(b[cnt, d] + b[cnt + dim_stride, d])
        elseif i_d == nd
            b[cnt, d]
        elseif i_d == 2
            b[cnt, d] + b[cnt - dim_stride, d] - b[cnt + dim_stride, d]
        else
            b[cnt, d] - b[cnt + dim_stride, d]
        end
        dim_stride *= nd
    end
    y_flat[cnt] = acc
end

function _var_fwd_gpu!(y::AbstractGPUArray, A::Variation, b::AbstractGPUArray)
    backend = KernelAbstractions.get_backend(b)
    n = length(b)
    b_flat = reshape(b, n)

    dim1_extent = size(b, 1)
    ker1 = _var_fwd_dim1_kernel!(backend, 256)
    ker1(view(y, :, 1), b_flat, dim1_extent; ndrange = n)

    N = ndims(b)
    dim_stride = dim1_extent
    for d in 2:N
        nd = size(b, d)
        kerd = _var_fwd_dimd_kernel!(backend, 256)
        kerd(view(y, :, d), b_flat, dim_stride, nd; ndrange = n)
        dim_stride *= nd
    end
    _ka_synchronize(backend)
    return y
end

function _var_adj_gpu!(y::AbstractGPUArray, A::AdjointOperator{<:Variation}, b::AbstractGPUArray)
    backend = KernelAbstractions.get_backend(y)
    n = length(y)
    ker = _var_adj_kernel!(backend, 256)
    ker(reshape(y, n), b, A.A.dim_in; ndrange = n)
    _ka_synchronize(backend)
    return y
end

function mul!(y::AbstractGPUArray, A::Variation{T, N, false}, b::AbstractGPUArray) where {T, N}
    check(y, A, b)
    return _var_fwd_gpu!(y, A, b)
end

function mul!(y::AbstractGPUArray, A::Variation{T, N, true}, b::AbstractGPUArray) where {T, N}
    check(y, A, b)
    return _var_fwd_gpu!(y, A, b)
end

function mul!(
        y::AbstractGPUArray, A::AdjointOperator{<:Variation{T, N, false}}, b::AbstractGPUArray
    ) where {T, N}
    check(y, A, b)
    return _var_adj_gpu!(y, A, b)
end

function mul!(
        y::AbstractGPUArray, A::AdjointOperator{<:Variation{T, N, true}}, b::AbstractGPUArray
    ) where {T, N}
    check(y, A, b)
    return _var_adj_gpu!(y, A, b)
end
