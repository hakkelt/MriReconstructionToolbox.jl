# Hankel: KernelAbstractions kernels in place of the scalar window loops.
#
# The forward lift is a gather, one thread per matrix entry. The adjoint sums every entry that
# came from the same sample; it is written as a gather too, one thread per sample over the
# `prod(ksize)` windows that can contain it, so no two threads write the same element.

@kernel function _hankel_fwd_kernel!(y, @Const(b), nwin, ksize, gridsize)
    jw, col = @index(Global, NTuple)
    prodk = prod(ksize)
    c = (col - 1) ÷ prodk + 1
    jk = col - (c - 1) * prodk
    wi = CartesianIndices(nwin)[jw]
    ko = CartesianIndices(ksize)[jk]
    p = LinearIndices(gridsize)[(Tuple(wi) .+ Tuple(ko) .- 1)...]
    y[jw, col] = b[p + (c - 1) * prod(gridsize)]
end

@kernel function _hankel_adj_kernel!(b, @Const(y), nwin, ksize, gridsize)
    lin, c = @index(Global, NTuple)
    prodk = prod(ksize)
    p = Tuple(CartesianIndices(gridsize)[lin])
    acc = zero(eltype(b))
    for jk in 1:prodk
        w = p .- Tuple(CartesianIndices(ksize)[jk]) .+ 1
        if all(map((wd, nd) -> 1 <= wd <= nd, w, nwin))
            acc += y[LinearIndices(nwin)[w...], (c - 1) * prodk + jk]
        end
    end
    b[lin + (c - 1) * prod(gridsize)] = acc
end

function mul!(y::AbstractGPUArray{<:Any, 2}, L::Hankel{T, N, C}, b::AbstractGPUArray) where {T, N, C}
    check(y, L, b)
    backend = KernelAbstractions.get_backend(y)
    _hankel_fwd_kernel!(backend)(y, vec(b), L.nwin, L.ksize, L.gridsize; ndrange = size(y))
    _ka_synchronize(backend)
    return y
end

function mul!(
        b::AbstractGPUArray, Lc::AdjointOperator{<:Hankel{T, N, C}}, y::AbstractGPUArray{<:Any, 2}
    ) where {T, N, C}
    L = Lc.A
    check(b, Lc, y)
    backend = KernelAbstractions.get_backend(b)
    _hankel_adj_kernel!(backend)(
        vec(b), y, L.nwin, L.ksize, L.gridsize; ndrange = (prod(L.gridsize), L.nchannels)
    )
    _ka_synchronize(backend)
    return b
end
