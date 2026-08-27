# ZeroPad: vectorized GPU mul! to avoid scalar indexing in @generated loop

function mul!(y::AbstractGPUArray, L::ZeroPad, b::AbstractGPUArray)
    check(y, L, b)
    fill!(y, zero(eltype(y)))
    N = length(L.dim_in)
    dst = view(y, ntuple(i -> 1:L.dim_in[i], N)...)
    copyto!(dst, b)
    return y
end

function mul!(y::AbstractGPUArray, Lc::AdjointOperator{<:ZeroPad}, b::AbstractGPUArray)
    check(y, Lc, b)
    N = length(Lc.A.dim_in)
    src = view(b, ntuple(i -> 1:Lc.A.dim_in[i], N)...)
    copyto!(y, src)
    return y
end
