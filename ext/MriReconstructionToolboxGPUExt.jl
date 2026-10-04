module MriReconstructionToolboxGPUExt

# The NFFT, operator and proximal packages are submodules of MriReconstructionToolbox, and a submodule
# cannot load package extensions, so their GPU extensions are included here instead. Each goes
# into a module of its own that binds the package it extends, because the extensions import it
# relatively (`..AbstractOperators`) and two of them are called `GpuExt`.

module AbstractOperatorsGPU
    using MriReconstructionToolbox: AbstractOperators
    include(joinpath(@__DIR__, "..", "deps", "AbstractOperators", "ext", "GpuExt", "GpuExt.jl"))
end

module FFTWOperatorsGPU
    using MriReconstructionToolbox: AbstractOperators, FFTWOperators
    include(
        joinpath(@__DIR__, "..", "deps", "AbstractOperators", "FFTWOperators", "ext", "GpuExt", "GpuExt.jl")
    )
end

module NFFTGPU
    using MriReconstructionToolbox: NFFT
    include(joinpath(@__DIR__, "..", "deps", "NFFT", "ext", "NFFTGPUArraysExt", "NFFTGPUArraysExt.jl"))
end

module NFFTOperatorsGPU
    using MriReconstructionToolbox: NFFTOperators
    include(
        joinpath(
            @__DIR__, "..", "deps", "AbstractOperators", "NFFTOperators", "ext", "NFFTOperatorsGPUArraysExt.jl"
        )
    )
end

module ProximalOperatorsGPU
    using MriReconstructionToolbox: ProximalOperators
    include(joinpath(@__DIR__, "..", "deps", "ProximalOperators", "ext", "GpuExt", "GpuExt.jl"))
end

include(joinpath(@__DIR__, "..", "deps", "ProximalOperators", "ext", "GpuRecursiveArrayToolsExt.jl"))

using MriReconstructionToolbox: MriReconstructionToolbox as MRT
using GPUArrays: AbstractGPUArray
using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const


MRT._is_device(::AbstractGPUArray) = true
MRT._device_adaptor(x::AbstractGPUArray) = KA.get_backend(x)

# ─── LORAKS lifts ───────────────────────────────────────────────────────────────────────────
#
# The phase-constrained lifts as kernels: the lift is a gather, one thread per (row, column
# block) writing its entries of the real matrix; the adjoint is rewritten as a gather as well,
# one thread per k-space sample summing every matrix entry that came from it, so that no two
# threads write the same sample. A sample `p` enters the `+` side of window `wi` at offset `ko`
# when `wi = p - ko + 1`, and the reflected side when `wi - ko + 1 ≡ reflect(p)` modulo the grid,
# the reflection being an involution.

@inline _reflect(p::NTuple{N, Int}, center, gridsize) where {N} =
    map((pd, cd, gd) -> mod1(2cd - pd, gd), p, center, gridsize)

@inline _in_window(w, nwin) = all(map((wd, nd) -> 1 <= wd <= nd, w, nwin))

@kernel function _loraks_lift_s_kernel!(M, @Const(x), nwin, ksize, gridsize, center)
    jw, col = @index(Global, NTuple)
    prodk = prod(ksize)
    nrow = prod(nwin)
    nblock = prodk * (length(x) ÷ prod(gridsize))
    c = (col - 1) ÷ prodk + 1
    ko = Tuple(CartesianIndices(ksize)[col - (c - 1) * prodk])
    wi = Tuple(CartesianIndices(nwin)[jw])
    plus = wi .+ ko .- 1
    minus = _reflect(wi .- ko .+ 1, center, gridsize)
    off = (c - 1) * prod(gridsize)
    zp = x[LinearIndices(gridsize)[plus...] + off]
    zm = x[LinearIndices(gridsize)[minus...] + off]
    M[jw, col] = real(zp) - real(zm)
    M[jw, col + nblock] = imag(zm) - imag(zp)
    M[jw + nrow, col] = imag(zp) + imag(zm)
    M[jw + nrow, col + nblock] = real(zp) + real(zm)
end

@kernel function _loraks_lift_g_kernel!(M, @Const(x), nwin, ksize, gridsize, center, nch)
    jw, col = @index(Global, NTuple)
    prodk = prod(ksize)
    nrow = prod(nwin)
    nblock = prodk * nch
    wi = Tuple(CartesianIndices(nwin)[jw])
    if col <= nch
        c = col
        z = x[LinearIndices(gridsize)[_reflect(wi, center, gridsize)...] + (c - 1) * prod(gridsize)]
        M[jw, c] = -real(z)
        M[jw + nrow, c] = imag(z)
    else
        k = col - nch
        c = (k - 1) ÷ prodk + 1
        ko = Tuple(CartesianIndices(ksize)[k - (c - 1) * prodk])
        z = x[LinearIndices(gridsize)[(wi .+ ko .- 1)...] + (c - 1) * prod(gridsize)]
        M[jw, nch + k] = real(z)
        M[jw, nch + nblock + k] = -imag(z)
        M[jw + nrow, nch + k] = imag(z)
        M[jw + nrow, nch + nblock + k] = real(z)
    end
end

@kernel function _loraks_unlift_s_kernel!(y, @Const(M), nwin, ksize, gridsize, center)
    lin, c = @index(Global, NTuple)
    T = eltype(y)
    prodk = prod(ksize)
    nrow = prod(nwin)
    nblock = prodk * (length(y) ÷ prod(gridsize))
    p = Tuple(CartesianIndices(gridsize)[lin])
    q = _reflect(p, center, gridsize)
    acc = zero(T)
    for jk in 1:prodk
        ko = Tuple(CartesianIndices(ksize)[jk])
        col = (c - 1) * prodk + jk
        w = p .- ko .+ 1
        if _in_window(w, nwin)
            jw = LinearIndices(nwin)[w...]
            d11, d12 = M[jw, col], M[jw, col + nblock]
            d21, d22 = M[jw + nrow, col], M[jw + nrow, col + nblock]
            acc += T(d11 + d22, d21 - d12)
        end
        w = map((qd, kd, gd) -> mod1(qd + kd - 1, gd), q, ko, gridsize)
        if _in_window(w, nwin)
            jw = LinearIndices(nwin)[w...]
            d11, d12 = M[jw, col], M[jw, col + nblock]
            d21, d22 = M[jw + nrow, col], M[jw + nrow, col + nblock]
            acc += T(d22 - d11, d12 + d21)
        end
    end
    y[lin + (c - 1) * prod(gridsize)] = acc
end

@kernel function _loraks_unlift_g_kernel!(y, @Const(M), nwin, ksize, gridsize, center, nch)
    lin, c = @index(Global, NTuple)
    T = eltype(y)
    prodk = prod(ksize)
    nrow = prod(nwin)
    nblock = prodk * nch
    p = Tuple(CartesianIndices(gridsize)[lin])
    acc = zero(T)
    w = _reflect(p, center, gridsize)
    if _in_window(w, nwin)
        jw = LinearIndices(nwin)[w...]
        acc += T(-M[jw, c], M[jw + nrow, c])
    end
    for jk in 1:prodk
        ko = Tuple(CartesianIndices(ksize)[jk])
        w = p .- ko .+ 1
        if _in_window(w, nwin)
            jw = LinearIndices(nwin)[w...]
            k = (c - 1) * prodk + jk
            acc += T(M[jw, nch + k] + M[jw + nrow, nch + nblock + k], M[jw + nrow, nch + k] - M[jw, nch + nblock + k])
        end
    end
    y[lin + (c - 1) * prod(gridsize)] = acc
end

# The slabs reach these as views of the batch array; the kernels read and write them through a
# contiguous copy of the slab, which the view of a column-major slab already is in memory.
_slab(x) = x isa AbstractGPUArray ? x : copy(x)

function MRT._loraks_lift!(M::AbstractGPUArray, lift::MRT.LoraksLift, x::AbstractArray, ::Val{:s})
    backend = KA.get_backend(M)
    ncol = prod(lift.ksize) * lift.nchannels
    _loraks_lift_s_kernel!(backend)(
        M, vec(_slab(x)), lift.nwin, lift.ksize, lift.gridsize, lift.center; ndrange = (prod(lift.nwin), ncol)
    )
    return M
end

function MRT._loraks_lift!(M::AbstractGPUArray, lift::MRT.LoraksLift, x::AbstractArray, ::Val{:g})
    backend = KA.get_backend(M)
    ncol = prod(lift.ksize) * lift.nchannels + lift.nchannels
    _loraks_lift_g_kernel!(backend)(
        M, vec(_slab(x)), lift.nwin, lift.ksize, lift.gridsize, lift.center, lift.nchannels;
        ndrange = (prod(lift.nwin), ncol),
    )
    return M
end

function _loraks_unlift_device!(y, lift, M, kernel!, extra...)
    backend = KA.get_backend(M)
    out = y isa AbstractGPUArray ? y : similar(M, eltype(y), size(y))
    kernel!(backend)(
        vec(out), M, lift.nwin, lift.ksize, lift.gridsize, lift.center, extra...;
        ndrange = (prod(lift.gridsize), lift.nchannels),
    )
    out === y || copyto!(y, out)
    return y
end

MRT._loraks_unlift!(y::AbstractArray, lift::MRT.LoraksLift, M::AbstractGPUArray, ::Val{:s}) =
    _loraks_unlift_device!(y, lift, M, _loraks_unlift_s_kernel!)
MRT._loraks_unlift!(y::AbstractArray, lift::MRT.LoraksLift, M::AbstractGPUArray, ::Val{:g}) =
    _loraks_unlift_device!(y, lift, M, _loraks_unlift_g_kernel!, lift.nchannels)

end # module MriReconstructionToolboxGPUExt
