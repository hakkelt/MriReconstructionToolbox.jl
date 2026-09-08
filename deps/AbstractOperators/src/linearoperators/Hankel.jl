export Hankel

"""
	Hankel([domain_type::Type,] gridsize::Tuple, ksize::Tuple; nchannels=1, structure=:c, array_type)
	Hankel(x::AbstractArray, ksize::Tuple; channels=false, structure=:c)

Create a `LinearOperator` which lifts an array into a **block-Hankel matrix** built from all
sliding windows of size `ksize` over the first `length(ksize)` dimensions of the input (the
"embedded" dimensions of size `gridsize`). This embedding is also known as the
sliding-window, trajectory, or Page matrix, and is the basic building block of subspace
methods (SSA, ESPRIT/MUSIC), Cadzow denoising, dynamic mode decomposition and structured
low-rank matrix completion.

For an input `x` of size `gridsize` the output `y` is a matrix of size
`(prod(gridsize .- ksize .+ 1), prod(ksize))`: row `r` holds, flattened, the window of `x`
whose lower corner is the `r`-th element of `CartesianIndices(gridsize .- ksize .+ 1)`.

If `channels=true` (or `nchannels>1`) the input carries one trailing axis of length
`nchannels`; each channel contributes its own block of `prod(ksize)` columns, so the output
has `prod(ksize) * nchannels` columns. This is the multi-coil ("SAKE") layout.

The adjoint scatters each matrix entry back to the k-space sample it came from, summing over
the overlapping windows. Consequently `A'A` is the real diagonal operator that multiplies
each input sample by the number of windows it appears in; `get_normal_op` returns exactly
that `DiagOp`.

Only `structure=:c` (plain block-Hankel) is currently implemented.

```jldoctest
julia> H = Hankel(Float64, (5,), (3,))
𝓗  ℝ^5 -> ℝ^(3, 3)

julia> H * collect(1.0:5.0)
3×3 Matrix{Float64}:
 1.0  2.0  3.0
 2.0  3.0  4.0
 3.0  4.0  5.0

```
"""
struct Hankel{T, N, C, S <: AbstractArray{T}, M <: AbstractArray} <: LinearOperator
    gridsize::NTuple{N, Int}
    ksize::NTuple{N, Int}
    nwin::NTuple{N, Int}
    nchannels::Int
    mult::M
end

function _hankel_mult(gridsize::NTuple{N, Int}, ksize::NTuple{N, Int}, nwin::NTuple{N, Int}) where {N}
    m = Array{Int, N}(undef, gridsize...)
    @inbounds for idx in CartesianIndices(gridsize)
        c = 1
        for d in 1:N
            i = idx[d]
            lo = max(1, i - ksize[d] + 1)
            hi = min(i, nwin[d])
            c *= (hi - lo + 1)
        end
        m[idx] = c
    end
    return m
end

# Constructors

function Hankel(
        domain_type::Type{T}, gridsize::NTuple{N, Int}, ksize::NTuple{N, Int};
        nchannels::Int = 1, channels::Bool = nchannels > 1,
        structure::Symbol = :c, array_type::Type = Array{T},
    ) where {T, N}
    structure === :c || error("Hankel: only structure = :c is currently supported")
    all(ksize .>= 1) || error("Hankel: window sizes must be positive")
    all(ksize .<= gridsize) || error("Hankel: window cannot exceed grid size")
    nchannels >= 1 || error("Hankel: nchannels must be positive")
    nwin = gridsize .- ksize .+ 1
    S = _normalize_array_type(array_type, T)
    mult = _hankel_mult(gridsize, ksize, nwin)
    C = channels || nchannels > 1
    return Hankel{T, N, C, S, typeof(mult)}(gridsize, ksize, nwin, nchannels, mult)
end

function Hankel(gridsize::NTuple{N, Int}, ksize::NTuple{N, Int}; kwargs...) where {N}
    return Hankel(Float64, gridsize, ksize; kwargs...)
end

function Hankel(x::AbstractArray{T}, ksize::NTuple{N, Int}; channels::Bool = false, structure::Symbol = :c) where {T, N}
    if channels
        ndims(x) == N + 1 || error("Hankel: with channels=true, ndims(x) must be length(ksize) + 1")
        gridsize = ntuple(d -> size(x, d), N)
        nchannels = size(x, N + 1)
    else
        ndims(x) == N || error("Hankel: ndims(x) must match length(ksize)")
        gridsize = size(x)
        nchannels = 1
    end
    S = _normalize_array_type(_array_wrapper(x), T)
    return Hankel(T, gridsize, ksize; nchannels, channels, structure, array_type = S)
end

# Mappings

function mul!(y::AbstractMatrix, L::Hankel{T, N, C}, b::AbstractArray) where {T, N, C}
    check(y, L, b)
    fill!(y, zero(T))
    winidx = CartesianIndices(L.nwin)
    koff = CartesianIndices(L.ksize)
    shift = oneunit(eltype(koff))
    prodk = prod(L.ksize)
    @inbounds for c in 1:L.nchannels
        bc = C ? view(b, ntuple(_ -> Colon(), N)..., c) : b
        coloff = (c - 1) * prodk
        jk = 0
        for ko in koff
            jk += 1
            col = coloff + jk
            jw = 0
            for wi in winidx
                jw += 1
                y[jw, col] = bc[wi + ko - shift]
            end
        end
    end
    return y
end

function mul!(b::AbstractArray, L::AdjointOperator{<:Hankel{T, N, C}}, y::AbstractMatrix) where {T, N, C}
    A = L.A
    check(b, L, y)
    fill!(b, zero(T))
    winidx = CartesianIndices(A.nwin)
    koff = CartesianIndices(A.ksize)
    shift = oneunit(eltype(koff))
    prodk = prod(A.ksize)
    @inbounds for c in 1:A.nchannels
        bc = C ? view(b, ntuple(_ -> Colon(), N)..., c) : b
        coloff = (c - 1) * prodk
        jk = 0
        for ko in koff
            jk += 1
            col = coloff + jk
            jw = 0
            for wi in winidx
                jw += 1
                bc[wi + ko - shift] += y[jw, col]
            end
        end
    end
    return b
end

# Properties

domain_type(::Hankel{T}) where {T} = T
codomain_type(::Hankel{T}) where {T} = T
domain_array_type(::Hankel{T, N, C, S}) where {T, N, C, S} = S
codomain_array_type(::Hankel{T, N, C, S}) where {T, N, C, S} = S
is_thread_safe(::Hankel) = true

function size(L::Hankel{T, N, C}) where {T, N, C}
    dim_in = C ? (L.gridsize..., L.nchannels) : L.gridsize
    dim_out = (prod(L.nwin), prod(L.ksize) * L.nchannels)
    return dim_out, dim_in
end

fun_name(::Hankel) = "𝓗"

function _hankel_diag(L::Hankel{T, N, C}) where {T, N, C}
    d = Array{T, N}(undef, L.gridsize...)
    @inbounds @. d = T(L.mult)
    if C
        rep = similar(d, L.gridsize..., L.nchannels)
        @inbounds for c in 1:L.nchannels
            selectdim(rep, N + 1, c) .= d
        end
        return rep
    end
    return d
end

is_AcA_diagonal(::Hankel) = true
diag_AcA(L::Hankel) = _hankel_diag(L)

has_optimized_normalop(::Hankel) = true
get_normal_op(L::Hankel) = DiagOp(_hankel_diag(L))

is_full_column_rank(::Hankel) = true

has_fast_opnorm(::Hankel) = true
LinearAlgebra.opnorm(L::Hankel) = sqrt(real(domain_type(L))(maximum(L.mult)))

# `threaded` is accepted for uniform forwarding and has no effect (no threaded path).
# `storage_type` is honoured.
function _copy_operator_impl(
        op::Hankel{T, N, C, S}; storage_type = nothing, threaded = nothing
    ) where {T, N, C, S}
    S2 = storage_type === nothing ? S : _normalize_array_type(storage_type, T)
    # `mult` is read-only metadata -> shared, not copied (see AGENTS.md copy semantics).
    return Hankel{T, N, C, S2, typeof(op.mult)}(op.gridsize, op.ksize, op.nwin, op.nchannels, op.mult)
end
