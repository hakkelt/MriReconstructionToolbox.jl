"""
	BlockNuclearNorm(λ, block_size, spatial_size, num_frames, num_batch; threaded=true)

Proximable function implementing the sum of nuclear norms of the Casorati matrices of a set of
non-overlapping spatial blocks: `f(x) = λ ∑_b ‖𝓧_b‖_*`, where `𝓧_b` is the `(∏ block_size) × num_frames`
matrix formed by the voxels of block `b`.

This is the function behind [`LocallyLowRank`](@ref); it is applied to the image itself (identity operator),
because the block decomposition is a permutation of the voxels and the prox therefore decouples exactly into
one singular value thresholding per block.
"""
struct BlockNuclearNorm{R <: Real, N}
    λ::R
    block_size::NTuple{N, Int}
    spatial_size::NTuple{N, Int}
    num_frames::Int
    num_batch::Int
    threaded::Bool
    # The tiling depends only on the fields above, so it is computed once instead of per prox call.
    blocks::Vector{NTuple{N, UnitRange{Int}}}
end

function BlockNuclearNorm(
        λ::R, block_size::NTuple{N, Int}, spatial_size::NTuple{N, Int},
        num_frames::Int, num_batch::Int, threaded::Bool
    ) where {R <: Real, N}
    return BlockNuclearNorm{R, N}(
        λ, block_size, spatial_size, num_frames, num_batch, threaded,
        _block_ranges(spatial_size, block_size)
    )
end

ProximalCore.is_convex(::Type{<:BlockNuclearNorm}) = true
ProximalCore.is_positively_homogeneous(::Type{<:BlockNuclearNorm}) = true
ProximalCore.is_smooth(::Type{<:BlockNuclearNorm}) = false
ProximalCore.is_separable(::Type{<:BlockNuclearNorm}) = false

"""
	_block_ranges(spatial_size, block_size)

Return the tuple of index ranges of every block of the (possibly incomplete at the boundary) tiling of
`spatial_size` with blocks of size `block_size`.
"""
function _block_ranges(spatial_size::NTuple{N, Int}, block_size::NTuple{N, Int}) where {N}
    per_dim = ntuple(
        d -> [i:min(i + block_size[d] - 1, spatial_size[d]) for i in 1:block_size[d]:spatial_size[d]],
        Val(N)
    )
    return vec([ranges for ranges in Iterators.product(per_dim...)])
end

_reshaped(f::BlockNuclearNorm, x) = reshape(x, f.spatial_size..., f.num_frames, f.num_batch)

# Gather one block into a dense `(voxels × frames)` Casorati matrix. The block is a strided but
# non-contiguous view, so it has to be copied before the SVD anyway.
function _gather_block!(M, xr, ranges::NTuple{N, UnitRange{Int}}, batch) where {N}
    block = @view xr[ranges..., :, batch]
    return copyto!(M, reshape(block, size(M)...))
end

function _scatter_block!(xr, M, ranges::NTuple{N, UnitRange{Int}}, batch) where {N}
    block = @view xr[ranges..., :, batch]
    return copyto!(block, reshape(M, size(block)...))
end

function _block_matrix(f::BlockNuclearNorm, ::Type{T}, ranges) where {T}
    return Matrix{T}(undef, prod(length.(ranges)), f.num_frames)
end

function (f::BlockNuclearNorm)(x)
    xr = _reshaped(f, x)
    R = real(eltype(x))
    value = R(0)
    for batch in 1:(f.num_batch), ranges in f.blocks
        M = _block_matrix(f, eltype(x), ranges)
        _gather_block!(M, xr, ranges, batch)
        value += sum(svdvals!(M))
    end
    return f.λ * value
end

function ProximalCore.prox!(y, f::BlockNuclearNorm, x, gamma)
    xr = _reshaped(f, x)
    yr = _reshaped(f, y)
    R = real(eltype(x))
    threshold = f.λ * gamma
    nblocks = length(f.blocks)
    # One task per (block, batch slice) pair; every task writes into a disjoint part of `y`.
    partial = zeros(R, nblocks * f.num_batch)
    if f.threaded
        @budgeted_threads for k in eachindex(partial)
            partial[k] = _prox_block!(yr, xr, f, k, nblocks, threshold)
        end
    else
        for k in eachindex(partial)
            partial[k] = _prox_block!(yr, xr, f, k, nblocks, threshold)
        end
    end
    return f.λ * sum(partial)
end

function _prox_block!(yr, xr, f::BlockNuclearNorm, k::Int, nblocks::Int, threshold)
    ranges = f.blocks[mod1(k, nblocks)]
    batch = cld(k, nblocks)
    R = real(eltype(xr))
    M = _block_matrix(f, eltype(xr), ranges)
    _gather_block!(M, xr, ranges, batch)
    F = svd!(M)
    σ = max.(R(0), F.S .- threshold)
    lmul!(Diagonal(σ), F.Vt)
    mul!(M, F.U, F.Vt)
    _scatter_block!(yr, M, ranges, batch)
    return sum(σ)
end

"""
	LocallyLowRank(λ; block_size, time_dim=nothing)

Create a locally low-rank (LLR) regularization term with parameter `λ`. The image is tiled into
non-overlapping spatial blocks, and the nuclear norm of the Casorati matrix of every block is penalized:
`λ ∑_b ‖𝓧_b‖_*`, where the rows of `𝓧_b` are the voxels of block `b` and its columns are the frames along
`time_dim`.

# Arguments
- `λ`: Regularization parameter, must be a scalar.
- `block_size`: Edge length of the blocks. Either an `Integer` (same length in every spatial dimension) or a
  tuple with one entry per spatial dimension (the dimensions before `time_dim`). Typical values are 4-16.
- `time_dim`: (optional) Dimension holding the frames / contrasts. Can be an `Integer` (1-based index) or a
  `Symbol` (dimension name). If not provided, it will be inferred as the dimension named `:time` if `x` is a
  `NamedDimsArray`.

# Notes
- LLR exploits that a *local* neighbourhood has far fewer independent temporal dynamics than the whole
  image, so it is a strictly stronger prior than the globally low-rank [`LowRank`](@ref) whenever the
  dynamics vary across the field of view. Introduced for dynamic MRI and parameter mapping by
  Trzasko & Manduca, *Local versus global low-rank promotion in dynamic MRI series reconstruction*,
  ISMRM 2011, and Zhang, Pauly & Levesque, *Accelerating parameter mapping with a locally low rank
  constraint*, Magn Reson Med 2015.
- The prox is exact: since the blocks tile the image without overlap, the block decomposition is a
  permutation of the voxels and the proximal operator decouples into one singular value thresholding per
  block. The blocks are however fixed to one grid; the block artifacts this can produce are usually removed
  in the literature by randomly shifting the grid between iterations, which is not done here.
- Cost is dominated by one SVD of a `(∏ block_size) × n_frames` matrix per block and iteration, so small
  blocks are cheaper but capture less spatial correlation.
"""
struct LocallyLowRank{T, B, D} <: Regularization
    λ::T
    block_size::B
    time_dim::D
    function LocallyLowRank(λ::T; block_size::B, time_dim::D = nothing) where {T, B, D}
        @argcheck λ isa Real "LocallyLowRank requires a scalar λ"
        @argcheck block_size isa Integer || block_size isa Tuple{Vararg{Integer}} "block_size must be an Integer or a tuple of Integers"
        @argcheck all(block_size .> 0) "block_size must be positive"
        _check_lowrank_dims(time_dim)
        return new{T, B, D}(λ, block_size, time_dim)
    end
end

get_operator(::LocallyLowRank, x::AbstractArray; threaded::Bool = true) = Eye(x)
function get_operator(::LocallyLowRank, x::NamedDimsArray; threaded::Bool = true)
    return NamedDimsOp{dimnames(x), dimnames(x)}(Eye(parent(x)))
end

function get_affected_dims(reg::LocallyLowRank, ::Nothing, image_dims)
    # Blocks couple all spatial dimensions up to (and including) the temporal one.
    return image_dims[1:get_time_dim(reg.time_dim, image_dims)]
end

function get_affected_dims(reg::LocallyLowRank, ::AcquisitionInfo, image_dims)
    return get_affected_dims(reg, nothing, image_dims)
end

# The nuclear norm is homogeneous of degree 1, so λ scales linearly (see scale_regularization docstring).
function scale_regularization(reg::LocallyLowRank, factor::Real)
    return LocallyLowRank(reg.λ * factor; block_size = reg.block_size, time_dim = reg.time_dim)
end

function _llr_block_size(reg::LocallyLowRank, spatial_size::NTuple{N, Int}) where {N}
    block_size = reg.block_size isa Integer ? ntuple(_ -> Int(reg.block_size), Val(N)) : Int.(reg.block_size)
    @argcheck length(block_size) == N "block_size must have one entry per spatial dimension ($N), got $(length(block_size))"
    @argcheck all(block_size .<= spatial_size) "block_size $(block_size) exceeds the spatial image size $(spatial_size)"
    return NTuple{N, Int}(block_size)
end

function materialize(reg::LocallyLowRank, x::Variable{T}; threaded::Bool) where {T}
    x_val = ~x
    dims = x_val isa NamedDimsArray ? dimnames(x_val) : (1:ndims(x_val))
    time_dim = get_time_dim(reg.time_dim, dims)
    @argcheck time_dim > 1 "LocallyLowRank needs at least one spatial dimension before the temporal one"
    spatial_size = NTuple{time_dim - 1, Int}(size(x_val)[1:(time_dim - 1)])
    block_size = _llr_block_size(reg, spatial_size)
    num_batch = prod(size(x_val)[(time_dim + 1):end]; init = 1)
    f = BlockNuclearNorm(
        real(T)(reg.λ), block_size, spatial_size, size(x_val, time_dim), num_batch, threaded
    )
    op = get_operator(reg, x_val; threaded)
    repr = @sprintf "%g ⋅ ∑_b ‖𝓧_b(%s)‖_*" real(T)(reg.λ) get_name(x)
    return StructuredOptimization.Term(1, f, op * x, repr)
end
