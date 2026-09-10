"""
    PartitionedKSpace(parts; ragged_dim, dimnames = nothing)

k-space data whose frames select **different numbers of samples**, held as one array per frame
instead of as a single dense array.

A dense `kspace_data` array can only hold a per-frame sampling pattern when every frame selects
the same number of samples, because the sample axis has one length for the whole array. When the
per-frame masks have different counts — a genuinely variable-density dynamic acquisition, rather
than a shifted mask of fixed size — the data no longer fits in a rectangle, and this container
holds it instead: `parts[t]` is frame `t`'s own k-space, with its own sample count.

Every part must have the same element type, the same number of dimensions, and the same size in
every dimension **except one**, the *ragged* dimension. The partition itself is the last k-space
dimension (`:time`, in the usual dynamic layout).

This is deliberately **not** an `AbstractArray`: there is no honest `size` for it, and giving it
one would let code that assumes a rectangle keep running and return a plausible wrong answer.
Code that must handle both layouts asks with [`is_partitioned`](@ref), `ragged_dim`, `nparts` and
`parts`; everything else gets a `MethodError` it can act on.

`reconstruct` accepts a partitioned acquisition and still returns an ordinary dense `Array` — the
*image* size is the same for every frame, so only the measurement side is ragged.

See also [`CartesianAcquisitionInfo`](@ref).
"""
struct PartitionedKSpace{T, N, A <: AbstractArray{T}, D}
    parts::Vector{A}
    ragged_dim::Int
    dimnames::D

    function PartitionedKSpace(parts::Vector{A}, ragged_dim::Int, names::D) where {T, A <: AbstractArray{T}, D}
        @argcheck !isempty(parts) "PartitionedKSpace needs at least one part"
        M = ndims(first(parts))
        @argcheck all(p -> ndims(p) == M, parts) "every part must have the same number of dimensions"
        @argcheck 1 <= ragged_dim <= M "ragged_dim ($ragged_dim) out of range for $(M)-dimensional parts"
        ref = size(first(parts))
        for p in parts
            for d in 1:M
                d == ragged_dim && continue
                @argcheck size(p, d) == ref[d] "parts may differ only along the ragged dimension ($ragged_dim); got sizes $(size(p)) and $ref"
            end
        end
        if !isnothing(names)
            @argcheck length(names) == M + 1 "dimnames must name the $(M) part dimensions plus the partition dimension"
        end
        return new{T, M + 1, A, D}(parts, ragged_dim, names)
    end
end

function PartitionedKSpace(parts::AbstractVector; ragged_dim::Int, dimnames = nothing)
    parts = collect(parts)
    if !isnothing(dimnames)
        # Each part carries the part-dimension names, so a task-splitting slice of a partitioned
        # acquisition is an ordinary `NamedDimsArray` and every downstream path sees exactly what
        # it would have seen from a dense acquisition.
        part_names = Tuple(dimnames[1:(end - 1)])
        parts = map(part -> part isa NamedDimsArray ? part : NamedDimsArray{part_names}(part), parts)
        for part in parts
            @argcheck NamedDims.dimnames(part) == part_names "part dimnames $(NamedDims.dimnames(part)) do not match the leading dimnames $(part_names)"
        end
    end
    return PartitionedKSpace(parts, ragged_dim, dimnames)
end
PartitionedKSpace(parts::Tuple; kwargs...) = PartitionedKSpace(collect(parts); kwargs...)
# The forward model's output is an `ArrayPartition`; this is how a simulated measurement becomes
# the `kspace_data` of an acquisition again.
PartitionedKSpace(parts::ArrayPartition; kwargs...) = PartitionedKSpace(collect(parts.x); kwargs...)

"""
    is_partitioned(x) -> Bool

Whether `x` is k-space held one frame at a time ([`PartitionedKSpace`](@ref)) rather than as a
single dense array.
"""
is_partitioned(::PartitionedKSpace) = true
is_partitioned(::Any) = false

parts(p::PartitionedKSpace) = p.parts
nparts(p::PartitionedKSpace) = length(p.parts)
ragged_dim(p::PartitionedKSpace) = p.ragged_dim

Base.eltype(::PartitionedKSpace{T}) where {T} = T
Base.ndims(::PartitionedKSpace{T, N}) where {T, N} = N
Base.length(p::PartitionedKSpace) = sum(length, p.parts)
Base.similar(p::PartitionedKSpace) = PartitionedKSpace(map(similar, p.parts), p.ragged_dim, p.dimnames)

NamedDims.dimnames(p::PartitionedKSpace) =
    isnothing(p.dimnames) ?
    throw(ArgumentError("this PartitionedKSpace carries no dimension names")) : p.dimnames
NamedDims.dimnames(p::PartitionedKSpace, d::Integer) = dimnames(p)[d]

# Whether k-space carries dimension names, in either layout. The checks that used to ask
# `ksp isa NamedDimsArray` mean this: a `PartitionedKSpace` can be named too, and skipping its
# name checks would let a mislabelled acquisition through.
_has_dimnames(ksp) = ksp isa NamedDimsArray
_has_dimnames(p::PartitionedKSpace) = !isnothing(p.dimnames)

# `size(ksp)[start:end]`, but asking one dimension at a time so it also works on a partitioned
# k-space. The trailing dimensions of an acquisition — coil, slice, frame — are never the ragged
# one, which is why this is well defined for both layouts.
_ksp_trailing_size(ksp, start::Int) =
    ntuple(i -> size(ksp, start + i - 1), max(ndims(ksp) - start + 1, 0))

# The k-space layout as task splitting records it: sizes where there are sizes, a `Colon` where the
# axis is ragged. Task splitting only ever indexes the batch dimensions, which are never ragged.
_ksp_plan_size(ksp) = size(ksp)
_ksp_plan_size(p::PartitionedKSpace) = _partition_shape(p)

# The task-splitting slices of a k-space, in the order `CartesianIndices` over the batch dimensions
# visits them (first dimension fastest). For a partitioned k-space the partition dimension *is* one
# of the batch dimensions — that is what makes the unequal-count case decompose at all — so each
# slice is an ordinary dense array again.
_ksp_eachslice(ksp, dims::Tuple) = eachslice(ksp; dims)
function _ksp_eachslice(p::PartitionedKSpace, dims::Tuple)
    N = ndims(p)
    @argcheck N ∈ dims "a partitioned k-space can only be split over its partition dimension ($N); task splitting asked for $dims"
    inner = filter(!=(N), dims)
    isempty(inner) && return parts(p)
    return [slice for part in parts(p) for slice in eachslice(part; dims = inner)]
end

# `size` along any dimension but the ragged one is well defined and is what the encoding and
# task-splitting code actually asks for (coil count, frame count, the non-ragged sample axes).
# Along the ragged dimension there is no single answer, so asking is a bug in the caller.
function Base.size(p::PartitionedKSpace{T, N}, d::Integer) where {T, N}
    d == N && return length(p.parts)
    d > N && return 1
    d == p.ragged_dim && throw(
        ArgumentError(
            "dimension $d of this k-space is ragged (per-frame sample counts $(map(q -> size(q, p.ragged_dim), p.parts))), so it has no single size"
        )
    )
    return size(first(p.parts), d)
end

Base.size(p::PartitionedKSpace) = throw(
    ArgumentError(
        "PartitionedKSpace has no dense size: dimension $(p.ragged_dim) is ragged. Ask for `size(ksp, d)` of a non-ragged dimension, or use `parts(ksp)`."
    )
)

# The sizes that *are* shared, with the ragged axis left out — the honest summary of the layout.
function _partition_shape(p::PartitionedKSpace{T, N}) where {T, N}
    return ntuple(d -> d == p.ragged_dim ? (:) : size(p, d), N)
end

"""
    to_array_partition(p::PartitionedKSpace) -> ArrayPartition

The solver-facing storage: the same data as a `RecursiveArrayTools.ArrayPartition`, which is what
the `VCAT` of per-frame `GetIndex` operators produces and consumes.
"""
to_array_partition(p::PartitionedKSpace) = ArrayPartition(map(unname, p.parts)...)

# The measurement in the form the operators and the solver take it: an ordinary array when k-space
# is dense, the `ArrayPartition` the `VCAT` of per-frame `GetIndex`es produces when it is not.
_measurement(ksp) = ksp
_measurement(p::PartitionedKSpace) = to_array_partition(p)

"""
    _reject_partitioned(ksp, what)

Throw a clear error when `what` is handed k-space held one frame at a time.

Most preprocessing and analysis paths index `kspace_data` as a rectangle. A silent wrong answer is
the failure mode to avoid, so anything that has not been made partition-aware says so outright
instead of producing a plausible-looking image from the wrong samples.
"""
function _reject_partitioned(ksp, what::AbstractString)
    ksp isa PartitionedKSpace && throw(
        ArgumentError(
            "$what is not supported for partitioned k-space (frames selecting different numbers of samples). Reconstruct the frames whose sample counts agree as one dense acquisition, or apply $what per frame via `parts(kspace_data)`."
        )
    )
    return nothing
end

# `ksp ./ scale`, in either layout.
_scale_kspace(ksp, scale) = ksp ./ scale
_scale_kspace(p::PartitionedKSpace, scale) =
    PartitionedKSpace(map(part -> part ./ scale, p.parts), p.ragged_dim, p.dimnames)

function Base.show(io::IO, p::PartitionedKSpace{T, N}) where {T, N}
    counts = join(map(q -> size(q, p.ragged_dim), p.parts), ",")
    shape = map(s -> s === (:) ? "{$counts}" : string(s), _partition_shape(p))
    labels = isnothing(p.dimnames) ? shape : ["$n: $s" for (n, s) in zip(p.dimnames, shape)]
    return print(io, "PartitionedKSpace{$T}<", join(labels, ", "), ">")
end

NamedDims.unname(p::PartitionedKSpace) =
    isnothing(p.dimnames) ? p : PartitionedKSpace(map(unname, p.parts), p.ragged_dim, nothing)
