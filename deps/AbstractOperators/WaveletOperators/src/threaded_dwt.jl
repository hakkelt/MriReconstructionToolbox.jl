# Threaded 2-D and 3-D filter-bank DWT.
#
# The same algorithm as Wavelets.jl's `_dwt!` for an `OrthoFilter` (`Transforms/transforms_filter.jl`),
# and built from its per-line kernels, so the result is identical to `dwt!`/`idwt!` bit for bit.
# Each level transforms every line along one dimension, then along the next; the lines of one
# such stage are disjoint, so a stage is split across threads, each with its own scratch
# vectors, and the stages run one after another.

const _TF = Wavelets.Transforms

# Split lines `1:nlines` into one contiguous block per thread and call
# `f(first_line, count, si, tmp)` on each, with scratch private to the block.
function _foreach_line_block(f::F, nlines::Int, ::Type{T}, filtlen::Int, buflen::Int) where {F, T}
    nblocks = min(Threads.nthreads(), nlines)
    @budgeted_threads for b in 1:nblocks
        lo = div((b - 1) * nlines, nblocks) + 1
        hi = div(b * nlines, nblocks)
        si = Vector{T}(undef, filtlen - 1)
        tmp = Vector{T}(undef, buflen)
        f(lo, hi - lo + 1, si, tmp)
    end
    return nothing
end

function _check_dwt_args(y, x, L)
    size(x) == size(y) || throw(DimensionMismatch("in and out array size must match"))
    0 <= L || throw(ArgumentError("L must be positive"))
    _TF.sufficientpoweroftwo(y, L) || throw(ArgumentError("size must have a sufficient power of 2 factor"))
    y === x && throw(ArgumentError("in array is out array"))
    return nothing
end

function _threaded_dwt!(
        y::Array{<:Number, 2}, x::Array{<:Number, 2}, filter::Wavelets.WT.OrthoFilter, L::Integer, fw::Bool
    )
    _check_dwt_args(y, x, L)
    L == 0 && return copyto!(y, x)
    T = promote_type(eltype(x), eltype(y))
    scfilter, dcfilter = Wavelets.WT.makereverseqmfpair(filter, fw, T)
    m, n = size(x)
    flen = length(filter)
    buflen = max(n << 1, m)
    msub, nsub = fw ? (m, n) : (div(m, 2^(L - 1)), div(n, 2^(L - 1)))
    fw || copyto!(y, x)
    input = x
    for l in (fw ? (1:L) : (L:-1:1))
        rows! = (src) -> _foreach_line_block(msub, T, flen, buflen) do lo, cnt, si, tmp
            _TF.dwt_transform_strided!(
                y, src, cnt, nsub, m, i -> _TF.row_idx(i + lo - 1, m),
                _TF.unsafe_vectorslice(tmp, 1, nsub), _TF.unsafe_vectorslice(tmp, nsub + 1, nsub),
                filter, fw, dcfilter, scfilter, si,
            )
        end
        cols! = (src) -> _foreach_line_block(nsub, T, flen, buflen) do lo, cnt, si, tmp
            _TF.dwt_transform_cols!(
                y, src, msub, cnt, i -> _TF.col_idx(i + lo - 1, m),
                _TF.unsafe_vectorslice(tmp, 1, msub), filter, fw, dcfilter, scfilter, si,
            )
        end
        if fw
            rows!(input)
            input = y
            cols!(y)
        else
            cols!(input)
            input = y
            rows!(y)
        end
        msub = fw ? msub >> 1 : msub << 1
        nsub = fw ? nsub >> 1 : nsub << 1
    end
    return y
end

function _threaded_dwt!(
        y::Array{<:Number, 3}, x::Array{<:Number, 3}, filter::Wavelets.WT.OrthoFilter, L::Integer, fw::Bool
    )
    _check_dwt_args(y, x, L)
    L == 0 && return copyto!(y, x)
    T = promote_type(eltype(x), eltype(y))
    scfilter, dcfilter = Wavelets.WT.makereverseqmfpair(filter, fw, T)
    m, n, d = size(x)
    flen = length(filter)
    buflen = max(m, n << 1, d << 1)
    msub, nsub, dsub = fw ? (m, n, d) : (div(m, 2^(L - 1)), div(n, 2^(L - 1)), div(d, 2^(L - 1)))
    fw || copyto!(y, x)
    input = x
    for l in (fw ? (1:L) : (L:-1:1))
        planes! = (src) -> _foreach_line_block(nsub, T, flen, buflen) do lo, cnt, si, tmp
            for j in lo:(lo + cnt - 1)
                _TF.dwt_transform_strided!(
                    y, src, msub, dsub, m * n, i -> _TF.plane_idx(i, j, m),
                    _TF.unsafe_vectorslice(tmp, 1, dsub), _TF.unsafe_vectorslice(tmp, dsub + 1, dsub),
                    filter, fw, dcfilter, scfilter, si,
                )
            end
        end
        rows! = () -> _foreach_line_block(dsub, T, flen, buflen) do lo, cnt, si, tmp
            for j in lo:(lo + cnt - 1)
                _TF.dwt_transform_strided!(
                    y, y, msub, nsub, m, i -> _TF.row_idx(i, j, m, n),
                    _TF.unsafe_vectorslice(tmp, 1, nsub), _TF.unsafe_vectorslice(tmp, nsub + 1, nsub),
                    filter, fw, dcfilter, scfilter, si,
                )
            end
        end
        cols! = (src) -> _foreach_line_block(dsub, T, flen, buflen) do lo, cnt, si, tmp
            for j in lo:(lo + cnt - 1)
                _TF.dwt_transform_cols!(
                    y, src, msub, nsub, i -> _TF.col_idx(i, j, m, n),
                    _TF.unsafe_vectorslice(tmp, 1, msub), filter, fw, dcfilter, scfilter, si,
                )
            end
        end
        if fw
            planes!(input)
            input = y
            rows!()
            cols!(y)
        else
            cols!(input)
            input = y
            rows!()
            planes!(y)
        end
        msub = fw ? msub >> 1 : msub << 1
        nsub = fw ? nsub >> 1 : nsub << 1
        dsub = fw ? dsub >> 1 : dsub << 1
    end
    return y
end
