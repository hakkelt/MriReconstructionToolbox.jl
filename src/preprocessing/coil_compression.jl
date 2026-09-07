"""
    CoilCompression

Abstract type representing receiver coil compression algorithms.
"""
abstract type CoilCompression end

"""
    _resolve_coil_dim(data, coil_dim) -> Int

Resolve the coil axis of `data`: an explicit `coil_dim` (a `Symbol` name or integer index), else
the `:coil` dimension of a `NamedDimsArray`, else the trailing-dims convention (4 if `ndims ≥ 4`,
otherwise 3). Throws if the result is out of range.
"""
function _resolve_coil_dim(data::AbstractArray, coil_dim; fallback::Int = ndims(data) >= 4 ? 4 : 3)
    c_idx = if !isnothing(coil_dim)
        coil_dim isa Symbol ? findfirst(==(coil_dim), dimnames(data)) : coil_dim
    elseif data isa NamedDimsArray && :coil ∈ dimnames(data)
        findfirst(==(:coil), dimnames(data))
    else
        fallback
    end
    @argcheck !isnothing(c_idx) && 1 <= c_idx <= ndims(data) "Invalid coil dimension"
    return c_idx
end

"""
    _front_perm(c_idx, n), _front_inv_perm(c_idx, n)

Permutation (and its inverse) that moves dimension `c_idx` of an `n`-dimensional array to the
front, leaving the relative order of the remaining dimensions unchanged.
"""
_front_perm(c_idx::Int, n::Int) = ntuple(i -> i == 1 ? c_idx : (i <= c_idx ? i - 1 : i), n)
_front_inv_perm(c_idx::Int, n::Int) = ntuple(i -> i == c_idx ? 1 : (i < c_idx ? i + 1 : i), n)

"""
    _trailing_perm(c_idx, n), _trailing_inv_perm(c_idx, n)

Permutation (and its inverse) that moves dimension `c_idx` of an `n`-dimensional array to the
trailing position, leaving the relative order of the remaining dimensions unchanged.
"""
_trailing_perm(c_idx::Int, n::Int) = ntuple(i -> i == n ? c_idx : (i >= c_idx ? i + 1 : i), n)
_trailing_inv_perm(c_idx::Int, n::Int) = ntuple(i -> i == c_idx ? n : (i >= c_idx ? i - 1 : i), n)

_rewrap_like(ref::AbstractArray, data::AbstractArray) =
    ref isa NamedDimsArray ? NamedDimsArray{dimnames(ref)}(data) : data

"""
    _apply_slicewise_compression(hybrid, C, c_idx) -> Array

Apply a per-readout-slice compression matrix `C` of size `(n_virtual, Nc, Nx)` to `hybrid`
(readout on dim 1, coils on `c_idx`), returning the compressed hybrid-space array. Shared by
`GeometricCompression` calibration and `compress_coils_with_matrix(::AbstractArray)`.
"""
function _apply_slicewise_compression(hybrid::AbstractArray, C::AbstractArray, c_idx::Int)
    Nc = size(hybrid, c_idx)
    n_virtual = size(C, 1)
    Nx = size(hybrid, 1)

    slice_ndims = ndims(hybrid) - 1
    slice_c_idx = c_idx - 1
    perm_slice = _front_perm(slice_c_idx, slice_ndims)
    inv_perm_slice = _front_inv_perm(slice_c_idx, slice_ndims)

    out_size = ntuple(i -> i == c_idx ? n_virtual : size(hybrid, i), ndims(hybrid))
    out = zeros(eltype(hybrid), out_size)

    for ix in 1:Nx
        slice_data = selectdim(hybrid, 1, ix)
        flat_slice = reshape(permutedims(slice_data, perm_slice), Nc, :)
        comp_flat = C[:, :, ix] * flat_slice

        slice_sz = size(slice_data)
        perm_sz = ntuple(i -> i == 1 ? n_virtual : slice_sz[perm_slice[i]], slice_ndims)
        selectdim(out, 1, ix) .= permutedims(reshape(comp_flat, perm_sz), inv_perm_slice)
    end
    return out
end

"""
    SVDCompression <: CoilCompression

Principal component / SVD coil compression (Buehrer et al. 2007, Huang et al. 2008).
Transforms multi-coil arrays to a reduced number of virtual channels spanning the principal signal subspace.
"""
struct SVDCompression <: CoilCompression end

"""
    GeometricCompression <: CoilCompression

Geometric coil compression (Zhang et al. 2013): computes SVD compression along the readout axis
and aligns virtual coil bases across `x` via Procrustes rotation so the compressed channels vary smoothly.
"""
struct GeometricCompression <: CoilCompression end

"""
    compress_coils(acq::AcquisitionInfo, n_virtual::Int; method = SVDCompression(), coil_dim = nothing)
    compress_coils(data::AbstractArray, n_virtual::Int; method = SVDCompression(), coil_dim = nothing)

Compresses multi-coil k-space data (and sensitivity maps if present) to `n_virtual` channels.
Returns a tuple `(compressed_acq, compression_matrix)` or `(compressed_data, compression_matrix)`.
For `SVDCompression`, `compression_matrix` has size `(n_virtual, n_coils)`.
For `GeometricCompression`, `compression_matrix` has size `(n_virtual, n_coils, Nx)`.
"""
function compress_coils(
        acq::AcquisitionInfo,
        n_virtual::Int;
        method::CoilCompression = SVDCompression(),
        coil_dim = nothing,
    )
    compressed_ksp, C = compress_coils(acq.kspace_data, n_virtual; method, coil_dim)
    compressed_sens = if !isnothing(acq.sensitivity_maps)
        first(compress_coils_with_matrix(acq.sensitivity_maps, C; coil_dim))
    else
        nothing
    end
    return AcquisitionInfo(acq; kspace_data = compressed_ksp, sensitivity_maps = compressed_sens), C
end

function compress_coils(
        data::AbstractArray,
        n_virtual::Int;
        method::CoilCompression = SVDCompression(),
        coil_dim = nothing,
    )
    c_idx = _resolve_coil_dim(data, coil_dim)

    Nc = size(data, c_idx)
    @argcheck 1 <= n_virtual <= Nc "n_virtual ($n_virtual) must be between 1 and coil count ($Nc)"

    if method isa SVDCompression
        perm = _front_perm(c_idx, ndims(data))
        perm_data = permutedims(unname(data), perm)
        flat_data = reshape(perm_data, Nc, :)

        F = svd(flat_data)
        C = Matrix(F.U[:, 1:n_virtual]') # (n_virtual, Nc)

        return compress_coils_with_matrix(data, C; coil_dim)
    elseif method isa GeometricCompression
        raw = unname(data)
        Nx = size(raw, 1)
        hybrid = ifft(ifftshift(raw, 1), 1)

        C_3d = zeros(eltype(raw), n_virtual, Nc, Nx)
        V_prev = nothing

        slice_ndims = ndims(raw) - 1
        slice_c_idx = c_idx - 1
        perm_slice = _front_perm(slice_c_idx, slice_ndims)

        for ix in 1:Nx
            slice_data = selectdim(hybrid, 1, ix)
            flat_slice = reshape(permutedims(slice_data, perm_slice), Nc, :)

            F = svd(flat_slice)
            V_cur = Matrix(F.U[:, 1:n_virtual])

            if !isnothing(V_prev)
                P = svd(V_prev' * V_cur)
                R = P.U * P.Vt
                V_cur = V_cur * R'
            end
            V_prev = V_cur
            C_3d[:, :, ix] = Matrix(V_cur')
        end

        # Apply here, reusing the k-space `hybrid` we already transformed, instead of letting
        # `compress_coils_with_matrix` recompute the same `ifft(ifftshift(...))`.
        comp_data = fftshift(fft(_apply_slicewise_compression(hybrid, C_3d, c_idx), 1), 1)
        return _rewrap_like(data, comp_data), C_3d
    else
        throw(ArgumentError("Unknown coil compression method: $(typeof(method))"))
    end
end

function compress_coils_with_matrix(data::AbstractArray, C::AbstractMatrix; coil_dim = nothing)
    c_idx = _resolve_coil_dim(data, coil_dim)

    Nc = size(data, c_idx)
    n_virtual = size(C, 1)
    @argcheck size(C, 2) == Nc "Compression matrix columns ($(size(C, 2))) must match coil count ($Nc)"

    perm = _front_perm(c_idx, ndims(data))
    inv_perm = _front_inv_perm(c_idx, ndims(data))

    perm_data = permutedims(unname(data), perm)
    flat_data = reshape(perm_data, Nc, :)
    comp_flat = C * flat_data

    out_perm_size = ntuple(i -> i == 1 ? n_virtual : size(perm_data, i), ndims(data))
    comp_perm = reshape(comp_flat, out_perm_size)
    comp_data = permutedims(comp_perm, inv_perm)

    return _rewrap_like(data, comp_data), C
end

function compress_coils_with_matrix(data::AbstractArray, C::AbstractArray; coil_dim = nothing)
    @argcheck ndims(C) == 3 "Slice-wise compression matrix must have 3 dimensions (n_virtual, n_coils, n_readout)"
    c_idx = _resolve_coil_dim(data, coil_dim)

    Nc = size(data, c_idx)
    n_virtual = size(C, 1)
    Nx = size(data, 1)
    @argcheck size(C, 2) == Nc "Compression matrix columns ($(size(C, 2))) must match coil count ($Nc)"
    @argcheck size(C, 3) == Nx "Compression matrix readout dimension ($(size(C, 3))) must match data readout dimension ($Nx)"

    raw = unname(data)
    is_image_space = data isa NamedDimsArray && (:x ∈ dimnames(data) || :kx ∉ dimnames(data))

    hybrid = is_image_space ? raw : ifft(ifftshift(raw, 1), 1)
    comp_hybrid = _apply_slicewise_compression(hybrid, C, c_idx)
    comp_data = is_image_space ? comp_hybrid : fftshift(fft(comp_hybrid, 1), 1)

    return _rewrap_like(data, comp_data), C
end
