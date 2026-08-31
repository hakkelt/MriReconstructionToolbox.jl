"""
    CoilCompressionMethod

Abstract type representing receiver coil compression algorithms.
"""
abstract type CoilCompressionMethod end

"""
    SVDCompression <: CoilCompressionMethod

Principal component / SVD coil compression (Buehrer et al. 2007, Huang et al. 2008).
Transforms multi-coil arrays to a reduced number of virtual channels spanning the principal signal subspace.
"""
struct SVDCompression <: CoilCompressionMethod end

"""
    GeometricCompression <: CoilCompressionMethod

Geometric coil compression (Zhang et al. 2013): computes SVD compression along the readout axis
and aligns virtual coil bases across `x` via Procrustes rotation so the compressed channels vary smoothly.
"""
struct GeometricCompression <: CoilCompressionMethod end

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
        method::CoilCompressionMethod = SVDCompression(),
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
        method::CoilCompressionMethod = SVDCompression(),
        coil_dim = nothing,
    )
    c_idx = if !isnothing(coil_dim)
        coil_dim isa Symbol ? findfirst(==(coil_dim), dimnames(data)) : coil_dim
    elseif data isa NamedDimsArray && :coil ∈ dimnames(data)
        findfirst(==(:coil), dimnames(data))
    else
        ndims(data) >= 4 ? 4 : 3
    end
    @argcheck !isnothing(c_idx) && 1 <= c_idx <= ndims(data) "Invalid coil dimension"

    Nc = size(data, c_idx)
    @argcheck 1 <= n_virtual <= Nc "n_virtual ($n_virtual) must be between 1 and coil count ($Nc)"

    if method isa SVDCompression
        perm = ntuple(i -> i == 1 ? c_idx : (i <= c_idx ? i - 1 : i), ndims(data))
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
        perm_slice = ntuple(i -> i == 1 ? slice_c_idx : (i <= slice_c_idx ? i - 1 : i), slice_ndims)

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

        return compress_coils_with_matrix(data, C_3d; coil_dim)
    else
        throw(ArgumentError("Unknown coil compression method: $(typeof(method))"))
    end
end

function compress_coils_with_matrix(data::AbstractArray, C::AbstractMatrix; coil_dim = nothing)
    c_idx = if !isnothing(coil_dim)
        coil_dim isa Symbol ? findfirst(==(coil_dim), dimnames(data)) : coil_dim
    elseif data isa NamedDimsArray && :coil ∈ dimnames(data)
        findfirst(==(:coil), dimnames(data))
    else
        ndims(data) >= 4 ? 4 : 3
    end
    @argcheck !isnothing(c_idx) && 1 <= c_idx <= ndims(data) "Invalid coil dimension"

    Nc = size(data, c_idx)
    n_virtual = size(C, 1)
    @argcheck size(C, 2) == Nc "Compression matrix columns ($(size(C, 2))) must match coil count ($Nc)"

    perm = ntuple(i -> i == 1 ? c_idx : (i <= c_idx ? i - 1 : i), ndims(data))
    inv_perm = ntuple(i -> i == c_idx ? 1 : (i < c_idx ? i + 1 : i), ndims(data))

    perm_data = permutedims(unname(data), perm)
    flat_data = reshape(perm_data, Nc, :)
    comp_flat = C * flat_data

    out_perm_size = ntuple(i -> i == 1 ? n_virtual : size(perm_data, i), ndims(data))
    comp_perm = reshape(comp_flat, out_perm_size)
    comp_data = permutedims(comp_perm, inv_perm)

    if data isa NamedDimsArray
        return NamedDimsArray{dimnames(data)}(comp_data), C
    else
        return comp_data, C
    end
end

function compress_coils_with_matrix(data::AbstractArray, C::AbstractArray; coil_dim = nothing)
    @argcheck ndims(C) == 3 "Slice-wise compression matrix must have 3 dimensions (n_virtual, n_coils, n_readout)"
    c_idx = if !isnothing(coil_dim)
        coil_dim isa Symbol ? findfirst(==(coil_dim), dimnames(data)) : coil_dim
    elseif data isa NamedDimsArray && :coil ∈ dimnames(data)
        findfirst(==(:coil), dimnames(data))
    else
        ndims(data) >= 4 ? 4 : 3
    end
    @argcheck !isnothing(c_idx) && 1 <= c_idx <= ndims(data) "Invalid coil dimension"

    Nc = size(data, c_idx)
    n_virtual = size(C, 1)
    Nx = size(data, 1)
    @argcheck size(C, 2) == Nc "Compression matrix columns ($(size(C, 2))) must match coil count ($Nc)"
    @argcheck size(C, 3) == Nx "Compression matrix readout dimension ($(size(C, 3))) must match data readout dimension ($Nx)"

    raw = unname(data)
    is_image_space = data isa NamedDimsArray && (:x ∈ dimnames(data) || :kx ∉ dimnames(data))

    hybrid = if is_image_space
        copy(raw)
    else
        ifft(ifftshift(raw, 1), 1)
    end

    slice_ndims = ndims(raw) - 1
    slice_c_idx = c_idx - 1
    perm_slice = ntuple(i -> i == 1 ? slice_c_idx : (i <= slice_c_idx ? i - 1 : i), slice_ndims)
    inv_perm_slice = ntuple(i -> i == slice_c_idx ? 1 : (i < slice_c_idx ? i + 1 : i), slice_ndims)

    out_size = ntuple(i -> i == c_idx ? n_virtual : size(raw, i), ndims(raw))
    comp_hybrid = zeros(eltype(raw), out_size)

    for ix in 1:Nx
        slice_data = selectdim(hybrid, 1, ix)
        flat_slice = reshape(permutedims(slice_data, perm_slice), Nc, :)
        comp_flat = C[:, :, ix] * flat_slice

        slice_sz = size(slice_data)
        perm_sz = ntuple(i -> i == 1 ? n_virtual : slice_sz[perm_slice[i]], slice_ndims)
        comp_perm = reshape(comp_flat, perm_sz)
        comp_slice = permutedims(comp_perm, inv_perm_slice)
        selectdim(comp_hybrid, 1, ix) .= comp_slice
    end

    comp_data = if is_image_space
        comp_hybrid
    else
        fftshift(fft(comp_hybrid, 1), 1)
    end

    if data isa NamedDimsArray
        return NamedDimsArray{dimnames(data)}(comp_data), C
    else
        return comp_data, C
    end
end
