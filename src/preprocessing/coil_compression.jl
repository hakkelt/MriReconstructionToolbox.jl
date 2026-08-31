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

Geometric coil compression (Zhang et al. 2013): aligns the per-readout virtual-coil subspaces
across `x` so the compressed channels vary smoothly.

!!! warning
    Not implemented yet — `compress_coils` throws for this method. Use [`SVDCompression`](@ref).
"""
struct GeometricCompression <: CoilCompressionMethod end

"""
    compress_coils(acq::AcquisitionInfo, n_virtual::Int; method = SVDCompression(), coil_dim = nothing)
    compress_coils(data::AbstractArray, n_virtual::Int; method = SVDCompression(), coil_dim = nothing)

Compresses multi-coil k-space data (and sensitivity maps if present) to `n_virtual` channels.
Returns a tuple `(compressed_acq, compression_matrix)` or `(compressed_data, compression_matrix)`
where `compression_matrix` has size `(n_virtual, n_coils)`.
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

    method isa SVDCompression ||
        throw(ArgumentError("compress_coils is only implemented for SVDCompression(); got $(typeof(method))"))

    Nc = size(data, c_idx)
    @argcheck 1 <= n_virtual <= Nc "n_virtual ($n_virtual) must be between 1 and coil count ($Nc)"

    perm = ntuple(i -> i == 1 ? c_idx : (i <= c_idx ? i - 1 : i), ndims(data))
    perm_data = permutedims(unname(data), perm)
    flat_data = reshape(perm_data, Nc, :)

    F = svd(flat_data)
    C = Matrix(F.U[:, 1:n_virtual]') # (n_virtual, Nc)

    return compress_coils_with_matrix(data, C; coil_dim)
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
