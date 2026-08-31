"""
    TemporalBasis(Φ::AbstractMatrix; time_dim = nothing)

Signal model representing dynamic image series expanded in a temporal subspace:
``x(r, t) = \\sum_{k=1}^K \\Phi(t, k) c(r, k)``.

# Arguments
- `Φ`: ``N_t \\times K`` basis matrix where ``N_t`` is the number of time frames and ``K`` is the number of subspace coefficients.
- `time_dim`: (optional) Dimension index or name for the temporal dimension (defaults to `:time` or last image dimension).
"""
struct TemporalBasis{T, M <: AbstractMatrix{T}, D}
    Φ::M
    time_dim::D
    function TemporalBasis(Φ::M; time_dim::D = nothing) where {T, M <: AbstractMatrix{T}, D}
        _check_dim_spec(time_dim, "time_dim")
        return new{T, M, D}(Φ, time_dim)
    end
end

"""
    signal_model_operator(method::AbstractReconstructionMethod, acq::AcquisitionInfo; threaded::Bool)

Constructs the signal model linear operator `ℳ` for the given reconstruction method and acquisition data.
Returns `nothing` if no signal model is specified.
"""
signal_model_operator(::AbstractReconstructionMethod, ::AcquisitionInfo; threaded::Bool = true) = nothing

function signal_model_operator(method::IterativeReconstruction, acq::AcquisitionInfo; threaded::Bool = true)
    return signal_model_operator(method.signal_model, acq; threaded)
end

signal_model_operator(::Nothing, ::AcquisitionInfo; threaded::Bool = true) = nothing

function signal_model_operator(model::TemporalBasis, acq::AcquisitionInfo; threaded::Bool = true)
    img_dims = get_image_dims(acq)
    img_size = get_image_size(acq)
    time_dim_idx = get_time_dim(model.time_dim, img_dims)
    Nt = size(model.Φ, 1)
    K = size(model.Φ, 2)
    @argcheck img_size[time_dim_idx] == Nt "TemporalBasis time frame count ($Nt) does not match acquisition time dimension size ($(img_size[time_dim_idx]))"

    coeff_size = ntuple(i -> i == time_dim_idx ? K : img_size[i], length(img_size))
    T = complex(eltype(model.Φ))

    # Construct matrix multiplication operator along time_dim
    # For trailing time_dim, flat spatial dimension is prod(img_size[1:end-1])
    if time_dim_idx == length(img_size)
        N_spatial = prod(img_size[1:(end - 1)])
        R_in = Reshape(Eye(T, (N_spatial, K)), coeff_size...)
        L = LMatrixOp(T, (N_spatial, K), Matrix(transpose(model.Φ)); threaded)
        R_out = Reshape(Eye(T, (N_spatial, Nt)), img_size...)
        ℳ = R_out * (L * R_in')
    else
        # General case via PermuteDims to trailing axis, LMatrixOp, and permute back.
        # `perm` moves the time/coeff axis to the last position (output dim time_dim_idx..N-1
        # come from input time_dim_idx+1..N, output dim N from input time_dim_idx); `inv_perm`
        # is its inverse and restores the original axis order.
        N = length(img_size)
        perm = ntuple(i -> i < time_dim_idx ? i : (i == N ? time_dim_idx : i + 1), N)
        inv_perm = ntuple(i -> i == time_dim_idx ? N : (i >= time_dim_idx ? i - 1 : i), N)
        perm_img_size = ntuple(i -> img_size[perm[i]], length(img_size))
        perm_coeff_size = ntuple(i -> coeff_size[perm[i]], length(coeff_size))

        N_spatial = prod(perm_img_size[1:(end - 1)])
        P_in = PermuteDims(T, coeff_size, perm)
        R_in = Reshape(Eye(T, (N_spatial, K)), perm_coeff_size...)
        L = LMatrixOp(T, (N_spatial, K), Matrix(transpose(model.Φ)); threaded)
        R_out = Reshape(Eye(T, (N_spatial, Nt)), perm_img_size...)
        P_out = PermuteDims(T, perm_img_size, inv_perm)
        ℳ = P_out * R_out * L * R_in' * P_in
    end

    if acq.kspace_data isa NamedDimsArray
        in_dimnames = ntuple(i -> i == time_dim_idx ? :coeff : img_dims[i], length(img_dims))
        out_dimnames = img_dims
        ℳ = NamedDimsOp{in_dimnames, out_dimnames}(ℳ)
    end
    return ℳ
end

# Dimension queries for signal models and methods

get_affected_dims(::Nothing, ::Any, ::Any) = ()
get_affected_dims(model::TemporalBasis, ::Any, image_dims) = (image_dims[get_time_dim(model.time_dim, image_dims)],)

function variable_dims(method::IterativeReconstruction, acq::AcquisitionInfo)
    if method.domain isa KSpaceDomain
        k_dims = dimnames(acq.kspace_data)
        return k_dims
    end
    return variable_dims(method.signal_model, acq)
end
variable_dims(::Nothing, acq::AcquisitionInfo) = get_image_dims(acq)
function variable_dims(model::TemporalBasis, acq::AcquisitionInfo)
    img_dims = get_image_dims(acq)
    t_idx = get_time_dim(model.time_dim, img_dims)
    return ntuple(i -> i == t_idx ? :coeff : img_dims[i], length(img_dims))
end

function variable_size(method::IterativeReconstruction, acq::AcquisitionInfo)
    if method.domain isa KSpaceDomain
        img_sz = get_image_size(acq)
        raw_ksp = unname(acq.kspace_data)
        trailing_dims = size(raw_ksp)[3:end]
        return (img_sz[1], img_sz[2], trailing_dims...)
    end
    return variable_size(method.signal_model, acq)
end
variable_size(::Nothing, acq::AcquisitionInfo) = get_image_size(acq)
function variable_size(model::TemporalBasis, acq::AcquisitionInfo)
    img_size = get_image_size(acq)
    t_idx = get_time_dim(model.time_dim, get_image_dims(acq))
    return ntuple(i -> i == t_idx ? size(model.Φ, 2) : img_size[i], length(img_size))
end

function output_dims(method::IterativeReconstruction, acq::AcquisitionInfo)
    if method.domain isa KSpaceDomain
        if method.domain.coil_combination isa NoCoilCombination
            return get_image_dims(acq)
        else
            return filter(!=(:coil), get_image_dims(acq))
        end
    end
    return output_dims(method.signal_model, acq)
end
output_dims(::Nothing, acq::AcquisitionInfo) = get_image_dims(acq)
output_dims(::TemporalBasis, acq::AcquisitionInfo) = get_image_dims(acq)

"""
    build_encoding_operator(acq::AcquisitionInfo, method::AbstractReconstructionMethod; threaded::Bool = true, fast_planning::Bool = false)

Builds the complete encoding operator for the reconstruction method, composing the physical
acquisition encoding operator `𝒜` with the signal model `ℳ` (if present) as `𝒜 * ℳ`.
Under `KSpaceDomain`, builds only the subsampling operator `𝒫`.
"""
function build_encoding_operator(
        acq::AcquisitionInfo,
        method::AbstractReconstructionMethod;
        threaded::Bool = true,
        fast_planning::Bool = false,
    )
    if method isa IterativeReconstruction && method.domain isa KSpaceDomain
        if isnothing(acq.subsampling)
            raw = unname(acq.kspace_data)
            P = Eye(eltype(raw), size(raw)...)
            if acq.kspace_data isa NamedDimsArray
                return NamedDimsOp{dimnames(acq.kspace_data), dimnames(acq.kspace_data)}(P)
            else
                return P
            end
        else
            return get_subsampling_operator(acq)
        end
    end

    𝒜 = get_encoding_operator(acq; threaded, fast_planning)
    ℳ = signal_model_operator(method, acq; threaded)
    isnothing(ℳ) && return 𝒜
    if 𝒜 isa NamedDimsOp && ℳ isa NamedDimsOp
        @argcheck dimnames(𝒜, 2) == dimnames(ℳ, 1) "signal model codomain does not match encoding operator domain"
    end
    return 𝒜 * ℳ
end
