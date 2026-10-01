struct BatchedNFFTOp{
        T,
        D,
        M,
        P <: NFFT.AbstractNFFTPlan{T, D},
        K <: AbstractArray{Complex{T}},
        DC <: AbstractArray{T},
    } <: AbstractOperators.LinearOperator
    plan::P
    dim_in::NTuple{M, Int}
    nframe::Int
    ksp_buffer::K
    dcf::DC
    normal_op::Base.RefValue{Any}
end

"""
    BatchedNFFTOp(image_size::NTuple{D,Int}, trajectory::AbstractArray{T}, dcf = nothing;
                  batch::Tuple = (), nframe::Int = 0, array_type::Type, kwargs...)

The non-uniform fast Fourier transform of a stack of images, as one transform on a device.

The last `nframe` axes of `trajectory` are frames: `trajectory` is `(D, samples..., frames...)`,
and frame `t` of the stack is transformed with `trajectory[:, .., t]`. `batch` are axes between
the samples and the frames whose images share their frame's trajectory, such as coils. The
operator maps `(image_size..., batch..., frames...)` to `(samples..., batch..., frames...)`.
It is what `BatchOp` of one [`NFFTOp`](@ref) per frame, repeated over `batch`, computes, but
each step of the transform runs once for the whole stack rather than once per image.

`dcf` follows [`NFFTOp`](@ref)'s contract, with the shape `(samples..., frames...)`: `nothing`
is no density compensation, `:auto` estimates it per frame, an array is used as given. The
remaining keywords go to the NFFT plan.

The operator needs device storage: `array_type` must be the array type of a GPU backend, whose
NFFT plan transforms a batch at once. Its normal operator is the Toeplitz embedding of
[`NFFTOp`](@ref)'s, built once and kept by the operator.
"""
function BatchedNFFTOp(
        image_size::NTuple{D, Int},
        trajectory::AbstractArray{T},
        dcf::Union{Nothing, Symbol, AbstractArray} = nothing;
        batch::Tuple = (),
        nframe::Int = 0,
        array_type::Type = Array{T},
        dcf_estimation_iterations::Int = 20,
        dcf_correction_function::Function = identity,
        kwargs...,
    ) where {T, D}
    check_traj(trajectory, D)
    0 <= nframe <= ndims(trajectory) - 2 ||
        throw(ArgumentError("a trajectory of $(ndims(trajectory)) axes has no room for $nframe frame axes"))
    sample_size = size(trajectory)[2:(end - nframe)]
    frame_size = size(trajectory)[(end - nframe + 1):end]
    arr_wrapper = _array_wrapper_type(array_type)
    k = Matrix{T}(reshape(trajectory, D, :))
    plan = _batched_nfft_plan(arr_wrapper, k, image_size, prod(batch; init = 1), prod(frame_size; init = 1); kwargs...)
    dcf_host = _resolve_batched_dcf(
        dcf, k, image_size, sample_size, frame_size, T, dcf_estimation_iterations, dcf_correction_function; kwargs...
    )
    dcf_shape = (sample_size..., map(_ -> 1, batch)..., frame_size...)
    adapted_dcf = _nfft_adapt(arr_wrapper, reshape(collect(T, dcf_host), dcf_shape))
    ksp_buffer = similar(adapted_dcf, Complex{T}, (sample_size..., batch..., frame_size...))
    dim_in = (image_size..., batch..., frame_size...)
    return BatchedNFFTOp{T, D, length(dim_in), typeof(plan), typeof(ksp_buffer), typeof(adapted_dcf)}(
        plan, dim_in, nframe, ksp_buffer, adapted_dcf, Ref{Any}(nothing)
    )
end

"""
    _batched_nfft_plan(array_type, k, image_size, batch, frames; kwargs...)

The NFFT plan of a [`BatchedNFFTOp`](@ref): `batch × frames` transforms, `frames` consecutive
groups of the nodes `k`. Defined for device array types by the GPU extension.
"""
_batched_nfft_plan(::Type, k, image_size, batch, frames; kwargs...) = throw(
    ArgumentError("BatchedNFFTOp needs device storage: pass the array type of a GPU backend as `array_type`")
)

function _resolve_batched_dcf(::Nothing, k, image_size, sample_size, frame_size, T, iters, correction; kwargs...)
    return ones(T, sample_size..., frame_size...)
end
function _resolve_batched_dcf(dcf::Symbol, k, image_size, sample_size, frame_size, T, iters, correction; kwargs...)
    dcf === :auto || throw(ArgumentError("dcf as a Symbol must be :auto, got :$dcf"))
    J = prod(sample_size)
    frames = map(1:prod(frame_size; init = 1)) do t
        plan = create_plan(k[:, ((t - 1) * J + 1):(t * J)], image_size, false; kwargs...)
        correction(reshape(NFFTTools.sdc(plan; iters), sample_size))
    end
    return reshape(stack(frames), sample_size..., frame_size...)
end
function _resolve_batched_dcf(dcf::AbstractArray, k, image_size, sample_size, frame_size, T, iters, correction; kwargs...)
    size(dcf) == (sample_size..., frame_size...) || throw(
        DimensionMismatch("dcf of size $(size(dcf)) does not match the trajectory's $((sample_size..., frame_size...))")
    )
    eltype(dcf) == T || throw(ArgumentError("the element type of dcf must be the trajectory's, $T"))
    return dcf
end

function mul!(ksp::AbstractArray, op::BatchedNFFTOp, img::AbstractArray)
    AbstractOperators.check(ksp, op, img)
    mul!(ksp, op.plan, img)
    return ksp
end

function mul!(img::AbstractArray, adjop::AdjointOperator{<:BatchedNFFTOp}, ksp::AbstractArray)
    AbstractOperators.check(img, adjop, ksp)
    op = adjop.A
    op.ksp_buffer .= ksp .* op.dcf
    mul!(img, op.plan', op.ksp_buffer)
    return img
end

# Properties

size(L::BatchedNFFTOp) = size(L.ksp_buffer), L.dim_in
fun_name(::BatchedNFFTOp) = "𝒩"
domain_type(::BatchedNFFTOp{T}) where {T} = complex(T)
codomain_type(::BatchedNFFTOp{T}) where {T} = complex(T)
domain_array_type(op::BatchedNFFTOp{T, D, M, P, K}) where {T, D, M, P, K} = _array_wrapper_type(K){domain_type(op)}
codomain_array_type(op::BatchedNFFTOp{T, D, M, P, K}) where {T, D, M, P, K} = _array_wrapper_type(K){codomain_type(op)}

# The transform runs on the device; there is no host threading to switch.
is_threaded(::BatchedNFFTOp) = false
supports_threading(::BatchedNFFTOp) = false

# Not thread-safe: `ksp_buffer` and the plan's grid are scratch written by every `mul!`.
is_thread_safe(::BatchedNFFTOp) = false

# The normal operator owns FFT plans and a grid-sized buffer; it is built on the first request
# and returned to every later one.
AbstractOperators.has_optimized_normalop(::BatchedNFFTOp) = true
function AbstractOperators.get_normal_op(op::BatchedNFFTOp)
    cached = op.normal_op[]
    cached === nothing || return cached
    normal = _get_normal_op(op)
    op.normal_op[] = normal
    return normal
end

function _copy_operator_impl(
        op::BatchedNFFTOp{T, D, M, P, K, DC}; storage_type = nothing, threaded = nothing
    ) where {T, D, M, P, K, DC}
    if storage_type === nothing
        return BatchedNFFTOp{T, D, M, P, K, DC}(
            op.plan, op.dim_in, op.nframe, similar(op.ksp_buffer), op.dcf, Ref{Any}(nothing)
        )
    end
    nb = M - D - op.nframe
    sample_size = size(op.ksp_buffer)[1:(end - nb - op.nframe)]
    frame_size = op.dim_in[(end - op.nframe + 1):end]
    trajectory = reshape(collect(op.plan.k), D, sample_size..., frame_size...)
    dcf = reshape(collect(op.dcf), sample_size..., frame_size...)
    return BatchedNFFTOp(
        op.dim_in[1:D], trajectory, dcf;
        batch = op.dim_in[(D + 1):(D + nb)], nframe = op.nframe, array_type = storage_type{T},
        m = op.plan.params.m, σ = op.plan.params.σ,
    )
end
