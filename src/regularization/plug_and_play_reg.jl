"""
	DenoiserProx(denoiser, strength, complex_handling, spatial_size, num_batch)

Proximable function whose proximal operator is an arbitrary image denoiser, as used by plug-and-play priors.

This is the function behind [`PlugAndPlay`](@ref). It is *not* the proximal operator of any function that is
known in closed form, so `DenoiserProx(...)(x)` returns `NaN`: the value of the implicit prior is unavailable.
Solvers that only need the prox (`ISTA`, `FISTA`, `ADMM`) work with it; solvers that compare objective values
across iterations (`PANOC`, `PANOCplus`, `ZeroFPR`) do not.
"""
struct DenoiserProx{D, R <: Real, N}
    denoiser::D
    strength::R
    complex_handling::Symbol
    spatial_size::NTuple{N, Int}
    num_batch::Int
end

# The prior behind a denoiser is generally unknown, so nothing may be claimed about it. In particular it is
# not assumed convex: plug-and-play convergence results rest on the denoiser being non-expansive, not on
# convexity of any underlying function.
ProximalCore.is_convex(::Type{<:DenoiserProx}) = false
ProximalCore.is_smooth(::Type{<:DenoiserProx}) = false
ProximalCore.is_separable(::Type{<:DenoiserProx}) = false

(f::DenoiserProx)(x) = real(eltype(x))(NaN)

_pnp_reshaped(f::DenoiserProx, x) = reshape(x, f.spatial_size..., f.num_batch)

function ProximalCore.prox!(y, f::DenoiserProx, x, gamma)
    # The denoiser is parameterized by an assumed noise standard deviation. For the proximal step of a
    # penalty of weight λ with step size γ the effective noise level is sqrt(γλ); `strength` plays the role
    # of sqrt(λ), so that halving `strength` halves the noise level the denoiser is told to remove.
    σ = f.strength * sqrt(gamma)
    xr = _pnp_reshaped(f, x)
    yr = _pnp_reshaped(f, y)
    for batch in 1:(f.num_batch)
        slice = selectdim(xr, ndims(xr), batch)
        out = selectdim(yr, ndims(yr), batch)
        copyto!(out, _denoise(f, collect(slice), σ))
    end
    return real(eltype(x))(NaN)
end

_denoise(f::DenoiserProx, slice::AbstractArray{<:Real}, σ) = f.denoiser(slice, σ)

function _denoise(f::DenoiserProx, slice::AbstractArray{<:Complex}, σ)
    f.complex_handling === :native && return f.denoiser(slice, σ)
    if f.complex_handling === :split
        return complex.(f.denoiser(real.(slice), σ), f.denoiser(imag.(slice), σ))
    end
    # :magnitude -- denoise the magnitude and keep the original phase, which is the usual choice when the
    # denoiser was trained on magnitude images and the phase is smooth anyway.
    magnitude = f.denoiser(abs.(slice), σ)
    return magnitude .* cis.(angle.(slice))
end

"""
	PlugAndPlay(denoiser; strength=1, complex_handling=:split, spatial_dims=nothing)

Create a plug-and-play regularization term: the proximal operator of the regularizer is replaced by an
off-the-shelf image `denoiser`, so that any denoiser can act as an implicit image prior in an iterative
reconstruction.

# Arguments
- `denoiser`: A callable `denoiser(image, σ)` returning a denoised copy of `image`, where `σ` is the noise
  standard deviation to remove. It is called once per batch slice (see `spatial_dims`).
- `strength`: (optional) Scales the noise level handed to the denoiser: the prox at step size `γ` calls the
  denoiser with `σ = strength * sqrt(γ)`. This is the tuning knob that plays the role of `λ` in an explicit
  penalty; larger values denoise more aggressively.
- `complex_handling`: (optional) How to apply a real-valued denoiser to a complex image. `:split` (default)
  denoises the real and imaginary parts separately, `:magnitude` denoises the magnitude and keeps the phase,
  and `:native` passes the complex array to the denoiser unchanged.
- `spatial_dims`: (optional) Number of leading dimensions forming one image for the denoiser. Defaults to
  `min(ndims(x), 2)` if not given; everything after them is looped over.

# Notes
- Plug-and-play priors were introduced by Venkatakrishnan, Bouman & Wohlberg, *Plug-and-play priors for model
  based reconstruction*, GlobalSIP 2013, and applied to MRI by, among others, Ahmad et al., *Plug-and-play
  methods for magnetic resonance imaging*, IEEE Signal Process Mag 2020. Common choices of `denoiser` are
  BM3D and learned denoisers (DnCNN and successors).
- The implicit prior has no value function, so the objective reported by the solvers is `NaN` and any
  convergence check based on the objective is meaningless. Convergence is a fixed-point property here, not a
  descent property. Use `ISTA`, `FISTA` or `ADMM`; the line-search algorithms (`PANOC`, `PANOCplus`,
  `ZeroFPR`) require objective values and will not work.
- Convergence guarantees for plug-and-play require the denoiser to be non-expansive (and, for the strongest
  results, a proximal operator of a convex function itself). Ordinary denoisers satisfy neither exactly, so
  the iteration can in principle stall or oscillate; in practice a moderate `strength` and a bounded number
  of iterations is what makes it work.
- No denoiser is bundled with this package. Anything callable works, e.g.
  `PlugAndPlay((img, σ) -> bm3d(img, σ); strength=0.05)` with BM3D.jl, or a wrapper around a neural network.
  A soft-thresholding "denoiser" reproduces [`L1Image`](@ref) exactly and is a useful sanity check.

# Example
```julia
julia> soft_threshold(img, σ) = sign.(img) .* max.(abs.(img) .- σ^2, 0);

julia> reg = PlugAndPlay(soft_threshold; strength=sqrt(0.1));
```
"""
struct PlugAndPlay{D, T} <: Regularization
    denoiser::D
    strength::T
    complex_handling::Symbol
    spatial_dims::Union{Nothing, Int}
    function PlugAndPlay(
            denoiser::D; strength::T = 1, complex_handling::Symbol = :split,
            spatial_dims::Union{Nothing, Int} = nothing
        ) where {D, T}
        @argcheck strength isa Real "strength must be a scalar"
        @argcheck strength >= 0 "strength must be non-negative"
        @argcheck complex_handling in (:split, :magnitude, :native) "complex_handling must be :split, :magnitude or :native, got :$complex_handling"
        @argcheck spatial_dims === nothing || spatial_dims > 0 "spatial_dims must be positive"
        return new{D, T}(denoiser, strength, complex_handling, spatial_dims)
    end
end

get_operator(::PlugAndPlay, x::AbstractArray; threaded::Bool = true) = Eye(x)
function get_operator(::PlugAndPlay, x::NamedDimsArray; threaded::Bool = true)
    return NamedDimsOp{dimnames(x), dimnames(x)}(Eye(parent(x)))
end

_pnp_spatial_dims(reg::PlugAndPlay, n::Int) = reg.spatial_dims === nothing ? min(n, 2) : min(reg.spatial_dims, n)

function get_affected_dims(reg::PlugAndPlay, ::Nothing, image_dims)
    # The denoiser sees the leading dimensions jointly; the rest are looped over and stay decomposable.
    return image_dims[1:_pnp_spatial_dims(reg, length(image_dims))]
end

get_affected_dims(reg::PlugAndPlay, ::AcquisitionInfo, image_dims) = get_affected_dims(reg, nothing, image_dims)

# The denoiser is a black box: nothing is known about how its output scales with its input, so the noise
# level it is asked to remove is scaled with the image instead (σ has the units of the image).
scale_regularization(reg::PlugAndPlay, factor::Real) = PlugAndPlay(
    reg.denoiser; strength = reg.strength * factor, complex_handling = reg.complex_handling,
    spatial_dims = reg.spatial_dims
)

function materialize(reg::PlugAndPlay, x::Variable{T}; threaded::Bool) where {T}
    x_val = ~x
    n_spatial = _pnp_spatial_dims(reg, ndims(x_val))
    spatial_size = NTuple{n_spatial, Int}(size(x_val)[1:n_spatial])
    num_batch = prod(size(x_val)[(n_spatial + 1):end]; init = 1)
    f = DenoiserProx(reg.denoiser, real(T)(reg.strength), reg.complex_handling, spatial_size, num_batch)
    op = get_operator(reg, x_val; threaded)
    repr = @sprintf "PnP[σ=%g√γ](%s)" real(T)(reg.strength) get_name(x)
    return StructuredOptimization.Term(1, f, op * x, repr)
end
