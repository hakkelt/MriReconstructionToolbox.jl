"""
	ProximalAverage(functions, weights)

Proximal average of a tuple of proximable `functions` with convex `weights` (non-negative, summing to one).

The proximal average `f̄` of `{f_j}` with weights `{w_j}` is the convex function whose Moreau envelope is the
weighted average of the individual Moreau envelopes. Its proximal operator is exactly the weighted average of
the individual proximal operators,

	prox_{γ f̄}(x) = ∑_j w_j prox_{γ f_j}(x),

(Bauschke, Goebel, Lucet & Wang, *The proximal average: basic theory*, SIAM J Optim 2008), which is what this
type implements. `f̄` is close to, but not equal to, the weighted average `∑_j w_j f_j`; calling a
`ProximalAverage` returns that weighted average, because `f̄` itself has no closed form. Callers that need an
exact objective value should therefore not rely on it.

This is the function behind [`MultiScaleLowRank`](@ref), where the `f_j` are block nuclear norms over
different block sizes and no single prox exists for their sum.
"""
struct ProximalAverage{F <: Tuple, R <: Real}
    functions::F
    weights::Vector{R}
    function ProximalAverage(functions::F, weights::Vector{R}) where {F <: Tuple, R <: Real}
        @argcheck length(functions) == length(weights) "one weight is required per function"
        @argcheck !isempty(functions) "ProximalAverage needs at least one function"
        @argcheck all(≥(0), weights) "weights must be non-negative"
        @argcheck sum(weights) ≈ 1 "weights must sum to 1, got $(sum(weights))"
        return new{F, R}(functions, weights)
    end
end

# The proximal average of convex functions is convex, and it is smooth only in degenerate cases, so the
# conservative answer is the right one here.
ProximalCore.is_convex(::Type{<:ProximalAverage}) = true
ProximalCore.is_smooth(::Type{<:ProximalAverage}) = false
ProximalCore.is_separable(::Type{<:ProximalAverage}) = false

(f::ProximalAverage)(x) = sum(w * g(x) for (w, g) in zip(f.weights, f.functions))

function ProximalCore.prox!(y, f::ProximalAverage, x, gamma)
    # Callers are allowed to take an in-place step (`prox!(x, f, x, gamma)`), and every scale needs
    # to see the original `x`, so keep a copy of the input whenever it aliases the output.
    input = y === x ? copy(x) : x
    fill!(y, zero(eltype(y)))
    buffer = similar(y)
    for (w, g) in zip(f.weights, f.functions)
        ProximalCore.prox!(buffer, g, input, gamma)
        y .+= w .* buffer
    end
    # The contract is that `prox!` returns the function's value *at the point it wrote*, so evaluate
    # at `y`. Averaging the per-scale values at their own prox points would return a different (and,
    # by convexity, strictly smaller) number whenever the scales disagree.
    return f(y)
end

"""
	MultiScaleLowRank(λ; block_sizes, time_dim=nothing, weights=nothing, shift=:none, rng=Random.default_rng())

Create a multi-scale locally low-rank regularization term: the same block-wise nuclear norm as
[`LocallyLowRank`](@ref), but evaluated over several block sizes at once so that both fine, localized
dynamics and large, globally correlated dynamics are penalized by the same term.

# Arguments
- `λ`: Regularization parameter, must be a scalar. Applied to every scale; use `weights` to bias the scales
  relative to each other.
- `block_sizes`: Collection of block edge lengths, one entry per scale. Each entry is either an `Integer`
  (same edge in every spatial dimension) or a tuple with one entry per spatial dimension. A geometric
  progression such as `(4, 8, 16)` is the usual choice.
- `time_dim`: (optional) Dimension holding the frames / contrasts, as for [`LocallyLowRank`](@ref).
- `weights`: (optional) Convex weights of the scales; defaults to uniform. Must be non-negative and sum to 1.
- `shift`, `rng`: (optional) Grid-shift policy, as for [`LocallyLowRank`](@ref). Applied to every scale.

# Notes
- Multi-scale low-rank modelling of dynamic images was introduced by Ong & Lustig, *Beyond low rank + sparse:
  Multiscale low rank matrix decomposition*, IEEE J Sel Top Signal Process 2016, and is available in BART as
  `-R M`. Their formulation decomposes the image into one *separate component per scale*
  (`x = ∑_j x_j`, each `x_j` penalized at its own block size), which is an exact model with an exact prox per
  component; that formulation is expressible directly with this package's image decomposition — one
  [`Component`](@ref) per scale, each carrying a [`LocallyLowRank`](@ref) — and is the better choice when the
  separated components are themselves of interest.
- This term instead penalizes a *single* image at several scales at once, which has no closed-form proximal
  operator, so it uses the proximal average of the per-scale block nuclear norms (see
  [`ProximalAverage`](@ref)). This is an approximation of the sum of the per-scale penalties, not the sum
  itself: the reported objective value is the weighted average of the per-scale penalties and the minimizer
  is that of the proximal average. Prefer it when one regularized image (not a decomposition) is wanted.
- With a single entry in `block_sizes` and weight 1, the term reduces exactly to [`LocallyLowRank`](@ref).
- Cost is the sum of the per-scale costs: one SVD per block per scale and iteration.
"""
struct MultiScaleLowRank{T, B, D, W, RNG} <: Regularization
    λ::T
    block_sizes::B
    time_dim::D
    weights::W
    shift::Symbol
    rng::RNG
    function MultiScaleLowRank(
            λ::T; block_sizes::B, time_dim::D = nothing, weights::W = nothing,
            shift::Symbol = :none, rng::RNG = Random.default_rng()
        ) where {T, B, D, W, RNG}
        @argcheck λ isa Real "MultiScaleLowRank requires a scalar λ"
        @argcheck !isempty(block_sizes) "block_sizes must contain at least one scale"
        for block_size in block_sizes
            @argcheck block_size isa Integer || block_size isa Tuple{Vararg{Integer}} "every entry of block_sizes must be an Integer or a tuple of Integers"
            @argcheck all(block_size .> 0) "block sizes must be positive"
        end
        if weights !== nothing
            @argcheck length(weights) == length(block_sizes) "one weight is required per scale"
        end
        @argcheck shift in (:none, :fixed, :random) "shift must be :none, :fixed or :random, got :$shift"
        _check_dim_spec(time_dim, "time_dim")
        return new{T, B, D, W, RNG}(λ, block_sizes, time_dim, weights, shift, rng)
    end
end

get_operator(::MultiScaleLowRank, x::AbstractArray; threaded::Bool = true) = identity_operator(x)

function get_affected_dims(reg::MultiScaleLowRank, ::Nothing, image_dims)
    # As for LocallyLowRank, the blocks couple every dimension up to and including the temporal one.
    return image_dims[1:get_time_dim(reg.time_dim, image_dims)]
end

# Each scale is a nuclear norm, homogeneous of degree 1, and so is their average: λ scales linearly.
function scale_regularization(reg::MultiScaleLowRank, factor::Real)
    return MultiScaleLowRank(
        reg.λ * factor;
        block_sizes = reg.block_sizes, time_dim = reg.time_dim, weights = reg.weights,
        shift = reg.shift, rng = reg.rng
    )
end

function bind_dimensions(reg::MultiScaleLowRank, image_dims)
    return MultiScaleLowRank(
        reg.λ;
        block_sizes = reg.block_sizes,
        time_dim = get_time_dim(reg.time_dim, image_dims),
        weights = reg.weights,
        shift = reg.shift,
        rng = reg.rng,
    )
end

function _mslr_weights(reg::MultiScaleLowRank, ::Type{R}) where {R <: Real}
    n = length(reg.block_sizes)
    reg.weights === nothing && return fill(R(1 / n), n)
    return R.(collect(reg.weights))
end

function materialize(reg::MultiScaleLowRank, x::Variable{T}; threaded::Bool) where {T}
    x_val = ~x
    dims = dims_of(x_val)
    time_dim = get_time_dim(reg.time_dim, dims)
    @argcheck time_dim > 1 "MultiScaleLowRank needs at least one spatial dimension before the temporal one"
    spatial_size = NTuple{time_dim - 1, Int}(size(x_val)[1:(time_dim - 1)])
    num_batch = prod(size(x_val)[(time_dim + 1):end]; init = 1)
    λ = real(T)(reg.λ)
    scales = Tuple(
        BlockNuclearNorm(
                λ, _llr_block_size(block_size, spatial_size),
                spatial_size, size(x_val, time_dim), num_batch, threaded;
                shift = reg.shift, rng = reg.rng
            ) for block_size in reg.block_sizes
    )
    f = ProximalAverage(scales, _mslr_weights(reg, real(T)))
    op = get_operator(reg, x_val; threaded)
    sizes_repr = join((string(s.block_size) for s in scales), ", ")
    repr = @sprintf "%g ⋅ avg_{b ∈ {%s}} ∑ ‖𝓧_b(%s)‖_*" λ sizes_repr get_name(x)
    return StructuredOptimization.Term(1, f, op * x, repr)
end

# Prox takes a per-scale, per-block SVD: level-3 BLAS, worth threading. See `uses_blas3`.
uses_blas3(::MultiScaleLowRank) = true
