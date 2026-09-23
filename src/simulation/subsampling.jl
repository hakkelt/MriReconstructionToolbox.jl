abstract type Subsampling end

"""
    UniformRandomSampling(acceleration::Float64; center_fraction::Float64=0.1)

Create a uniform random sampling pattern with the specified acceleration factor and center fraction.
The `acceleration` parameter controls the overall undersampling factor, while the `center_fraction` parameter
specifies the fraction of low-frequency k-space positions to be fully sampled.
"""
struct UniformRandomSampling <: Subsampling
    acceleration::Float64
    center_fraction::Float64
    function UniformRandomSampling(acceleration, center_fraction = 0.1)
        @argcheck 1 <= acceleration "Acceleration factor must be >= 1"
        @argcheck 0 <= center_fraction < 1 "Center fraction must be in [0, 1)"
        return new(acceleration, center_fraction)
    end
end

abstract type VariableDensityDistribution end
"""
    GaussianDistribution(std::Float64=1/3)

Create a Gaussian variable density distribution with the specified standard deviation `std`.
The sampling probability follows a Gaussian profile centered at k-space center:
    W(r) = exp(-0.5 * (r / std)^2), where r is the normalized distance from the k-space center.
"""
struct GaussianDistribution <: VariableDensityDistribution
    std::Float64
    function GaussianDistribution(std = 1 / 3)
        @argcheck 0 < std "Standard deviation must be positive"
        return new(std)
    end
end

"""
    PolynomialDistribution(p::Float64=4)

Create a Polynomial variable density distribution with the specified exponent `p`.
The sampling probability is proportional to power of the distance from the k-space center:
    W(r) = (1 - r)^p, where r is the normalized distance from the k-space center.
"""
struct PolynomialDistribution <: VariableDensityDistribution
    p::Float64
    function PolynomialDistribution(p = 4)
        @argcheck 0 < p "Polynomial exponent must be positive"
        return new(p)
    end
end

"""
    VariableDensitySampling(distribution::VariableDensityDistribution, acceleration::Float64; center_fraction::Float64=0.1)

Create a variable density random sampling pattern based on the specified distribution, acceleration factor,
and center fraction. The `distribution` parameter can be either `GaussianDistribution` or `PolynomialDistribution`.
The `acceleration` parameter controls the overall undersampling factor, while the `center_fraction` parameter
specifies the fraction of low-frequency k-space positions to be fully sampled.
"""
struct VariableDensitySampling{D <: VariableDensityDistribution} <: Subsampling
    distribution::D
    acceleration::Float64
    center_fraction::Float64
    function VariableDensitySampling(distribution::D, acceleration::Real, center_fraction::Real = 0.1) where {D <: VariableDensityDistribution}
        @argcheck 1 <= acceleration "Acceleration factor must be >= 1"
        @argcheck 0 <= center_fraction < 1 "Center fraction must be in [0, 1)"
        return new{D}(distribution, acceleration, center_fraction)
    end
end

"""
    PoissonDiskSampling(acceleration::Float64, center_fraction::Float64=0.1)

Create a Poisson disk sampling pattern with the specified acceleration factor and center fraction.
The `acceleration` parameter controls the overall undersampling factor, while the `center_fraction` parameter
specifies the fraction of low-frequency k-space positions to be fully sampled.
"""
struct PoissonDiskSampling <: Subsampling
    acceleration::Float64
    center_fraction::Float64
    function PoissonDiskSampling(acceleration::Real, center_fraction::Real = 0.1)
        @argcheck 1 <= acceleration "Acceleration factor must be >= 1"
        @argcheck 0 <= center_fraction < 1 "Center fraction must be in [0, 1)"
        return new(acceleration, center_fraction)
    end
end

"""
    RegularLatticeSampling(acceleration::Real; center_fraction=0.0)

Regular (equispaced) undersampling: every `acceleration`-th phase encode is acquired, optionally
with a fully sampled autocalibration (ACS) band in the centre. This is the pattern GRAPPA and
every product parallel-imaging sequence use, and unlike the random generators it is
deterministic, so a single realisation is the pattern.

- `acceleration` must be a whole number. Over two subsampled dimensions (a 3D acquisition) it is
  factored into a stride per dimension, as close to equal as its divisors allow — `4` becomes
  2×2, `3` becomes 3×1.
- `center_fraction` is the fraction of k-space positions held in the fully sampled ACS band, the
  same meaning it has for the random generators. The default of `0` is a pure regular lattice;
  GRAPPA and SPIRiT need a non-zero value to calibrate on.

The net acceleration is `acceleration` only when `center_fraction` is at its default: the ACS
band adds samples on top of the lattice.

Partial Fourier is a separate scheme, not a modifier of this one — see
[`PartialFourierSampling`](@ref).

See also [`create_sampling_pattern`](@ref).
"""
struct RegularLatticeSampling <: Subsampling
    acceleration::Float64
    center_fraction::Float64
    function RegularLatticeSampling(acceleration::Real; center_fraction::Real = 0.0)
        @argcheck 1 <= acceleration "Acceleration factor must be >= 1"
        @argcheck isinteger(acceleration) "Regular lattice sampling acquires every R-th line, so the acceleration factor must be a whole number; got $acceleration"
        @argcheck 0 <= center_fraction < 1 "Center fraction must be in [0, 1)"
        return new(acceleration, center_fraction)
    end
end

"""
    PartialFourierSampling(fraction::Real)

Partial-Fourier truncation: every phase encode of the first `fraction` of the last subsampled
dimension is acquired, and nothing past it. It exploits the Hermitian symmetry of k-space rather
than parallel imaging, so it is a scheme of its own and not a modifier of
[`RegularLatticeSampling`](@ref) — combining a lattice with a truncated band would leave a
pattern neither a homodyne/POCS reconstruction nor a GRAPPA kernel handles as intended.

`fraction` must be in `(0, 1]` and, for the acquired band to cover the k-space centre at all,
above `0.5`. The net acceleration is `1 / fraction`.

See also [`create_sampling_pattern`](@ref), [`Homodyne`](@ref), [`POCS`](@ref).
"""
struct PartialFourierSampling <: Subsampling
    partial_fourier::Float64
    function PartialFourierSampling(fraction::Real)
        @argcheck 0 < fraction <= 1 "Partial Fourier fraction must be in (0, 1]"
        return new(fraction)
    end
end

"""
    _center_fraction(subsampling::Subsampling) -> Float64

The fraction of k-space every generator keeps fully sampled in the centre. All but
[`PartialFourierSampling`](@ref) carry it as a field; partial Fourier has no calibration band, so
its centre band is empty.
"""
_center_fraction(subsampling::Subsampling) = subsampling.center_fraction
_center_fraction(::PartialFourierSampling) = 0.0

"""
    _systematic_strides(R::Int, ndims::Int) -> NTuple{ndims, Int}

Factor a total acceleration `R` into one stride per subsampled dimension, each factor as close to
`R^(1/ndims)` as the divisors of `R` allow. Only the divisors are candidates, so the strides
multiply back to exactly `R` (a 3 spread over two dimensions stays 3×1 rather than becoming a
non-integer stride).
"""
function _systematic_strides(R::Int, n::Int)
    n == 1 && return (R,)
    target = R^(1 / n)
    divisors = [d for d in 1:R if R % d == 0]
    _, best = findmin(d -> abs(d - target), divisors)
    d = divisors[best]
    return (d, _systematic_strides(R ÷ d, n - 1)...)
end

function _create_sampling_pattern(subsampling::RegularLatticeSampling, dims, center_region)
    mask = falses(dims)
    strides = _systematic_strides(round(Int, subsampling.acceleration), length(dims))
    mask[ntuple(i -> 1:strides[i]:dims[i], length(dims))...] .= true
    if !isnothing(center_region)
        mask[center_region...] .= true
    end
    return mask
end

function _create_sampling_pattern(subsampling::PartialFourierSampling, dims, center_region)
    mask = falses(dims)
    # The acquired band is the first `partial_fourier` of the last dimension, fully sampled.
    last_acquired = round(Int, subsampling.partial_fourier * dims[end])
    selectdim(mask, length(dims), 1:last_acquired) .= true
    return mask
end

"""
    create_sampling_pattern(subsampling::Subsampling, dims; subsample_freq_encoding=false, number_of_trials=5)

Create a k-space sampling pattern for the given subsampling strategy and k-space size `dims`
(a 2- or 3-tuple). Unless `number_of_trials == 1`, several candidate patterns are generated and
the one with the lowest sidelobe-to-peak ratio of the point spread function is kept.

By default the frequency-encoding (first) dimension is fully sampled: the pattern is generated
over `dims[2:end]` and returned as `(:, mask)`, ready to be used as the `subsampling` argument
of `AcquisitionInfo`. With `subsample_freq_encoding = true` all dimensions are subsampled and a
`Bool` mask of size `dims` is returned instead.

See also [`to_displayable_mask`](@ref) to convert either return form into a full-size mask.
"""
function create_sampling_pattern(subsampling::Subsampling, dims::NTuple{N, Int}; subsample_freq_encoding::Bool = false, number_of_trials::Int = 5) where {N}
    @argcheck N == 2 || N == 3 "Only 2D and 3D sampling patterns are supported"
    if subsampling isa PoissonDiskSampling
        @argcheck (N == 2 && subsample_freq_encoding) || (N == 3 && !subsample_freq_encoding) "Only 2D Poisson disk sampling patterns are supported"
        number_of_trials = 1
    elseif subsampling isa Union{RegularLatticeSampling, PartialFourierSampling}
        # Deterministic: every trial would return the same pattern.
        number_of_trials = 1
    end
    if !subsample_freq_encoding
        dims = dims[2:end]
    end
    center_region = get_fully_sampled_region(dims, _center_fraction(subsampling))
    if number_of_trials == 1
        mask = _create_sampling_pattern(subsampling, dims, center_region)
    else
        best_mask = nothing
        best_sidelobe_ratio = Inf
        for _ in 1:number_of_trials
            mask = _create_sampling_pattern(subsampling, dims, center_region)
            sidelobe_ratio = get_sidelobe_to_peak_ratio(mask)
            if sidelobe_ratio < best_sidelobe_ratio
                best_sidelobe_ratio = sidelobe_ratio
                best_mask = mask
            end
        end
        mask = best_mask
    end
    if subsample_freq_encoding
        return mask
    else
        return (:, mask)
    end
end

function _create_sampling_pattern(subsampling::Subsampling, dims, center_region)
    W = construct_weights(subsampling, dims)
    mask = falses(dims)
    num_samples = round(Int, prod(dims) / subsampling.acceleration)
    if !isnothing(center_region)
        mask[center_region...] .= true
        W[center_region...] .= 0
        num_samples -= prod(map(length, center_region))
    end
    num_samples = max(num_samples, 0) # the center region may already exceed the sample budget
    for idx in sample(vec(CartesianIndices(dims)), ProbabilityWeights(vec(W)), num_samples; replace = false)
        mask[idx] = true
    end
    return mask
end

function _create_sampling_pattern(subsampling::PoissonDiskSampling, dims, center_region)
    mask = falses(dims)
    num_samples = round(Int, prod(dims) / subsampling.acceleration)
    if !isnothing(center_region)
        mask[center_region...] .= true
        num_samples -= prod(map(length, center_region))
    end
    if num_samples <= 0 # the center region may already exceed the sample budget
        return mask
    end
    # Simple dart throwing algorithm for Poisson disk sampling
    min_dist = sqrt(prod(dims) / num_samples) / 2
    points = CartesianIndex[]
    attempts = 0
    max_attempts = num_samples * 10
    while length(points) < num_samples && attempts < max_attempts
        candidate = CartesianIndex(rand(1:dims[1]), rand(1:dims[2]))
        if !mask[candidate] && all(p -> norm(Tuple(p) .- Tuple(candidate)) >= min_dist, points)
            push!(points, candidate)
            mask[candidate] = true
        end
        attempts += 1
    end
    return mask
end

"""
    _phase_encode_mask(keep::AbstractVector{Bool}, nsamples::Int) -> BitMatrix

The `(:, keep)` sampling-pattern form spelled out as a dense mask: every readout sample of the
phase encodes `keep` selects. Its own function so that the selector's concrete type is available
here (see the note at the call site in [`to_displayable_mask`](@ref)).
"""
function _phase_encode_mask(keep::AbstractVector{Bool}, nsamples::Int)
    # `reshape(keep, 1, length(keep))`, not `reshape(keep, 1, :)`: the colon form goes through
    # `_reshape_uncolon`, whose result the compiler gives up on (`::Any`), which turns the fill
    # below into a runtime dispatch.
    mask = falses(nsamples, length(keep))
    mask .= reshape(keep, 1, length(keep))
    return mask
end

"""
    to_displayable_mask(pattern, dims)

Convert a sampling pattern into a `Bool` mask of size `dims` suitable for display. Accepts every
form a [`CartesianAcquisitionInfo`](@ref)'s `subsampling` takes: a plain `Bool` mask (returned as
is), the `(:, mask)` form [`create_sampling_pattern`](@ref) returns when the frequency-encoding
dimension is fully sampled (a 2D acquisition, so `dims` must be two-dimensional for it), and a
per-axis tuple with one selector per dimension of `dims` — `:`, a `Bool` vector, or an index vector
or range, e.g. `(:, [1, 3, 5])`.
"""
function to_displayable_mask(pattern, dims::NTuple{N, Int}) where {N}
    if pattern isa Tuple && length(pattern) == 2 && first(pattern) isa Colon && pattern[2] isa AbstractVector{Bool}
        # `(:, keep)` says "every readout sample, the phase encodes `keep` selects", which is a
        # statement about a 2D array.
        N == 2 || throw(ArgumentError("the `(:, mask)` pattern form describes a 2D acquisition, got dims $dims"))
        # `last(pattern)`, not `pattern[2]`: inside this branch the compiler carries `pattern[2]`
        # as `Union{Colon, AbstractVector{Bool}}` (the `isa` test narrows the check, not the
        # value), and everything downstream of that union is a runtime dispatch. `last` of a
        # two-element tuple is the selector's own type, and `_phase_encode_mask` is then a
        # function barrier specialized on it.
        return _phase_encode_mask(last(pattern), dims[1])
    elseif pattern isa Tuple && length(pattern) == 2 && first(pattern) isa Colon && pattern[2] isa AbstractArray{Bool}
        # `(:, mask)` with a ky–kz mask: every readout sample of the phase encodes it selects.
        return pattern[2]
    elseif pattern isa Tuple && length(pattern) == 1 && first(pattern) isa AbstractArray{Bool}
        return first(pattern)
    elseif pattern isa AbstractArray{Bool}
        return pattern
    elseif pattern isa Tuple && all(p -> p isa Union{Colon, AbstractVector}, pattern)
        # One selector per axis, as the subsampling operator indexes the full k-space with it.
        length(pattern) == N || throw(ArgumentError("a per-axis pattern needs one selector per dimension of $dims, got $(length(pattern))"))
        return _per_axis_mask(pattern, dims)
    else
        throw(ArgumentError("Unsupported pattern format: $(typeof(pattern))"))
    end
end

# A function barrier: every branch of `to_displayable_mask` is compiled for every pattern type, and
# inlined here the view's type is only known as a union, making the fill a runtime dispatch.
@noinline function _per_axis_mask(pattern::Tuple, dims::Tuple)
    mask = falses(dims)
    fill!(view(mask, pattern...), true)
    return mask
end

construct_weights(::UniformRandomSampling, dims) = ones(Float64, dims)
function construct_weights(subsampling::VariableDensitySampling{GaussianDistribution}, dims)
    centers = [d / 2 for d in dims]
    W = ones(Float64, dims)
    for I in CartesianIndices(dims)
        dist2 = sum(((Tuple(I) .- centers) ./ (0.5 .* dims)) .^ 2)
        W[I] = exp(-0.5 * dist2 / subsampling.distribution.std^2)
    end
    center_region = get_fully_sampled_region(dims, subsampling.center_fraction)
    if !isnothing(center_region)
        W[center_region...] .= 1
    end
    return W
end

function construct_weights(subsampling::VariableDensitySampling{PolynomialDistribution}, dims)
    centers = [d / 2 for d in dims]
    W = ones(Float64, dims)
    center_region = get_fully_sampled_region(dims, subsampling.center_fraction)
    if isnothing(center_region)
        for I in CartesianIndices(dims)
            dist = min(sqrt(sum(((Tuple(I) .- centers) ./ (0.5 .* dims)) .^ 2)), 1.0)
            W[I] = (1 - dist)^subsampling.distribution.p
        end
    else
        center_width = length(center_region[1])
        normalizers = [(d - center_width) for d in dims]
        for I in CartesianIndices(dims)
            # An anisotropic `dims` combined with a single isotropic `center_width` can push
            # a corner's `dist` above 1; clamp it so `(1 - dist)^p` never goes negative for
            # an odd exponent `p`.
            dist = min(sqrt(sum(((Tuple(I) .- centers) ./ normalizers) .^ 2)), 1.0)
            W[I] = (1 - dist)^subsampling.distribution.p
        end
        W[center_region...] .= 1
    end
    return W
end

function get_fully_sampled_region(dims, center_fraction)
    if center_fraction == 0
        return nothing
    end
    width = (prod(dims) * center_fraction)^(1 / length(dims))
    centers = [d / 2 for d in dims]
    starts = [max(round(Int, s), 1) for s in centers .- width ./ 2]
    ends = [min(round(Int, e), d) for (e, d) in zip(centers .+ width ./ 2, dims)]
    return tuple([s:e for (s, e) in zip(starts, ends)]...)
end

function get_sidelobe_to_peak_ratio(mask)
    psf = FFTW.ifft(mask .* 1.0)
    return maximum(abs, @view(psf[2:end])) / abs(psf[1])
end
