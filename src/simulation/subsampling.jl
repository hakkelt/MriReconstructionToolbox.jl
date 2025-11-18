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
    function UniformRandomSampling(acceleration, center_fraction=0.1)
        @assert 1 <= acceleration "Acceleration factor must be >= 1"
        @assert 0 <= center_fraction < 1 "Center fraction must be in [0, 1)"
        new(acceleration, center_fraction)
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
    function GaussianDistribution(std=1/3)
        @assert 0 < std "Standard deviation must be positive"
        new(std)
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
    function PolynomialDistribution(p=4)
        @assert 0 < p "Polynomial exponent must be positive"
        new(p)
    end
end

"""
    VariableDensitySampling(distribution::VariableDensityDistribution, acceleration::Float64; center_fraction::Float64=0.1)

Create a variable density random sampling pattern based on the specified distribution, acceleration factor,
and center fraction. The `distribution` parameter can be either `GaussianDistribution` or `PolynomialDistribution`.
The `acceleration` parameter controls the overall undersampling factor, while the `center_fraction` parameter
specifies the fraction of low-frequency k-space positions to be fully sampled.
"""
struct VariableDensitySampling{D<:VariableDensityDistribution} <: Subsampling
    distribution::D
    acceleration::Float64
    center_fraction::Float64
    function VariableDensitySampling(distribution::D, acceleration::Real, center_fraction::Real=0.1) where {D<:VariableDensityDistribution}
        @assert 1 <= acceleration "Acceleration factor must be >= 1"
        @assert 0 <= center_fraction < 1 "Center fraction must be in [0, 1)"
        new{D}(distribution, acceleration, center_fraction)
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
    function PoissonDiskSampling(acceleration::Float64, center_fraction::Float64=0.1)
        @assert 1 <= acceleration "Acceleration factor must be >= 1"
        @assert 0 <= center_fraction < 1 "Center fraction must be in [0, 1)"
        new(acceleration, center_fraction)
    end
end

function create_sampling_pattern(subsampling::Subsampling, dims::NTuple{N,Int}; subsample_freq_encoding::Bool=false, number_of_trials::Int=5) where {N}
    @argcheck N == 2 || N == 3 "Only 2D and 3D sampling patterns are supported"
    if subsampling isa PoissonDiskSampling
        @argcheck (N == 2 && subsample_freq_encoding) || (N == 3 && !subsample_freq_encoding) "Only 2D Poisson disk sampling patterns are supported"
        number_of_trials = 1
    end
    if !subsample_freq_encoding
        dims = dims[2:end]
    end
    center_region = get_fully_sampled_region(dims, subsampling.center_fraction)
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
    if !isnothing(center_region)
        W[center_region...] .= 0
    end
    mask = falses(dims)
    num_samples = round(Int, prod(dims) / subsampling.acceleration)
    if !isnothing(center_region)
        mask[center_region...] .= true
        W[center_region...] .= 0
        num_samples -= prod(map(length, center_region))
    end
    for idx in sample(vec(CartesianIndices(dims)), ProbabilityWeights(vec(W)), num_samples; replace=false)
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

function to_displayable_mask(pattern, dims::NTuple{N,Int}) where {N}
    if pattern isa Tuple && length(pattern) == 2 && pattern[2] isa AbstractVector{Bool}
        mask = falses(dims)
        mask[:, pattern[2]] .= true
        return mask
    elseif pattern isa Tuple && length(pattern) == 2 && pattern[2] isa AbstractArray{Bool}
        return pattern[2]
    elseif pattern isa AbstractArray{Bool}
        return pattern
    else
        error("Unsupported pattern format")
    end
end

construct_weights(::UniformRandomSampling, dims) = ones(Float64, dims)
function construct_weights(subsampling::VariableDensitySampling{GaussianDistribution}, dims)
    centers = [d / 2 for d in dims]
    W = ones(Float64, dims)
    for I in CartesianIndices(dims)
        dist2 = sum(((Tuple(I) .- centers) ./ (0.5 .* dims)).^2)
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
            dist = sqrt(sum(((Tuple(I) .- centers) ./ (0.5 .* dims)).^2))
            W[I] = (1 - dist)^subsampling.distribution.p
        end
    else
        center_width = length(center_region[1])
        normalizers = [(d - center_width) for d in dims]
        for I in CartesianIndices(dims)
            dist = sqrt(sum(((Tuple(I) .- centers) ./ normalizers).^2))
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
    width = (prod(dims) * center_fraction)^(1/length(dims))
    centers = [d / 2 for d in dims]
    starts = [round(Int, s) for s in centers .- width ./ 2]
    ends = [round(Int, e) for e in centers .+ width ./ 2]
    return tuple([s:e for (s, e) in zip(starts, ends)]...)
end

function get_sidelobe_to_peak_ratio(mask)
    psf = FFTW.ifft(mask .* 1.0)
    return maximum(abs, @view(psf[2:end])) / abs(psf[1])
end
