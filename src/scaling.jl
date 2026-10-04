abstract type Scaling end

"""
    NoScaling() <: Scaling

A scaling strategy that applies no scaling to the data.
"""
struct NoScaling <: Scaling end

"""
    QuantileScaling(p = 0.99) <: Scaling

Scale by the `p`-quantile of `|x₀|`, where `x₀` is the adjoint of the encoding operator applied to
the measured k-space data. The default.

A high quantile is the intensity of the object's bright tissue while ignoring the brightest
`1 - p` of the voxels, so a few hot voxels (a k-space spike, fat, a vessel at inflow) do not move
it; with `p = 0.99` the scale is unchanged until about 1 % of the voxels are outliers. It is the
robust member of the family of intensity normalizations that divide by a statistic of the image:
the maximum ([`MaxScaling`](@ref)), the standard deviation ([`StdScaling`](@ref)) and BART's
90th-percentile-or-maximum rule ([`BartScaling`](@ref)).

The quantile is computed on a deterministic subsample of at most `2^18` voxels taken at a prime
stride, which does not alias with the grid, so its cost does not grow with the image. The rank
error of the subsample is `√(p(1-p)/m)`, 0.02 percentile points at `p = 0.99`; on the benchmark
cases the subsampled scale is within 1 % of the exact quantile.

Chosen as the default on the benchmark harness: with one `λ` per penalty for every case (1- and
8-coil 2D, radial, 3D, multi-slice, Cartesian and radial cine), the NRMSE it reaches is on
average 9.9 % above each case's own best `λ` and at worst 45 % above it, against 12.0 % and 95 %
for `BartScaling`, which on piecewise-constant images selects the maximum and then scales with a
single hot voxel.
"""
struct QuantileScaling <: Scaling
    p::Float64
    function QuantileScaling(p::Real = 0.99)
        @argcheck 0 < p <= 1 "p must be in (0, 1]"
        return new(Float64(p))
    end
end

"""
    BartScaling() <: Scaling

A scaling strategy that mimics the scaling used in BART's `pics`.
This approach inspects the distribution of the absolute values of the initial guess `x₀`
(obtained as the adjoint of the encoding operator applied to the measured k-space data)
and selects either the 90th percentile or the maximum value, depending on the spread of the values.
If the difference between the maximum and the 90th percentile is less than twice the difference
between the 90th percentile and the median, the 90th percentile is used; otherwise, the maximum value is used.
This helps to avoid scaling based on outliers in the data.

The median and the 90th percentile are computed on the same subsample as
[`QuantileScaling`](@ref)'s; the maximum is exact.
"""
struct BartScaling <: Scaling end

"""
    MaxScaling() <: Scaling

Scale by the maximum of `|x₀|`, where `x₀` is the adjoint of the encoding operator applied to the
measured k-space data. This is the magnitude normalization of the fastMRI dataset (Zbontar et
al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI", arXiv:1811.08839, 2018). A
single bright voxel sets it.
"""
struct MaxScaling <: Scaling end

"""
    StdScaling() <: Scaling

Scale by the standard deviation of `x₀`, where `x₀` is the adjoint of the encoding operator
applied to the measured k-space data: the instance normalization used by the fastMRI baseline
models (Zbontar et al., arXiv:1811.08839, 2018). It weights voxels quadratically, so a small
fraction of outliers moves it.
"""
struct StdScaling <: Scaling end

"""
    NoiseLevelScaling() <: Scaling

Scale by the noise level of `x₀`, where `x₀` is the adjoint of the encoding operator applied to
the measured k-space data, estimated as the median absolute deviation of its finest Haar details
along the first (readout) axis, divided by 0.6745: the complex standard deviation `√E|z|²` of the
noise (Donoho and Johnstone, "Ideal spatial adaptation
by wavelet shrinkage", Biometrika 81(3), 1994). `λ` is then in units of the noise standard
deviation, as a wavelet threshold is in the universal threshold `σ√(2 log N)`.

The estimate assumes the finest details are mostly noise. It reads low when the adjoint smooths
the noise, as a non-Cartesian adjoint without density compensation does.
"""
struct NoiseLevelScaling <: Scaling end

"""
    MeasurementBasedScaling() <: Scaling

A scaling strategy that scales the data based on the average absolute value of the k-space measurements.
This approach mimic the scaling of MeasurementBasedNormalization from RegularizedLeastSquares.jl
(which is used by MRIReco.jl).
"""
struct MeasurementBasedScaling <: Scaling end

"""
    KSpaceNormScaling(target = 100) <: Scaling

Scale the k-space data to Euclidean norm `target`, as BART's `nlinv` does with `target = 100`
(Uecker et al., "Image reconstruction by regularized nonlinear inversion — joint estimation of coil
sensitivities and image content", Magn. Reson. Med. 60(3), 2008).
"""
struct KSpaceNormScaling <: Scaling
    target::Float64
    function KSpaceNormScaling(target::Real = 100)
        @argcheck target > 0 "target must be positive"
        return new(Float64(target))
    end
end

"""
    SystemMatrixBasedScaling() <: Scaling

Scale by `trace(𝒜ᴴ𝒜) / N`, the mean energy of the encoding operator's columns, as
`SystemMatrixBasedNormalization` in RegularizedLeastSquares.jl does. The trace is estimated with
four Rademacher probes (Hutchinson), so it costs four applications of `𝒜ᴴ𝒜`.

For a positively homogeneous penalty, dividing the data by `s` and dividing `λ` by `s` are the same
problem up to the scale of the solution, so this is RegularizedLeastSquares' regularization weight
`λ·trace(𝒜ᴴ𝒜)/N`. It does not depend on the data, so unlike the other scalings the best `λ` changes
with the data's intensity.
"""
struct SystemMatrixBasedScaling <: Scaling end

"""
    FixedScaling(scale) <: Scaling

A scaling strategy that applies a user-provided, constant scaling factor.
Useful for reproducing a previous reconstruction or comparing reconstructions of
different datasets with a common scale. `scale` must be positive.
"""
struct FixedScaling <: Scaling
    scale::Float64
    function FixedScaling(scale::Real)
        @argcheck scale > 0 "scale must be positive"
        return new(Float64(scale))
    end
end

"""
    get_scale(scaling::Scaling, acq_data::AcquisitionInfo, x₀, 𝒜)

Computes the scaling factor based on the chosen scaling strategy.

# Arguments
- `scaling::Scaling`: The scaling strategy to use.
- `acq_data::AcquisitionInfo`: The acquisition information containing k-space data and other parameters.
- `x₀::AbstractArray`: The initial guess for the image, typically obtained as the adjoint of the encoding operator applied to the k-space data.
- `𝒜`: the encoding operator `x₀` was formed with; only [`SystemMatrixBasedScaling`](@ref) reads it.

# Returns
- `scale::Float64`: The computed scaling factor.
"""
get_scale(::NoScaling, acq_data::AcquisitionInfo, x₀, 𝒜) = 1.0

function get_scale(scaling::QuantileScaling, acq_data::AcquisitionInfo, x₀, 𝒜)
    # The subsample's maximum is not the image's.
    scaling.p == 1 && return _max_abs(x₀)
    v = _abs_subsample(x₀)
    return _quantile_select!(v, scaling.p)
end

function get_scale(::BartScaling, acq_data::AcquisitionInfo, x₀, 𝒜)
    v = _abs_subsample(x₀)
    max = _max_abs(x₀)
    median = _quantile_select!(v, 0.5)
    p90 = _quantile_select!(v, 0.9)
    return ((max - p90) < 2 * (p90 - median)) ? p90 : max
end

get_scale(::MaxScaling, acq_data::AcquisitionInfo, x₀, 𝒜) = _max_abs(x₀)

get_scale(::StdScaling, acq_data::AcquisitionInfo, x₀, 𝒜) = std(vec(unname(x₀)))

get_scale(::NoiseLevelScaling, acq_data::AcquisitionInfo, x₀, 𝒜) = _noise_level(unname(x₀))

# Any other storage (a device array): the neighbour differences are formed where `x` lives, and
# only the subsample the median reads is copied to the host. Same subsample, same median as below.
function _noise_level(x::AbstractArray)
    h = size(x, 1) ÷ 2
    h >= 1 || return zero(real(eltype(x)))
    a = reshape(x, size(x, 1), :)
    d = vec(view(a, 2:2:(2h), :) .- view(a, 1:2:(2h - 1), :))
    sub = Array(view(d, _subsample_indices(length(d))))
    parts = vcat(abs.(real.(sub)), abs.(imag.(sub)))
    return _quantile_select!(parts, 0.5) / 0.6745
end

function _noise_level(x::Array)
    h = size(x, 1) ÷ 2
    h >= 1 || return zero(real(eltype(x)))
    ncols = length(x) ÷ size(x, 1)
    a = reshape(x, size(x, 1), ncols)
    idx = _subsample_indices(h * ncols)
    parts = Vector{real(eltype(x))}(undef, 2 * length(idx))
    for (n, k) in enumerate(idx)
        i, j = mod1(k, h), cld(k, h)
        d = a[2i, j] - a[2i - 1, j]
        parts[2n - 1] = abs(real(d))
        parts[2n] = abs(imag(d))
    end
    # A detail's real and imaginary parts each have `√2` times the per-part deviation of `x₀`,
    # which is the complex standard deviation `√E|z|²` of `x₀`'s noise.
    return _quantile_select!(parts, 0.5) / 0.6745
end

function get_scale(::MeasurementBasedScaling, acquisition_data::AcquisitionInfo, x₀, 𝒜)
    return norm(acquisition_data.kspace_data, 1) / length(acquisition_data.kspace_data)
end

function get_scale(scaling::KSpaceNormScaling, acquisition_data::AcquisitionInfo, x₀, 𝒜)
    return norm(acquisition_data.kspace_data) / scaling.target
end

# `𝒜'` is `𝒜`'s inverse where its Fourier transform is unnormalized, not its adjoint, so the probe
# applies it rather than reading `‖𝒜z‖²`. A ±1 probe has `‖z‖² = N`.
function get_scale(::SystemMatrixBasedScaling, acq_data::AcquisitionInfo, x₀, 𝒜)
    x = unname(x₀)
    rng = Random.Xoshiro(0x0005ca1e)
    R = real(eltype(x))
    signs = Array{R}(undef, size(x))
    z = similar(x)
    total = zero(R)
    for _ in 1:SYSTEM_MATRIX_PROBES
        copyto!(z, Random.rand!(rng, signs, (-one(R), one(R))))
        total += real(dot(z, unname(𝒜' * (𝒜 * z))))
    end
    return total / (SYSTEM_MATRIX_PROBES * length(x))
end

const SYSTEM_MATRIX_PROBES = 4

get_scale(scaling::FixedScaling, acq_data::AcquisitionInfo, x₀, 𝒜) = scaling.scale

# The subsample the quantile rules read: at most `SCALING_SUBSAMPLE` elements at a prime stride
# that does not divide the length. A stride sharing a factor with an axis aliases with it and
# samples a few planes of the grid (on a 128³ volume, stride 32 visits four readout positions and
# put the 99th percentile 40 % high); a prime that divides none of the axes visits every residue of
# every axis.
const SCALING_SUBSAMPLE = 2^18

_subsample_indices(n::Int) = 1:(n <= SCALING_SUBSAMPLE ? 1 : _coprime_prime_stride(cld(n, SCALING_SUBSAMPLE), n)):n

# The smallest odd prime at least `k` that does not divide `n`.
function _coprime_prime_stride(k::Int, n::Int)
    p = max(3, isodd(k) ? k : k + 1)
    while !_is_odd_prime(p) || n % p == 0
        p += 2
    end
    return p
end

function _is_odd_prime(p::Int)
    for d in 3:2:isqrt(p)
        p % d == 0 && return false
    end
    return true
end

# `|x|` at the subsample indices, on the CPU.
function _abs_subsample(x₀)
    x = vec(unname(x₀))
    return convert(Array, abs.(view(x, _subsample_indices(length(x)))))
end

# `quantile(v, p)` (Statistics' default definition, linear interpolation between order statistics)
# by selecting the two order statistics it interpolates, which is linear in `length(v)`, instead of
# sorting the range between the smallest and largest `p` asked for. `v` is reordered.
function _quantile_select!(v::AbstractVector, p::Real)
    n = length(v)
    n == 1 && return v[1]
    h = fma(n, p, oftype(p, 1 - p))
    j = clamp(trunc(Int, h), 1, n - 1)
    γ = clamp(h - j, 0, 1)
    a, b = partialsort!(v, j:(j + 1))
    return (isfinite(a) && isfinite(b) && a ≈ b) ? a + γ * (b - a) : (1 - γ) * a + γ * b
end
