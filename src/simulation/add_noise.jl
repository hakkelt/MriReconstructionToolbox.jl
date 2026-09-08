"""
    add_noise(kspace_data; snr_db=nothing, noise_std=nothing, rng=Random.default_rng())
    add_noise(acq_info::AcquisitionInfo; snr_db=nothing, noise_std=nothing, rng=Random.default_rng())

Add complex white Gaussian noise to simulated k-space data.

Exactly one of `snr_db` or `noise_std` must be given (passing both, or neither, is an
`ArgumentError`):

- `snr_db`: target signal-to-noise ratio in dB, defined relative to the RMS of the data,
  `σ = rms(data) * 10^(-snr_db / 20)`. This is the convention already used by the
  `add_noise` helper in `benchmark/comparison/scripts/_toolkits.jl` and matches how SNR is
  reported for the pseudo-replica method (Robson et al., MRM 60:895-907, 2008; see
  [`pseudo_replica`](@ref)).
- `noise_std`: the absolute standard deviation `σ` of the complex noise, split evenly between
  the real and imaginary parts (each `~ 𝒩(0, (σ/√2)²)`, so that `std(noise) == σ`). This
  matches the `noise_std` keyword already used by [`pseudo_replica`](@ref) and BART's
  `noise -n <stdev>` command (a fixed absolute standard deviation rather than a relative SNR).

`kspace_data` may be a plain `AbstractArray` or a `NamedDimsArray` (dimension names are
preserved). Passing an `AcquisitionInfo` (`CartesianAcquisitionInfo` or
`NonCartesianAcquisitionInfo`) adds noise to `acq_info.kspace_data` and returns a *copy* of
`acq_info` with the noisy k-space (via the existing copy-constructor pattern;
`acq_info` itself is left untouched). `acq_info.kspace_data` must not be `nothing`.

`rng` fixes the noise realization for reproducibility, following the `rng` keyword already used
throughout the package (e.g. [`pseudo_replica`](@ref), `LocallyLowRank`).

# Examples
```julia
noisy_kspace = add_noise(kspace_data; snr_db = 20)
noisy_acq = add_noise(acq; noise_std = 0.01)
```
"""
function add_noise(
        kspace_data::AbstractArray;
        snr_db::Union{Real, Nothing} = nothing,
        noise_std::Union{Real, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
    )
    σ = _resolve_noise_std(kspace_data, snr_db, noise_std)
    raw = kspace_data isa NamedDimsArray ? NamedDims.unname(kspace_data) : kspace_data
    T = eltype(raw)
    noise = (randn(rng, real(T), size(raw)) .+ im .* randn(rng, real(T), size(raw))) .* (real(T)(σ / sqrt(2)))
    noisy = raw .+ noise
    return kspace_data isa NamedDimsArray ? NamedDimsArray{dimnames(kspace_data)}(noisy) : noisy
end

function add_noise(
        acq_info::AcquisitionInfo;
        snr_db::Union{Real, Nothing} = nothing,
        noise_std::Union{Real, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
    )
    @argcheck !isnothing(acq_info.kspace_data) "acq_info.kspace_data must not be `nothing`"
    noisy_kspace = add_noise(acq_info.kspace_data; snr_db, noise_std, rng)
    return AcquisitionInfo(acq_info; kspace_data = noisy_kspace)
end

function _resolve_noise_std(kspace_data, snr_db, noise_std)
    @argcheck !isnothing(snr_db) ⊻ !isnothing(noise_std) "exactly one of `snr_db` or `noise_std` must be provided"
    if !isnothing(noise_std)
        return noise_std
    end
    raw = kspace_data isa NamedDimsArray ? NamedDims.unname(kspace_data) : kspace_data
    rms = norm(raw) / sqrt(length(raw))
    return rms * 10.0^(-snr_db / 20)
end
