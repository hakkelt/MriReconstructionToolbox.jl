# Data scaling and noise for the benchmark case catalog.

"""
    norm_ksp(k) -> k scaled to unit RMS (‖k‖ = √length)

Every toolkit's λ is defined relative to the data-term scale, so a λ calibrated on a synthetic case
only transfers to real scanner data if both k-spaces are put on the same scale first. Applied to
every case, synthetic and real, before it is handed out. The scale factor is taken in `eltype(k)`,
so a `ComplexF32` k-space is not promoted by a `Float64` factor.
"""
norm_ksp(k) = k .* real(eltype(k))(sqrt(length(k)) / norm(k))

"""
    add_noise(k; snr_db = 30, seed = 1) -> k + complex Gaussian noise

Additive complex white noise at `snr_db` relative to the RMS of `k`. The synthetic phantoms are
noiseless, so without this every toolkit reconstructs them near-perfectly and TV / wavelet
regularisation only ever hurts: there is no non-trivial optimal λ to calibrate. Seeded, so
calibration and timing runs see the same realisation. Drawn in `ComplexF64` and converted, so the
realisation is the same sequence whichever precision is under test.
"""
function add_noise(k; snr_db::Real = 30, seed::Integer = 1)
    rng = MersenneTwister(seed)
    rms = norm(k) / sqrt(length(k))
    σ = rms * 10^(-snr_db / 20) / sqrt(2)
    return k .+ eltype(k).(σ .* randn(rng, ComplexF64, size(k)))
end

"""
    nrmse(x, xref)

`‖x - xref‖ / ‖xref‖` over all entries.
"""
nrmse(x, xref) = norm(vec(x) .- vec(xref)) / norm(vec(xref))

"""
    mag_nrmse(est, ref)

NRMSE of the magnitudes after scaling `|est|` to `‖|ref|‖`. Reconstructions differ by a global
scale (unnormalised maps, DCF scaling) and real ones by a receive phase the magnitude reference
lacks, neither of which is an error.
"""
function mag_nrmse(est, ref)
    a = abs.(est)
    r = abs.(ref)
    return nrmse(a .* (norm(r) / norm(a)), r)
end
