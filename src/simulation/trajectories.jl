## Non-Cartesian trajectory generators.
##
## All coordinates follow the NFFT.jl / NFFTOperators convention: normalized to `[-0.5, 0.5)`
## (open at +0.5). Returned trajectories are `NamedDimsArray`s with the coordinate axis first
## (named `:coord`, one of the two names `NonCartesianAcquisitionInfo` accepts) followed by the
## sample-layout dimensions, exactly as `NonCartesianAcquisitionInfo`/`get_encoding_operator`
## expect (see `src/acquisition_data/noncartesian_acquisition_info.jl`).

"""
    _open_range(lo, hi, n) -> Vector{Float32}

`n` points strictly inside `(lo, hi)`, evenly spaced as if `range(lo, hi; length = n + 2)` with
the two endpoints dropped. Used to keep radial readout coordinates inside NFFT.jl's half-open
`[-0.5, 0.5)` domain even at the extreme sample.
"""
_open_range(lo::Real, hi::Real, n::Integer) = Float32.(range(lo, hi; length = n + 2)[2:(end - 1)])

"""
    RadialOrdering

Abstract supertype of the spoke-to-spoke angle increments [`radial_trajectory`](@ref) accepts:
[`LinearOrdering`](@ref), [`GoldenAngle`](@ref) and [`TinyGoldenAngle`](@ref).
"""
abstract type RadialOrdering end

"""
    LinearOrdering() <: RadialOrdering

Spokes spaced uniformly over `[0, π)` — sequential (linear) profile order. Every spoke count has
to be acquired in full before k-space is covered evenly.
"""
struct LinearOrdering <: RadialOrdering end

"""
    GoldenAngle() <: RadialOrdering

Successive spokes rotated by the golden angle `π · (√5 - 1)/2 ≈ 111.246°` (Winkelmann et al., "An
optimal radial profile order based on the golden ratio for time-resolved MRI", IEEE Trans. Med.
Imaging 26(1):68-76, 2007). Any prefix of the sequence covers k-space near-uniformly, so one
trajectory can be retrospectively under-sampled to any spoke count without regenerating it.
"""
struct GoldenAngle <: RadialOrdering end

"""
    TinyGoldenAngle(index::Int = 2) <: RadialOrdering

The `index`-th member of the tiny-golden-angle family, `π / (φ + index - 1)` with `φ = (1 + √5)/2`
(Wundrak et al., Magn. Reson. Med. 2015 / IEEE Trans. Med. Imaging 2015). Consecutive spokes stay
closer together than with the standard golden angle, which is what sliding-window / view-sharing
reconstructions want, while the golden ratio's incremental-coverage property is retained.

`index = 1` is the standard golden angle, i.e. `TinyGoldenAngle(1)` and [`GoldenAngle`](@ref)
produce the same spokes; the default `2` is the first genuinely *tiny* member.
"""
struct TinyGoldenAngle <: RadialOrdering
    index::Int
    function TinyGoldenAngle(index::Integer = 2)
        @argcheck index >= 1 "the tiny-golden-angle index must be >= 1"
        return new(Int(index))
    end
end

"""
    radial_trajectory(nsamples::Int, nspokes::Int;
        ordering::RadialOrdering = GoldenAngle(), extent::Real = 0.5)
    -> NamedDimsArray{(:coord, :sample, :spoke)}

Generate a 2D radial (projection-reconstruction) k-space trajectory: `nspokes` straight lines
through the k-space center, each sampled at `nsamples` points spanning `(-extent, extent)`
(normalized units, NFFT.jl convention).

`ordering` controls the spoke-to-spoke angle increment (spokes are lines, so angles are taken
modulo `π`): [`LinearOrdering`](@ref), [`GoldenAngle`](@ref) (the default) or
[`TinyGoldenAngle`](@ref), which carries its own family index.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :spoke)`.
"""
function radial_trajectory(
        nsamples::Int, nspokes::Int;
        ordering::RadialOrdering = GoldenAngle(), extent::Real = 0.5,
    )
    @argcheck nsamples > 0 "nsamples must be positive"
    @argcheck nspokes > 0 "nspokes must be positive"
    @argcheck 0 < extent <= 0.5 "extent must be in (0, 0.5]"
    angles = _radial_spoke_angles(nspokes, ordering)
    r = _open_range(-extent, extent, nsamples)
    traj = Array{Float32}(undef, 2, nsamples, nspokes)
    for (s, θ) in enumerate(angles)
        traj[1, :, s] .= r .* Float32(cos(θ))
        traj[2, :, s] .= r .* Float32(sin(θ))
    end
    return NamedDimsArray{(:coord, :sample, :spoke)}(traj)
end

_radial_spoke_angles(nspokes::Integer, ::LinearOrdering) = range(0, π; length = nspokes + 1)[1:nspokes]
_radial_spoke_angles(nspokes::Integer, ::GoldenAngle) = _angle_sequence(nspokes, π * (sqrt(5) - 1) / 2)
_radial_spoke_angles(nspokes::Integer, o::TinyGoldenAngle) =
    _angle_sequence(nspokes, π / ((1 + sqrt(5)) / 2 + o.index - 1))

_angle_sequence(nspokes::Integer, step::Real) = [mod((n - 1) * step, π) for n in 1:nspokes]

"""
    stack_of_stars_trajectory(nsamples::Int, nspokes::Int, npartitions::Int; kwargs...)
    -> NamedDimsArray{(:coord, :sample, :spoke, :partition)}

3D "stack-of-stars" trajectory: the 2D radial in-plane pattern from
[`radial_trajectory`](@ref) (`nsamples`, `nspokes` and `kwargs` are forwarded to it) is repeated
identically at every one of `npartitions` Cartesian partition-encoding (`kz`) positions —
radial sampling in-plane, conventional Cartesian phase encoding along the partition direction.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :spoke, :partition)`.
"""
function stack_of_stars_trajectory(nsamples::Int, nspokes::Int, npartitions::Int; kwargs...)
    @argcheck npartitions > 0 "npartitions must be positive"
    radial = radial_trajectory(nsamples, nspokes; kwargs...)
    raw_radial = NamedDims.unname(radial)
    kz = npartitions == 1 ? Float32[0] : Float32.(range(-0.5, 0.5; length = npartitions + 1)[1:npartitions])
    traj = Array{Float32}(undef, 3, nsamples, nspokes, npartitions)
    for p in 1:npartitions
        traj[1:2, :, :, p] .= raw_radial
        traj[3, :, :, p] .= kz[p]
    end
    return NamedDimsArray{(:coord, :sample, :spoke, :partition)}(traj)
end

"""
    kooshball_trajectory(nsamples::Int, nspokes::Int; extent::Real = 0.5)
    -> NamedDimsArray{(:coord, :sample, :spoke)}

Full 3D radial ("kooshball") trajectory: `nspokes` spokes through the k-space center, each
sampled at `nsamples` points spanning `(-extent, extent)`, with spoke directions distributed
quasi-uniformly over the sphere using the multidimensional golden-means ordering of
Chan et al., "Temporal stability of adaptive 3D radial MRI using multidimensional golden means",
Magn. Reson. Med. 61(2):354-363, 2009 (golden means Φ₁ ≈ 0.4656, Φ₂ ≈ 0.6823).

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :spoke)`.
"""
function kooshball_trajectory(nsamples::Int, nspokes::Int; extent::Real = 0.5)
    @argcheck nsamples > 0 "nsamples must be positive"
    @argcheck nspokes > 0 "nspokes must be positive"
    @argcheck 0 < extent <= 0.5 "extent must be in (0, 0.5]"
    Φ1, Φ2 = 0.4656, 0.6823
    r = _open_range(-extent, extent, nsamples)
    traj = Array{Float32}(undef, 3, nsamples, nspokes)
    for n in 1:nspokes
        h = mod((n - 1) * Φ2, 1.0)
        z = 1 - 2h
        θ = acos(clamp(z, -1.0, 1.0))
        ϕ = 2π * mod((n - 1) * Φ1, 1.0)
        dx, dy, dz = sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)
        traj[1, :, n] .= r .* Float32(dx)
        traj[2, :, n] .= r .* Float32(dy)
        traj[3, :, n] .= r .* Float32(dz)
    end
    return NamedDimsArray{(:coord, :sample, :spoke)}(traj)
end

"""
    SpiralVariant

Abstract supertype of the radial growth laws [`spiral_trajectory`](@ref) accepts:
[`Archimedean`](@ref) and [`VariableDensity`](@ref).
"""
abstract type SpiralVariant end

"""
    Archimedean() <: SpiralVariant

Constant angular velocity, `ρ(t) = extent · t` — the classic Archimedean spiral, with uniform
radial sample density.
"""
struct Archimedean <: SpiralVariant end

"""
    VariableDensity(exponent::Real = 2.0) <: SpiralVariant

`ρ(t) = extent · t^exponent`. An `exponent` above 1 makes `dρ/dt → 0` as `t → 0`, i.e. slower
initial radial growth, which oversamples the k-space center relative to the Archimedean spiral (a
common compressed-sensing spiral design) at the cost of undersampling the periphery. `1` reduces to
[`Archimedean`](@ref); below 1 does the reverse — denser periphery, sparser center.
"""
struct VariableDensity <: SpiralVariant
    exponent::Float64
    function VariableDensity(exponent::Real = 2.0)
        @argcheck exponent > 0 "the variable-density exponent must be positive"
        return new(Float64(exponent))
    end
end

"""
    spiral_trajectory(nsamples::Int, ninterleaves::Int;
        variant::SpiralVariant = Archimedean(), nturns::Real = 8, extent::Real = 0.5)
    -> NamedDimsArray{(:coord, :sample, :interleave)}

2D spiral k-space trajectory: `ninterleaves` rotated copies of a single spiral arm, each sampled
at `nsamples` points from the k-space center out to `extent` (interleave `i` is the base arm
rotated by `2π(i-1)/ninterleaves`).

`variant` selects the radial growth law — [`Archimedean`](@ref) or [`VariableDensity`](@ref),
which carries its own exponent. `t` runs linearly over `[0, 1]` along the arm, and sample density
near radius `ρ` scales with `1/(dρ/dt)` there.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :interleave)`.
"""
function spiral_trajectory(
        nsamples::Int, ninterleaves::Int;
        variant::SpiralVariant = Archimedean(), nturns::Real = 8, extent::Real = 0.5,
    )
    @argcheck nsamples > 1 "nsamples must be > 1"
    @argcheck ninterleaves > 0 "ninterleaves must be positive"
    @argcheck 0 < extent <= 0.5 "extent must be in (0, 0.5]"
    # `t` excludes 1: at t=1, ρ=extent and a spoke aligned with an axis would land exactly on
    # the NFFT domain boundary `±0.5`, which is excluded (`[-0.5, 0.5)`).
    t = range(0, 1; length = nsamples + 1)[1:nsamples]
    ρ = _spiral_radius(variant, extent, t)
    ϕbase = Float32.(2 * Float64(π) * nturns .* t)
    traj = Array{Float32}(undef, 2, nsamples, ninterleaves)
    for i in 1:ninterleaves
        ϕ0 = Float32(2π * (i - 1) / ninterleaves)
        traj[1, :, i] .= ρ .* cos.(ϕbase .+ ϕ0)
        traj[2, :, i] .= ρ .* sin.(ϕbase .+ ϕ0)
    end
    return NamedDimsArray{(:coord, :sample, :interleave)}(traj)
end

_spiral_radius(::Archimedean, extent::Real, t) = Float32.(extent .* t)
_spiral_radius(v::VariableDensity, extent::Real, t) = Float32.(extent .* t .^ v.exponent)
