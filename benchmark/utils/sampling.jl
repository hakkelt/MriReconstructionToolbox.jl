# Sampling patterns for the benchmark case catalog. Every pattern takes an explicit RNG, so a case
# is the same on every machine and in every process (MRT's `create_sampling_pattern` draws from the
# global RNG, which is why it is not used here).

"""
    vd_lines(rng, n, nlines; acs) -> Vector{Int}

`nlines` sorted phase-encode indices out of `1:n`: the `acs` central lines (the autocalibration
region, centred on `n ÷ 2 + 1`), plus `nlines - acs` further lines drawn without replacement with a
variable density `(1 - |k|/kmax)² + 0.05` that favours the centre of k-space.
"""
function vd_lines(rng::AbstractRNG, n::Int, nlines::Int; acs::Int)
    0 <= acs <= nlines <= n || throw(ArgumentError("need 0 ≤ acs ≤ nlines ≤ n, got $acs, $nlines, $n"))
    c = n ÷ 2 + 1
    centre = (c - acs ÷ 2):(c - acs ÷ 2 + acs - 1)
    rest = setdiff(1:n, centre)
    w = [(1 - abs(i - c) / (n / 2))^2 + 0.05 for i in rest]
    return sort!(vcat(collect(centre), _weighted_pick(rng, rest, w, nlines - acs)))
end

# Weighted sampling without replacement (Efraimidis-Spirakis): the k largest `u^(1/w)` keys.
function _weighted_pick(rng::AbstractRNG, items, w, k::Int)
    keys = [rand(rng)^(1 / wi) for wi in w]
    return items[partialsortperm(keys, 1:k; rev = true)]
end

"""
    line_mask(nx, lines, ny) -> BitMatrix

The `(nx, ny)` Cartesian mask that acquires every readout sample of the phase-encode `lines`.
"""
function line_mask(nx::Int, lines::AbstractVector{<:Integer}, ny::Int)
    m = falses(nx, ny)
    m[:, lines] .= true
    return m
end

"""
    vd_mask_2d(rng, ny, nz; R, calib) -> BitMatrix

A `(ny, nz)` phase-encode mask for a 3D Cartesian encode at acceleration `R`: a fully sampled
`calib × calib` centre plus points drawn without replacement with a radial variable density
`(1 - r)² + 0.02` (`r` the normalised distance from the centre). A variable-density random pattern,
not a Poisson-disc one: the point is incoherent aliasing at a fixed sample count, which this gives.
"""
function vd_mask_2d(rng::AbstractRNG, ny::Int, nz::Int; R::Real, calib::Int)
    cy, cz = ny ÷ 2 + 1, nz ÷ 2 + 1
    m = falses(ny, nz)
    cyr = (cy - calib ÷ 2):(cy - calib ÷ 2 + calib - 1)
    czr = (cz - calib ÷ 2):(cz - calib ÷ 2 + calib - 1)
    m[intersect(cyr, 1:ny), intersect(czr, 1:nz)] .= true
    target = round(Int, ny * nz / R)
    rest = [I for I in CartesianIndices(m) if !m[I]]
    w = [(1 - min(1.0, hypot((I[1] - cy) / (ny / 2), (I[2] - cz) / (nz / 2))))^2 + 0.02 for I in rest]
    k = max(0, target - count(m))
    m[_weighted_pick(rng, rest, w, k)] .= true
    return m
end

"""
    per_frame_lines(rng, n, nframes; centre, random) -> Vector{Vector{Int}}

For each frame, the `centre` central phase-encode lines plus `random` further lines drawn uniformly
without replacement from the rest, independently per frame. Every frame acquires the same number
of lines, so the undersampled k-space is one dense array; the aliasing is incoherent across frames,
which is what a temporal regularizer exploits.
"""
function per_frame_lines(rng::AbstractRNG, n::Int, nframes::Int; centre::Int, random::Int)
    c = n ÷ 2 + 1
    fixed = (c - centre ÷ 2):(c - centre ÷ 2 + centre - 1)
    rest = setdiff(1:n, fixed)
    return [sort!(vcat(collect(fixed), rest[randperm(rng, length(rest))[1:random]])) for _ in 1:nframes]
end

"""
    GOLDEN_ANGLE

The golden-angle increment for radial spokes (lines, so angles are taken modulo π):
`π (√5 - 1) / 2 ≈ 111.25°`. The same increment as MRT's `GoldenAngle()`.
"""
const GOLDEN_ANGLE = π * (sqrt(5) - 1) / 2

"""
    golden_angle_radial(nsamples, nspokes; first_spoke = 0) -> Array{Float32, 3}

`(2, nsamples, nspokes)` radial trajectory in cycles/sample (`[-0.5, 0.5)`, the NFFT.jl convention
MRT and the other toolkits share), coordinate 1 along `x`. Spoke `j` is at angle
`(first_spoke + j - 1) · GOLDEN_ANGLE`; passing `first_spoke = (t - 1) nspokes` continues the
sequence frame after frame, which is how a golden-angle cine rotates its trajectory.
"""
function golden_angle_radial(nsamples::Int, nspokes::Int; first_spoke::Int = 0)
    r = (-(nsamples ÷ 2):(nsamples - nsamples ÷ 2 - 1)) ./ nsamples
    traj = Array{Float32}(undef, 2, nsamples, nspokes)
    for j in 1:nspokes
        θ = mod((first_spoke + j - 1) * GOLDEN_ANGLE, π)
        s, c = sincos(θ)
        for i in 1:nsamples
            traj[1, i, j] = r[i] * c
            traj[2, i, j] = r[i] * s
        end
    end
    return traj
end

"""
    ramp_dcf(traj) -> Array{Float32}

Radial ramp density compensation `|k|`, floored at a quarter sample so the k-space centre is not
discarded, and scaled to a maximum of 1. The scale is irrelevant to every score here
(reconstructions are compared after magnitude alignment); the shape is what gridding needs.
"""
function ramp_dcf(traj::AbstractArray{<:Real})
    nsamples = size(traj, 2)
    w = dropdims(sqrt.(sum(abs2, traj; dims = 1)); dims = 1)
    w = max.(w, 0.25f0 / nsamples)
    return Float32.(w ./ maximum(w))
end
