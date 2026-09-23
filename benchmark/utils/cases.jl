# The benchmark case catalog: one toolkit-neutral description of every reconstruction problem the
# MRT harness (`benchmark/run.jl`) and the comparison suite (`benchmark/comparison/`) time.
#
# A case holds plain arrays in fixed layouts (see `BenchCase`), generated only from
# GeometricMedicalPhantoms, FFTW and seeded RNGs. MRT is used for exactly two things: simulating
# non-Cartesian k-space (its NFFT, pinned at a high-accuracy operating point) and, for real data,
# estimating ESPIRiT maps. Everything toolkit-specific -- MRT's `AcquisitionInfo`, BART's and
# SigPy's layouts -- is a converter from these arrays, never a second source of data.

"""
    BenchCase

One benchmark problem. Array layouts, by `family`:

| family | `reference` | `smaps` | `kspace` | `mask` |
|---|---|---|---|---|
| `:single_slice` | `(x, y)` | `(x, y, coil)` | Cartesian `(kx, ky, coil)`, radial `(sample, spoke, coil)` | `(kx, ky)` |
| `:multislice` | `(x, y, slice)` | `(x, y, coil, slice)` | `(kx, ky, coil, slice)` | `(kx, ky)`, shared by every slice |
| `:volume` | `(x, y, z)` | `(x, y, z, coil)` | `(kx, ky, kz, coil)` | `(kx, ky, kz)` |
| `:cine` | `(x, y, time)` | `(x, y, coil)` | Cartesian `(kx, ky, coil, time)`, radial `(sample, spoke, coil, time)` | `(kx, ky, time)` |

- Cartesian `kspace` is **zero-filled** on the full grid (unsampled entries are zero) and centred
  (`centred_fft`); `mask` says which entries were sampled. Radial cases have `mask === nothing` and
  carry `traj` (`(2, sample, spoke)`, cycles/sample, shared by every frame) and a ramp `dcf`
  (`(sample, spoke)`).
- `smaps === nothing` for a single-channel case; its `kspace` still has a coil axis of length 1.
- k-space is unit-RMS normalised (`norm_ksp`) and carries complex Gaussian noise at
  `MRT_BENCH_SNR_DB` (30 dB) before undersampling, so λ calibrated on one case transfers to another.
- `reference` is the ground truth: the phantom for synthetic cases, the fully sampled
  reconstruction for real ones (which are undersampled retrospectively).
- `heavy` cases (the volume and the cine series) are timed once after one warm-up, not min-of-3.
- `analogue` is the id of the synthetic case a real case mirrors; its λ is used for the real one.
"""
Base.@kwdef struct BenchCase
    id::String
    family::Symbol
    trajectory::Symbol
    reference::Array{ComplexF32}
    smaps::Union{Nothing, Array{ComplexF32}}
    kspace::Array{ComplexF32}
    mask::Union{Nothing, BitArray} = nothing
    traj::Union{Nothing, Array{Float32}} = nothing
    dcf::Union{Nothing, Array{Float32}} = nothing
    image_size::Tuple{Vararg{Int}}
    heavy::Bool = false
    real::Bool = false
    source::String = "synthetic"
    analogue::String = id
    seed::Int = 0
end

function Base.show(io::IO, c::BenchCase)
    print(io, "BenchCase(\"", c.id, "\", ", c.family, ", ", c.trajectory, ", image ", join(c.image_size, "×"))
    print(io, ", ", ncoils(c), " coil", ncoils(c) == 1 ? "" : "s")
    c.family === :multislice && print(io, ", ", size(c.reference, 3), " slices")
    c.family === :cine && print(io, ", ", size(c.reference, 3), " frames")
    c.mask === nothing || print(io, ", R = ", round(acceleration(c); digits = 2))
    c.traj === nothing || print(io, ", ", size(c.traj, 3), " spokes × ", size(c.traj, 2), " samples")
    return print(io, ", ", c.source, ")")
end

"""
    ncoils(c::BenchCase) -> Int
"""
ncoils(c::BenchCase) = size(c.kspace, c.family === :volume ? 4 : 3)

"""
    acceleration(c::BenchCase) -> Float64

Cartesian undersampling factor: grid points per sampled point (averaged over frames for cine).
"""
acceleration(c::BenchCase) = c.mask === nothing ? NaN : length(c.mask) / count(c.mask)

"""
    zero_filled(c::BenchCase) -> Array{ComplexF32}

The zero-filled k-space in the case's own layout (see [`BenchCase`](@ref)). Cartesian cases store it
this way already; this is the accessor the toolkit converters use, so the layout can change in one
place.
"""
zero_filled(c::BenchCase) = c.kspace

# ---------------------------------------------------------------- catalog

"""
    SYNTHETIC_CASES

Every synthetic case id, in the order the harness runs them.
"""
const SYNTHETIC_CASES = (
    "shepp_logan_2d_1ch_cartesian",
    "shepp_logan_2d_8ch_cartesian",
    "shepp_logan_2d_8ch_radial",
    "shepp_logan_multislice_8ch_cartesian",
    "shepp_logan_3d_8ch_cartesian",
    "torso_cine_8ch_cartesian",
    "torso_cine_8ch_radial",
)

"""
    REAL_CASES

Real-data analogues, `real id => synthetic analogue`. See `real_data.jl` for the datasets.
"""
const REAL_CASES = (
    "real_2d_1ch_cartesian" => "shepp_logan_2d_1ch_cartesian",
    "real_2d_multichannel_cartesian" => "shepp_logan_2d_8ch_cartesian",
    "real_2d_multichannel_radial" => "shepp_logan_2d_8ch_radial",
    "real_multislice_multichannel_cartesian" => "shepp_logan_multislice_8ch_cartesian",
    "real_3d_multichannel_cartesian" => "shepp_logan_3d_8ch_cartesian",
    "real_cine_multichannel_cartesian" => "torso_cine_8ch_cartesian",
    "real_cine_multichannel_radial" => "torso_cine_8ch_radial",
)

"""
    case_ids(; real = env_flag("MRT_BENCH_REAL_DATA"), synthetic = true) -> Vector{String}
"""
function case_ids(; real::Bool = env_flag("MRT_BENCH_REAL_DATA"), synthetic::Bool = true)
    ids = String[]
    synthetic && append!(ids, SYNTHETIC_CASES)
    real && append!(ids, first.(REAL_CASES))
    return ids
end

"""
    filter_case_ids(ids, patterns) -> Vector{String}

`ids` whose name contains any of `patterns` (case-insensitive substrings); all of them when
`patterns` is `nothing` or empty.
"""
filter_case_ids(ids, ::Nothing) = collect(ids)
filter_case_ids(ids, patterns) =
    isempty(patterns) ? collect(ids) : [i for i in ids if any(p -> occursin(lowercase(p), lowercase(i)), patterns)]

"""
    small_mode() -> Bool

`MRT_BENCH_SMALL=1` shrinks every case (32² images, 3 slices, 32³, 8 frames, 4 coils) for smoke
runs on a login node and for the TestItem. Results are tagged with it and never compared with
full-size ones.
"""
small_mode() = env_flag("MRT_BENCH_SMALL")

"""
    cine_frames() -> Int

Frame count of the cine cases: `MRT_BENCH_CINE_FRAMES`, default 30 (one cardiac cycle); 8 in small
mode.
"""
cine_frames() = small_mode() ? 8 : parse(Int, get(ENV, "MRT_BENCH_CINE_FRAMES", "30"))

snr_db() = parse(Float64, get(ENV, "MRT_BENCH_SNR_DB", "30"))

function _dims()
    return small_mode() ?
        (n = 32, slices = 3, coils = 4, acs = 8, calib3d = 8, radial = (64, 16), cine_radial = (64, 9), cine_lines = (4, 4)) :
        (n = 128, slices = 12, coils = 8, acs = 16, calib3d = 24, radial = (256, 64), cine_radial = (256, 34), cine_lines = (8, 24))
end

const _CASE_CACHE = Dict{String, BenchCase}()

"""
    get_case(id) -> BenchCase

Build (or return the memoised) case `id`, synthetic or real. Memoised per process and per
`small_mode()`, since the heavy cases take seconds to generate and every method of a case reuses it.
"""
function get_case(id::AbstractString; pattern::Symbol = :catalog)
    key = string(id, small_mode() ? "#small" : "", "#", cine_frames(), "#", snr_db(), "#", pattern)
    return get!(_CASE_CACHE, key) do
        pattern === :catalog || return _build_synthetic(String(id); pattern)
        id in SYNTHETIC_CASES && return _build_synthetic(String(id))
        for (rid, analogue) in REAL_CASES
            rid == id && return load_real_case(rid, analogue)
        end
        throw(ArgumentError("unknown case $(repr(id)); known: $(join(case_ids(real = true), ", "))"))
    end
end

# Stable per-case seed, so each case's sampling pattern and noise are independent of the others and
# of the order they are built in.
_case_seed(id) = Int(sum(Int(c) * 31^(i % 7) for (i, c) in enumerate(id)) % 100_000)

"""
    regular_lines(n; R = 2, acs) -> Vector{Int}

Every `R`-th phase encode plus a contiguous block of `acs` central lines: the regular pattern
GRAPPA needs (one kernel per missing-line offset cannot be fitted to a random pattern).
"""
function regular_lines(n::Int; R::Int = 2, acs::Int)
    c = n ÷ 2 + 1
    return sort!(union(1:R:n, (c - acs ÷ 2):(c - acs ÷ 2 + acs - 1)))
end

function _build_synthetic(id::String; pattern::Symbol = :catalog)
    d = _dims()
    n = d.n
    seed = _case_seed(id)
    rng = MersenneTwister(seed)
    if pattern === :regular
        # The same phantom, maps and noise as the catalog case, sampled every other phase encode
        # plus a 24-line (small: 12) calibration block, for GRAPPA.
        id in ("shepp_logan_2d_8ch_cartesian", "shepp_logan_multislice_8ch_cartesian") ||
            throw(ArgumentError("no regular-pattern variant of $id"))
        c = _build_synthetic(id)
        lines = regular_lines(n; R = 2, acs = small_mode() ? 12 : 24)
        ref, maps = c.reference, c.smaps
        kfull = c.family === :multislice ? centred_fft(reshape(ref, n, n, 1, :) .* maps, (1, 2)) : centred_fft(ref .* maps, (1, 2))
        return _finish_cartesian(id * "__regular", c.family, ref, maps, kfull, line_mask(n, lines, n); seed)
    end
    if id == "shepp_logan_2d_1ch_cartesian"
        img = shepp_logan_2d(n)
        kfull = reshape(centred_fft(img, (1, 2)), n, n, 1)
        lines = vd_lines(rng, n, round(Int, n / 2.5); acs = d.acs)
        return _finish_cartesian(id, :single_slice, img, nothing, kfull, line_mask(n, lines, n); seed)
    elseif id == "shepp_logan_2d_8ch_cartesian"
        img = shepp_logan_2d(n)
        maps = coil_maps_2d(n, n, d.coils)
        kfull = centred_fft(img .* maps, (1, 2))
        lines = vd_lines(rng, n, n ÷ 4; acs = d.acs)
        return _finish_cartesian(id, :single_slice, img, maps, kfull, line_mask(n, lines, n); seed)
    elseif id == "shepp_logan_2d_8ch_radial"
        img = shepp_logan_2d(n)
        maps = coil_maps_2d(n, n, d.coils)
        traj = golden_angle_radial(d.radial...)
        k = _nfft_forward(reshape(img .* maps, n, n, :), traj)                  # (sample, spoke, coil)
        return _finish_radial(id, :single_slice, img, maps, k, traj; seed)
    elseif id == "shepp_logan_multislice_8ch_cartesian"
        vol = shepp_logan_volume(n)
        zsel = round.(Int, range(n ÷ 4 + 1, 3n ÷ 4; length = d.slices))
        ref = vol[:, :, zsel]
        maps = permutedims(coil_maps_3d(n, n, n, d.coils)[:, :, zsel, :], (1, 2, 4, 3))   # (x, y, coil, slice)
        kfull = centred_fft(reshape(ref, n, n, 1, :) .* maps, (1, 2))
        lines = vd_lines(rng, n, n ÷ 4; acs = d.acs)
        return _finish_cartesian(id, :multislice, ref, maps, kfull, line_mask(n, lines, n); seed)
    elseif id == "shepp_logan_3d_8ch_cartesian"
        vol = shepp_logan_volume(n)
        maps = coil_maps_3d(n, n, n, d.coils)
        kfull = centred_fft(vol .* maps, (1, 2, 3))
        yz = vd_mask_2d(rng, n, n; R = 6, calib = d.calib3d)
        mask = BitArray(repeat(reshape(yz, 1, n, n), n, 1, 1))
        return _finish_cartesian(id, :volume, vol, maps, kfull, mask; seed, heavy = true)
    elseif id == "torso_cine_8ch_cartesian"
        nt = cine_frames()
        frames = torso_cine(n, nt)
        maps = coil_maps_2d(n, n, d.coils)
        kfull = centred_fft(reshape(frames, n, n, 1, nt) .* maps, (1, 2))     # (kx, ky, coil, time)
        lines = per_frame_lines(rng, n, nt; centre = d.cine_lines[1], random = d.cine_lines[2])
        mask = falses(n, n, nt)
        for t in 1:nt
            mask[:, lines[t], t] .= true
        end
        return _finish_cartesian(id, :cine, frames, maps, kfull, mask; seed, heavy = true)
    elseif id == "torso_cine_8ch_radial"
        nt = cine_frames()
        frames = torso_cine(n, nt)
        maps = coil_maps_2d(n, n, d.coils)
        traj = golden_angle_radial(d.cine_radial...)
        k = _nfft_forward(reshape(reshape(frames, n, n, 1, nt) .* maps, n, n, :), traj)
        k = reshape(k, size(traj, 2), size(traj, 3), d.coils, nt)              # (sample, spoke, coil, time)
        return _finish_radial(id, :cine, frames, maps, k, traj; seed, heavy = true)
    end
    throw(ArgumentError("no builder for synthetic case $id"))
end

# Normalise, add noise, then undersample: noise is drawn over the full grid so the realisation does
# not depend on the mask.
function _finish_cartesian(id, family, ref, maps, kfull, mask; seed, heavy = false, kw...)
    k = add_noise(norm_ksp(ComplexF32.(kfull)); snr_db = snr_db(), seed)
    k .*= _broadcast_mask(mask, family, size(k))
    image_size = family === :volume ? size(ref) : size(ref)[1:2]
    return BenchCase(;
        id, family, trajectory = :cartesian, reference = ComplexF32.(ref),
        smaps = maps === nothing ? nothing : ComplexF32.(maps), kspace = k, mask = BitArray(mask),
        image_size, heavy, seed, kw...,
    )
end

function _finish_radial(id, family, ref, maps, k, traj; seed, heavy = false, kw...)
    k = add_noise(norm_ksp(ComplexF32.(k)); snr_db = snr_db(), seed)
    return BenchCase(;
        id, family, trajectory = :noncartesian, reference = ComplexF32.(ref), smaps = ComplexF32.(maps),
        kspace = k, traj, dcf = ramp_dcf(traj), image_size = size(ref)[1:2], heavy, seed, kw...,
    )
end

function _broadcast_mask(mask, family, ksize)
    family === :single_slice && return reshape(mask, ksize[1], ksize[2], 1)
    family === :multislice && return reshape(mask, ksize[1], ksize[2], 1, 1)
    family === :volume && return reshape(mask, ksize[1], ksize[2], ksize[3], 1)
    family === :cine && return reshape(mask, ksize[1], ksize[2], 1, ksize[4])
    throw(ArgumentError("unknown family $family"))
end

"""
    _nfft_forward(images, traj) -> Array{ComplexF32, 3}

Non-Cartesian forward transform of each `(x, y)` image in `images` (`(x, y, k)`), returned as
`(sample, spoke, k)`. Uses MRT's NFFT operator at `m = 8, σ = 2.0` -- far more accurate than any
operating point a reconstruction uses -- single-threaded, so the simulated data are the same across
thread counts and checkouts.
"""
function _nfft_forward(images::AbstractArray{<:Complex, 3}, traj::AbstractArray{Float32, 3})
    ns, nsp = size(traj, 2), size(traj, 3)
    op = MriReconstructionToolbox.get_fourier_operator(
        zeros(ComplexF32, ns, nsp), size(images)[1:2], traj; threaded = false, m = 8, sigma = 2.0,
    )
    out = Array{ComplexF32}(undef, ns, nsp, size(images, 3))
    for k in axes(images, 3)
        out[:, :, k] = reshape(op * ComplexF32.(images[:, :, k]), ns, nsp)
    end
    return out
end

# ---------------------------------------------------------------- MRT adapter

"""
    mrt_acquisition(c::BenchCase; dcf = false) -> AcquisitionInfo

The case as MRT is meant to be given it: Cartesian k-space compacted to the sampled phase encodes
with `subsampling` recording which (a `Vector` of per-frame specs for cine), non-Cartesian k-space
with its trajectory, and `dcf = true` attaching the ramp DCF for a gridding reconstruction.

The volume case is built from plain arrays with integer `shifted_image_dims`: a `NamedDimsArray`
3D k-space subsampled on ky-kz (`(:kx, :kyz, :coil)`) is rejected when sensitivity maps are given,
because the map check demands a `:kz` axis (fixed on `fix/3d-subsampled-kspace-dimnames`; the
plain arrays stay until every measured ref carries the fix).
"""
function mrt_acquisition(c::BenchCase; dcf::Bool = false)
    c.trajectory === :noncartesian && return _mrt_noncartesian(c; dcf)
    nx, ny = size(c.kspace, 1), size(c.kspace, 2)
    # Phase encodes are given as a Bool vector, the form `create_sampling_pattern` returns: GRAPPA's
    # pattern check (`to_displayable_mask`) does not accept an index vector.
    if c.family === :single_slice
        lines = c.mask[1, :]
        k = c.kspace[:, lines, :]
        if c.smaps === nothing
            return CartesianAcquisitionInfo(
                NamedDimsArray{(:kx, :ky)}(k[:, :, 1]);
                is3D = false, image_size = (nx, ny), subsampling = (:, lines), shifted_image_dims = (:x, :y),
            )
        end
        return CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil)}(k);
            is3D = false, image_size = (nx, ny), subsampling = (:, lines), shifted_image_dims = (:x, :y),
            sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(c.smaps),
        )
    elseif c.family === :multislice
        lines = c.mask[1, :]
        return CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil, :z)}(c.kspace[:, lines, :, :]);
            is3D = false, image_size = (nx, ny), subsampling = (:, lines), shifted_image_dims = (:x, :y),
            sensitivity_maps = NamedDimsArray{(:x, :y, :coil, :z)}(c.smaps),
        )
    elseif c.family === :volume
        nz = size(c.kspace, 3)
        yz = c.mask[1, :, :]
        k = reshape(c.kspace, nx, ny * nz, :)[:, vec(yz), :]
        return CartesianAcquisitionInfo(
            k; is3D = true, image_size = (nx, ny, nz), subsampling = (:, yz), shifted_image_dims = (1, 2, 3),
            sensitivity_maps = c.smaps,
        )
    elseif c.family === :cine
        nt = size(c.kspace, 4)
        lines = [c.mask[1, :, t] for t in 1:nt]
        k = stack(c.kspace[:, lines[t], :, t] for t in 1:nt)                   # (kx, ky, coil, time)
        return CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil, :time)}(k);
            is3D = false, image_size = (nx, ny), subsampling = [(:, l) for l in lines],
            shifted_image_dims = (:x, :y), sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(c.smaps),
        )
    end
    throw(ArgumentError("unknown family $(c.family)"))
end

function _mrt_noncartesian(c::BenchCase; dcf::Bool)
    names = c.family === :cine ? (:sample, :spoke, :coil, :time) : (:sample, :spoke, :coil)
    return NonCartesianAcquisitionInfo(
        NamedDimsArray{names}(c.kspace);
        trajectory = NamedDimsArray{(:coord, :sample, :spoke)}(c.traj),
        dcf = dcf ? NamedDimsArray{(:sample, :spoke)}(c.dcf) : nothing,
        image_size = c.image_size, sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(c.smaps),
    )
end

# ---------------------------------------------------------------- methods

"""
    METHODS

Every method name the harness and the comparison suite know, in run order.
"""
const METHODS = (:adjoint, :gridding, :cgsense, :tv, :wavelet, :tgv, :lowrank, :llr, :ttv)

"""
    applicable_methods(c::BenchCase) -> Vector{Symbol}

The methods that make sense for case `c`: a direct reconstruction (adjoint for Cartesian, DCF
gridding for radial), CG-SENSE where there is more than one coil, spatial sparsity (TV, L1-wavelet,
TGV) for static images, and temporal priors (global and locally low rank, temporal TV) for cine.
TGV is 2D-only here (the 3D variant costs an order of magnitude more per iteration than anything
else in the catalog).
"""
function applicable_methods(c::BenchCase)
    ms = Symbol[c.trajectory === :noncartesian ? :gridding : :adjoint]
    ncoils(c) > 1 && push!(ms, :cgsense)
    if c.family === :cine
        append!(ms, (:lowrank, :llr, :ttv))
    elseif c.trajectory === :noncartesian
        push!(ms, :tv)
    elseif c.family === :volume
        append!(ms, (:tv, :wavelet))
    else
        append!(ms, (:tv, :wavelet, :tgv))
    end
    return ms
end
