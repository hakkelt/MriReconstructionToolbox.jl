# Real scanner data for the benchmark case catalog, via MRITestData.jl.
#
# Each real case mirrors one synthetic case (`REAL_CASES` in `cases.jl`) and comes out as the same
# `BenchCase` layout: fully sampled data undersampled retrospectively with the synthetic case's own
# pattern generator, so `reference` (the fully sampled reconstruction) is a genuine ground truth for
# the Cartesian cases. Real k-space is unit-RMS normalised but gets no added noise.
#
# ## Which dataset
#
# Every case lists candidates in order of preference. The open dataset is preferred when it is as
# good as the gated (registration-only) one for the purpose; the gated one comes first where the
# open one is less suitable:
#
# | case | first choice | fallback |
# |---|---|---|
# | 2D, 1 channel | gated fastMRI knee `singlecoil_val/file1000107` (native single channel) | open M4RAW, SENSE-combined to one channel |
# | 2D, multichannel | open M4RAW `multicoil_train/2022062402_T203`, middle slice | |
# | 2D, radial | gated fastMRI breast `fastMRI_breast_IDS_001_010/fastMRI_breast_006_2`, central partition of the golden-angle stack of stars | |
# | multislice | open M4RAW, the central 12 of its 18 slices | gated fastMRI brain `multicoil_val/file_brain_AXFLAIR_203_6000923` |
# | 3D | open MRIDATA knee `52c2fd53-d233-4444-8bfd-7c454240d314`, central 128³ of k-space | |
# | cine, Cartesian | open OCMR `fs_0001_1_5T` | |
# | cine, non-Cartesian | open USC speech spiral `sub001/2drt/09_northwind1_r1`, 3 of 13 arms per frame | |
#
# The breast DCE series would be the better radial cine (golden-angle radial, spokes binned into
# frames), but binned golden-angle spokes give every frame its own trajectory, which MRT cannot
# represent yet; the spiral series repeats its arms every frame, so it shares one trajectory.
#
# A gated source counts as available when its file is already cached or signed download URLs are
# registered and unexpired (`MRITestData.set_fastmri_urls!`). `MRT_BENCH_REAL_PREFER=open|gated`
# moves that kind to the front of every list. The source actually used is recorded in
# `BenchCase.source`.
#
# ## Caching
#
# Preparing a case is expensive (the breast file alone takes 80 s to load; ESPIRiT on a 3D volume
# takes minutes) and its maps come from MRT's own estimator, which could differ between two MRT
# checkouts under comparison. So every prepared case is serialised once, keyed by case id and
# dataset, under `MRT_BENCH_WORK_DIR` (default `<download path>/mrt_benchmark`), and every later run
# -- of any checkout -- loads those exact arrays.

"""
    RealSource(kind, source, id; combine = false)

One candidate dataset: `kind` is `:open` or `:gated`, `source` the MRITestData source name, `id` the
dataset id. `combine = true` reduces a multichannel dataset to one channel.
"""
struct RealSource
    kind::Symbol
    source::String
    id::String
    combine::Bool
end
RealSource(kind, source, id; combine::Bool = false) = RealSource(kind, source, id, combine)

const REAL_SOURCES = Dict(
    "real_2d_1ch_cartesian" => [
        RealSource(:gated, "FASTMRI", "singlecoil_val/file1000107"),
        RealSource(:open, "M4RAW", "multicoil_train/2022062402_T203"; combine = true),
    ],
    "real_2d_multichannel_cartesian" => [RealSource(:open, "M4RAW", "multicoil_train/2022062402_T203")],
    "real_2d_multichannel_radial" => [RealSource(:gated, "FASTMRI", "fastMRI_breast_IDS_001_010/fastMRI_breast_006_2")],
    "real_multislice_multichannel_cartesian" => [
        RealSource(:open, "M4RAW", "multicoil_train/2022062402_T203"),
        RealSource(:gated, "FASTMRI", "multicoil_val/file_brain_AXFLAIR_203_6000923"),
    ],
    "real_3d_multichannel_cartesian" => [RealSource(:open, "MRIDATA", "52c2fd53-d233-4444-8bfd-7c454240d314")],
    "real_cine_multichannel_cartesian" => [RealSource(:open, "OCMR", "fs_0001_1_5T")],
    "real_cine_multichannel_radial" => [RealSource(:open, "USC_SPEECH", "sub001/2drt/09_northwind1_r1")],
)

# Bumped whenever the preparation below changes, so stale cached cases are not reused.
const REAL_CACHE_VERSION = 1

function _mritestdata()
    return Base.require(Base.PkgId(Base.UUID("b3f1a2c4-5d6e-4a7b-9c8d-0e1f2a3b4c5d"), "MRITestData"))
end

function _source(name::AbstractString)
    M = _mritestdata()
    name == "M4RAW" && return M.M4RAW
    name == "MRIDATA" && return M.MRIDATA
    name == "OCMR" && return M.OCMR_SOURCE
    name == "FASTMRI" && return M.FASTMRI
    name == "USC_SPEECH" && return M.USC_SPEECH
    throw(ArgumentError("unknown MRITestData source $name"))
end

function _handle(s::RealSource)
    M = _mritestdata()
    return Base.invokelatest(M.dataset, _source(s.source), s.id; offline = true)
end

"""
    real_source_available(s::RealSource) -> Bool

Open sources are always available (MRITestData downloads them on demand). Gated ones only when the
file is cached or unexpired signed URLs are registered.
"""
function real_source_available(s::RealSource)
    s.kind === :open && return true
    M = _mritestdata()
    try
        Base.invokelatest(M.is_cached, _handle(s)) && return true
    catch
    end
    exp = Base.invokelatest(M.fastmri_url_expires)
    return exp !== nothing && exp > Dates.now(Dates.UTC)
end

"""
    real_candidates(real_id) -> Vector{RealSource}

The candidate datasets of `real_id`, in the order they are tried (see `MRT_BENCH_REAL_PREFER`).
"""
function real_candidates(real_id::AbstractString)
    cands = copy(REAL_SOURCES[real_id])
    pref = lowercase(get(ENV, "MRT_BENCH_REAL_PREFER", ""))
    if pref in ("open", "gated")
        sort!(cands; by = s -> string(s.kind) != pref, alg = Base.Sort.DEFAULT_STABLE)
    end
    return cands
end

work_dir() = get(ENV, "MRT_BENCH_WORK_DIR") do
    ensure_download_path!()
    joinpath(string(Base.invokelatest(_mritestdata().get_download_path)), "mrt_benchmark")
end

"""
    load_real_case(real_id, analogue) -> BenchCase

The first available candidate of `real_id`, prepared (or loaded from the case cache).
"""
function load_real_case(real_id::AbstractString, analogue::AbstractString)
    ensure_download_path!()
    errors = String[]
    for s in real_candidates(real_id)
        if !real_source_available(s)
            push!(errors, "$(s.source):$(s.id) is gated and neither cached nor registered")
            continue
        end
        path = joinpath(work_dir(), string(real_id, "__", replace(s.source * "_" * s.id, r"[^A-Za-z0-9_.-]" => "_"), "__v", REAL_CACHE_VERSION, ".jls"))
        if isfile(path)
            c = deserialize(path)
            c isa BenchCase && return c
        end
        c = try
            _prepare_real(real_id, analogue, s)
        catch err
            push!(errors, "$(s.source):$(s.id): " * first(sprint(showerror, err), 400))
            @warn "real case $real_id: $(s.source):$(s.id) failed, trying the next candidate" exception = (err, catch_backtrace())
            continue
        end
        mkpath(dirname(path))
        tmp = path * ".tmp$(getpid())"
        serialize(tmp, c)
        mv(tmp, path; force = true)
        return c
    end
    error("no dataset available for $real_id:\n  " * join(errors, "\n  "))
end

function _prepare_real(real_id, analogue, s::RealSource)
    label = string(s.source, ":", s.id)
    seed = _case_seed(real_id)
    rng = MersenneTwister(seed)
    raw = Base.invokelatest(_mritestdata().load_raw, _handle(s))
    if real_id == "real_2d_1ch_cartesian" || real_id == "real_2d_multichannel_cartesian"
        k = _remove_readout_oversampling(_assemble_cartesian_slices(raw; nslices = 1)[:, :, :, 1], raw)
        nx, ny, nc = size(k)
        single = real_id == "real_2d_1ch_cartesian"
        maps = nc == 1 ? nothing : _espirit_2d(k)
        if single && nc > 1
            k, maps = _combine_to_one_channel(k, maps), nothing
            label *= " (SENSE-combined to 1 channel)"
        end
        ref = maps === nothing ? centred_ifft(k[:, :, 1], (1, 2)) : _sense_combine(centred_ifft(k, (1, 2)), maps)
        sampled = vec(any(!iszero, k; dims = (1, 3)))
        R = single ? 2.5 : 4.0
        lines = vd_lines(rng, ny, round(Int, ny / R); acs = _scaled_acs(ny))
        mask = line_mask(nx, intersect(lines, findall(sampled)), ny)
        return _real_cartesian(real_id, analogue, :single_slice, ref, maps, k, mask, label, seed)
    elseif real_id == "real_multislice_multichannel_cartesian"
        k4 = _assemble_cartesian_slices(raw; nslices = 12)                       # (kx, ky, coil, slice)
        k4 = stack(_remove_readout_oversampling(k4[:, :, :, z], raw) for z in axes(k4, 4))
        nx, ny, nc, nz = size(k4)
        maps = stack(_espirit_2d(k4[:, :, :, z]) for z in 1:nz)                  # (x, y, coil, slice)
        ref = stack(_sense_combine(centred_ifft(k4[:, :, :, z], (1, 2)), maps[:, :, :, z]) for z in 1:nz)
        sampled = vec(any(!iszero, k4; dims = (1, 3, 4)))
        lines = vd_lines(rng, ny, ny ÷ 4; acs = _scaled_acs(ny))
        mask = line_mask(nx, intersect(lines, findall(sampled)), ny)
        return _real_cartesian(real_id, analogue, :multislice, ref, maps, k4, mask, label, seed)
    elseif real_id == "real_3d_multichannel_cartesian"
        k = _assemble_cartesian_3d_centre(raw, (128, 128, 128))                  # (kx, ky, kz, coil)
        nx, ny, nz, nc = size(k)
        maps = _espirit_3d(k)
        ref = _sense_combine(centred_ifft(k, (1, 2, 3)), maps)
        calib = 24
        yz = vd_mask_2d(rng, ny, nz; R = 6, calib)
        mask = BitArray(repeat(reshape(yz, 1, ny, nz), nx, 1, 1))
        return _real_cartesian(real_id, analogue, :volume, ref, maps, k, mask, label, seed; heavy = true)
    elseif real_id == "real_cine_multichannel_cartesian"
        k = _assemble_cartesian_dynamic(raw)                                      # (kx, ky, coil, time)
        size(k, 1) > 144 && (k = _center_readout(k, 144))
        nx, ny, nc, nt = size(k)
        # Calibrated from one frame: cardiac motion smears a time-averaged calibration region.
        maps = _espirit_2d(k[:, :, :, 1])
        ref = stack(_sense_combine(centred_ifft(k[:, :, :, t], (1, 2)), maps) for t in 1:nt)
        sampled = vec(any(!iszero, k; dims = (1, 3, 4)))
        acs = max(4, round(Int, 8 * ny / 128))
        lines = per_frame_lines(rng, ny, nt; centre = acs, random = max(1, ny ÷ 4 - acs))
        mask = falses(nx, ny, nt)
        for t in 1:nt
            mask[:, intersect(lines[t], findall(sampled)), t] .= true
        end
        return _real_cartesian(real_id, analogue, :cine, ref, maps, k, mask, label, seed; heavy = true)
    elseif real_id == "real_2d_multichannel_radial"
        return _prepare_breast_radial(real_id, analogue, raw, label, seed)
    elseif real_id == "real_cine_multichannel_radial"
        return _prepare_speech_spiral(real_id, analogue, raw, label, seed)
    end
    throw(ArgumentError("no preparation for real case $real_id"))
end

function _real_cartesian(real_id, analogue, family, ref, maps, kfull, mask, label, seed; heavy = false)
    k = norm_ksp(ComplexF32.(kfull))
    k .*= _broadcast_mask(mask, family, size(k))
    image_size = family === :volume ? size(ref) : size(ref)[1:2]
    return BenchCase(;
        id = real_id, family, trajectory = :cartesian, reference = ComplexF32.(ref),
        smaps = maps === nothing ? nothing : ComplexF32.(maps), kspace = k, mask = BitArray(mask),
        image_size, heavy, real = true, source = label, analogue, seed,
    )
end

# ACS width scaled with the phase-encode count, 16 lines at 128 like the synthetic cases.
_scaled_acs(ny) = max(8, round(Int, 16 * ny / 128))

# ---------------------------------------------------------------- Cartesian assembly

_idx(p) = p.head.idx
_readout_range(p) = (Int(p.head.discard_pre) + 1):(size(p.data, 1) - Int(p.head.discard_post))

"""
    _assemble_cartesian_slices(raw; nslices) -> Array{ComplexF64, 4}

`(kx, ky, coil, slice)` k-space of the `nslices` central slices (first contrast, repetition and
average), each profile placed by its `kspace_encode_step_1` counter.
"""
function _assemble_cartesian_slices(raw; nslices::Int)
    prof = [
        p for p in raw.profiles if
            Int(_idx(p).contrast) == 0 && Int(_idx(p).repetition) == 0 && Int(_idx(p).average) == 0
    ]
    slices = sort(unique(Int(_idx(p).slice) for p in prof))
    nslices <= length(slices) || error("the data has $(length(slices)) slices, $nslices requested")
    lo = (length(slices) - nslices) ÷ 2 + 1
    sel = slices[lo:(lo + nslices - 1)]
    rr = _readout_range(prof[1])
    ncoil = size(prof[1].data, 2)
    nky = maximum(Int(_idx(p).kspace_encode_step_1) for p in prof) + 1
    k = zeros(ComplexF64, length(rr), nky, ncoil, nslices)
    for p in prof
        z = findfirst(==(Int(_idx(p).slice)), sel)
        z === nothing && continue
        k[:, Int(_idx(p).kspace_encode_step_1) + 1, :, z] .= @view p.data[rr, :]
    end
    return k
end

# Keep the central half of the image along x when the readout is twice oversampled (encoded size
# twice the reconstruction size), as fastMRI's and most vendors' raw data are.
function _remove_readout_oversampling(k::AbstractArray{<:Complex, 3}, raw)
    enc = get(raw.params, "encodedSize", nothing)
    rec = get(raw.params, "reconSize", nothing)
    (enc === nothing || rec === nothing || enc[1] < 2 * rec[1] || size(k, 1) != enc[1]) && return k
    n = size(k, 1)
    img = centred_ifft(k, (1,))
    keep = (n ÷ 4 + 1):(n ÷ 4 + n ÷ 2)
    return centred_fft(img[keep, :, :], (1,))
end

# Crop dimension 1 to `target` samples centred on the actual DC (the readout energy peak), not the
# array midpoint: asymmetric-echo acquisitions put DC well off centre (OCMR: sample 147 of 404).
function _center_readout(ksp, target)
    n = size(ksp, 1)
    prof = vec(sum(abs2, reshape(ksp, n, :); dims = 2))
    dc = argmax(prof)
    lo = dc - target ÷ 2 + 1
    out = zeros(eltype(ksp), target, size(ksp)[2:end]...)
    src_lo, src_hi = max(1, lo), min(n, lo + target - 1)
    rest = ntuple(_ -> Colon(), ndims(ksp) - 1)
    out[(src_lo - lo + 1):(src_hi - lo + 1), rest...] .= ksp[src_lo:src_hi, rest...]
    return out
end

"""
    _assemble_cartesian_3d_centre(raw, target) -> Array{ComplexF64, 4}

`(kx, ky, kz, coil)` k-space of the central `target` window of a 3D Cartesian encode (partitions
in `idx.slice` or `kspace_encode_step_2`), so a large volume is never materialised in full: the
image keeps the full field of view at a lower resolution.
"""
function _assemble_cartesian_3d_centre(raw, target::NTuple{3, Int})
    prof = [
        p for p in raw.profiles if
            Int(_idx(p).contrast) == 0 && Int(_idx(p).repetition) == 0 && Int(_idx(p).average) == 0
    ]
    use_step2 = maximum(Int(_idx(p).kspace_encode_step_2) for p in prof) > 0
    kz_of(p) = use_step2 ? Int(_idx(p).kspace_encode_step_2) : Int(_idx(p).slice)
    nky = maximum(Int(_idx(p).kspace_encode_step_1) for p in prof) + 1
    nkz = maximum(kz_of(p) for p in prof) + 1
    rr = _readout_range(prof[1])
    ncoil = size(prof[1].data, 2)
    ylo, zlo = nky ÷ 2 + 1 - target[2] ÷ 2, nkz ÷ 2 + 1 - target[3] ÷ 2
    # DC along the readout from the centre profile's energy.
    centre = argmax(p -> (Int(_idx(p).kspace_encode_step_1) == nky ÷ 2) * (kz_of(p) == nkz ÷ 2) * 1.0, prof)
    dc = argmax(vec(sum(abs2, centre.data[rr, :]; dims = 2)))
    xlo = dc - target[1] ÷ 2
    k = zeros(ComplexF64, target..., ncoil)
    for p in prof
        y = Int(_idx(p).kspace_encode_step_1) + 1 - ylo + 1
        z = kz_of(p) + 1 - zlo + 1
        (1 <= y <= target[2] && 1 <= z <= target[3]) || continue
        for x in 1:target[1]
            s = xlo + x - 1
            1 <= s <= length(rr) && (k[x, y, z, :] .= @view p.data[rr[s], :])
        end
    end
    return k
end

# (kx, ky, coil, time) of the middle slice, frames in `idx.phase`.
function _assemble_cartesian_dynamic(raw)
    slices = sort(unique(Int(_idx(p).slice) for p in raw.profiles))
    sl = slices[cld(length(slices), 2)]
    prof = [
        p for p in raw.profiles if
            Int(_idx(p).slice) == sl && Int(_idx(p).contrast) == 0 &&
            Int(_idx(p).repetition) == 0 && Int(_idx(p).average) == 0
    ]
    rr = _readout_range(prof[1])
    ncoil = size(prof[1].data, 2)
    nky = maximum(Int(_idx(p).kspace_encode_step_1) for p in prof) + 1
    nfr = maximum(Int(_idx(p).phase) for p in prof) + 1
    k = zeros(ComplexF64, length(rr), nky, ncoil, nfr)
    for p in prof
        k[:, Int(_idx(p).kspace_encode_step_1) + 1, :, Int(_idx(p).phase) + 1] .= @view p.data[rr, :]
    end
    return k
end

# ---------------------------------------------------------------- maps and coil combination

# ESPIRiT through the acquisition method, so the maps come back in the centred image convention of
# an acquisition with `shifted_image_dims`. The raw-array method returns the plain-DFT convention,
# which multiplied into a centred image rolls the maps by half the field of view.
function _espirit_2d(k::AbstractArray{<:Complex, 3}; calib = 24, kernel = 6)
    nx, ny, _ = size(k)
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(ComplexF32.(k)); is3D = false, shifted_image_dims = (:x, :y),
    )
    method = MriReconstructionToolbox.ESPIRiT(; calib_size = min(calib, nx, ny), kernel_size = kernel)
    return Array(unname(MriReconstructionToolbox.estimate_sensitivities(acq; method).sensitivity_maps))
end

function _espirit_3d(k::AbstractArray{<:Complex, 4}; calib = 24, kernel = 6)
    acq = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :kz, :coil)}(ComplexF32.(k)); is3D = true, shifted_image_dims = (:x, :y, :z),
    )
    method = MriReconstructionToolbox.ESPIRiT(; calib_size = min(calib, size(k)[1:3]...), kernel_size = kernel)
    return Array(unname(MriReconstructionToolbox.estimate_sensitivities(acq; method).sensitivity_maps))
end

# SENSE-optimal coil combination Σ conj(S) x / Σ |S|², coil axis last of the maps' spatial axes.
function _sense_combine(coil_images, maps)
    cdim = ndims(maps)
    return dropdims(sum(conj.(maps) .* coil_images; dims = cdim) ./ (sum(abs2, maps; dims = cdim) .+ eps(Float32)); dims = cdim)
end

function _combine_to_one_channel(k, maps)
    img = _sense_combine(centred_ifft(k, (1, 2)), maps)
    return reshape(centred_fft(img, (1, 2)), size(img)..., 1)
end

# ---------------------------------------------------------------- non-Cartesian

# fastMRI breast: a golden-angle radial stack of stars (640 samples × 288 spokes × 83 partitions,
# 16 coils, readout twice oversampled). The central partition after the kz transform is the plain
# sum over partitions; the readout is cropped to its central 256 samples and rescaled onto a 256²
# grid (the full field of view at half the resolution). The first 128 spokes are the undersampled
# acquisition; maps and the reference come from all 288, the reference being a 30-iteration
# CG-SENSE. It is a DCE series, so the all-spoke reference averages the contrast change: read the
# NRMSE as agreement with that average, not as accuracy.
function _prepare_breast_radial(real_id, analogue, raw, label, seed; nkeep = 256, nspokes = 128)
    prof = raw.profiles
    nsamp, ncoil = size(prof[1].data)
    spokes = sort(unique(Int(_idx(p).kspace_encode_step_1) for p in prof))
    k = zeros(ComplexF32, nsamp, length(spokes), ncoil)
    traj = zeros(Float32, 2, nsamp, length(spokes))
    for p in prof
        j = Int(_idx(p).kspace_encode_step_1) + 1
        k[:, j, :] .+= p.data
        Int(_idx(p).slice) == 0 && (traj[:, :, j] .= p.traj[1:2, :])
    end
    c = Int(prof[1].head.center_sample) + 1
    keep = (c - nkeep ÷ 2):(c + nkeep ÷ 2 - 1)
    scale = Float32(nsamp / nkeep)
    k = k[keep, :, :]
    traj = clamp.(traj[:, keep, :] .* scale, -0.5f0, prevfloat(0.5f0))
    n = nkeep
    k = norm_ksp(k)
    full = _noncartesian_acq(k, traj, ramp_dcf(traj), (n, n))
    maps = Array(unname(MriReconstructionToolbox.estimate_sensitivities(full; method = MriReconstructionToolbox.ESPIRiT(; calib_size = 24, kernel_size = 6)).sensitivity_maps))
    ref = _cgsense_reference(k, traj, maps, (n, n))
    sel = 1:min(nspokes, size(k, 2))
    ksel, tsel = k[:, sel, :], traj[:, :, sel]
    return BenchCase(;
        id = real_id, family = :single_slice, trajectory = :noncartesian, reference = ComplexF32.(ref),
        smaps = ComplexF32.(maps), kspace = ksel, traj = tsel, dcf = ramp_dcf(tsel), image_size = (n, n),
        real = true, source = label * " (central partition, $(length(sel)) of $(size(k, 2)) spokes)", analogue, seed,
    )
end

# USC speech: a real-time spiral stream, 13 interleaves per fully sampled frame, the same 13 in
# every frame (sorted by interleaf counter), with vendor density compensation in `traj[3, :]`. The
# reference is the 13-arm gridding reconstruction per frame; the undersampled acquisition keeps 3 of
# the 13 arms (1, 6, 11) in every frame, which is what lets the series share one trajectory.
function _prepare_speech_spiral(real_id, analogue, raw, label, seed; narms = 13, arms = [1, 6, 11], frame0 = 104)
    nt = cine_frames()
    nsamp = Int(raw.profiles[1].head.number_of_samples)
    ncoil = size(raw.profiles[1].data, 2)
    nx, ny = Int.(raw.params["encodedSize"])[1:2]
    k = Array{ComplexF32}(undef, nsamp, narms, ncoil, nt)
    traj = Array{Float32}(undef, 2, nsamp, narms)
    dcf = Array{Float32}(undef, nsamp, narms)
    for f in 1:nt
        g = f + frame0
        ps = raw.profiles[((g - 1) * narms + 1):(g * narms)]
        ps = ps[sortperm([Int(_idx(p).kspace_encode_step_1) for p in ps])]
        for (a, p) in enumerate(ps)
            k[:, a, :, f] .= p.data
            if f == 1
                traj[:, :, a] .= p.traj[1:2, :]
                dcf[:, a] .= p.traj[3, :]
            end
        end
    end
    k = norm_ksp(k)
    full = _noncartesian_acq(k, traj, dcf, (nx, ny))
    est = MriReconstructionToolbox.estimate_sensitivities(full; method = MriReconstructionToolbox.ESPIRiT(; calib_size = 24, kernel_size = 6))
    maps = Array(unname(est.sensitivity_maps))
    ref = Array(unname(reconstruct(est, DirectReconstruction(); verbosity = Silent())))
    ksel, tsel = k[:, arms, :, :], traj[:, :, arms]
    return BenchCase(;
        id = real_id, family = :cine, trajectory = :noncartesian, reference = ComplexF32.(ref),
        smaps = ComplexF32.(maps), kspace = ksel, traj = tsel, dcf = dcf[:, arms], image_size = (nx, ny),
        heavy = true, real = true, source = label * " ($(length(arms)) of $narms arms, $nt frames)", analogue, seed,
    )
end

function _noncartesian_acq(k, traj, dcf, image_size)
    names = ndims(k) == 4 ? (:sample, :spoke, :coil, :time) : (:sample, :spoke, :coil)
    return NonCartesianAcquisitionInfo(
        NamedDimsArray{names}(k);
        trajectory = NamedDimsArray{(:coord, :sample, :spoke)}(traj),
        dcf = NamedDimsArray{(:sample, :spoke)}(dcf), image_size,
    )
end

function _cgsense_reference(k, traj, maps, image_size; maxit = 30)
    acq = NonCartesianAcquisitionInfo(
        NamedDimsArray{(:sample, :spoke, :coil)}(k);
        trajectory = NamedDimsArray{(:coord, :sample, :spoke)}(traj), image_size,
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(ComplexF32.(maps)),
    )
    m = IterativeReconstruction(; regularization = (), algorithm = MriReconstructionToolbox.CGNR(; maxit, tol = 0.0), maxit, reltol = 0.0)
    return Array(unname(reconstruct(acq, m; verbosity = Silent())))
end
