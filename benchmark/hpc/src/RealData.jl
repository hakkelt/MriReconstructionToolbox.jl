module RealData

# Real scanner k-space for the benchmark / comparison suites, via MRITestData.jl
# (https://github.com/hakkelt/MRITestData.jl — not yet registered, pulled in as a `[sources]`
# url by benchmark/ci/, benchmark/hpc/ and benchmark/comparison/).
#
# `load_real_case` downloads (cached, on first use) one fully-sampled Cartesian dataset,
# assembles its middle slice into `(:kx, :ky, :coil)` k-space, estimates ESPIRiT sensitivity
# maps and a root-sum-of-squares reference image — the same shape the synthetic
# `generate_multicoil_brain` produces, so it drops straight into the existing cases.
# `combine_coils = true` folds that multi-coil k-space down to one virtual channel (see below).
#
# ── Catalog survey (offline `list_datasets` over every source, 2026-09-01) ────────────────────
# `MRITestData.list_sources()` == M4RAW, MRIDATA, OCMR, CMRXRECON2024, CMRXRECON300, USC_SPEECH.
# There is **no fastMRI source** in the package (an earlier draft here referenced a non-existent
# `MRITestData.FASTMRI`). fastMRI's own `singlecoil_*` knee files are real single-channel data
# but the dataset is behind a name/email registration wall (emailed download links, not a
# click-through), so it is not an *open* alternative and is not wired in.
#
# By data type, with the open pick first:
#
#   multi-channel Cartesian, fully sampled
#     M4RAW  multicoil_train/2022062402_T203    4ch  0.3 T brain, 256², 11.7 MB   — PINNED, open (Zenodo CC-BY)
#     MRIDATA <uuid>                         8–15ch  1.5/3 T knee & brain, ISMRMRD, ~1–1.5 GB   — open (mridata.org per-dataset terms)
#     OCMR   fs_0001_1_5T                     multi  1.5 T cardiac cine, ~200 MB                — open (OCMR data-use terms + citation)
#
#   multi-channel non-Cartesian
#     USC_SPEECH sub029/2drt/04_bvt_r2          8ch  1.5 T spiral vocal tract, 60 MB            — open (figshare CC-BY)
#
#   GRAPPA-style undersampled (regular / pseudo-random) with an autocalibration region
#     OCMR   us_0001_3T                        multi  3 T cardiac sax, pseudo-random R≈4 with a
#                                                     dense k-space centre (VISTA-like ACS), ~90 MB   — open, replaces the gated pick
#     CMRXRECON300 DemoData/P001/cine_sax      multi  R≈3 + explicit fully-sampled `calib` file   — needs a free Synapse token
#
#   single-channel (Cartesian or non-Cartesian)
#     No source in the catalog ships true single-channel k-space — the smallest real array is
#     MRIDATA's two 3-channel spin-echo sets, then M4RAW at 4. `load_real_case(combine_coils=true)`
#     synthesizes one: SENSE-optimal combine of a real multi-coil member (∑ conj(sᵢ)·xᵢ / ∑|sᵢ|²)
#     to a single complex image, FFT back to a 1-channel k-space, sensitivity ≡ 1. Real anatomy
#     and noise, one channel — a compressed-sensing (no parallel imaging) benchmark point.

using NamedDims: NamedDimsArray, unname
using FFTW: fft, ifft, fftshift, ifftshift
using MRITestData: MRITestData, list_datasets, dataset, load_raw
using MriReconstructionToolbox: estimate_sensitivities, ESPIRiT

# The dataset `load_real_case` uses by default: `(source_name, id)`. Real scanner k-space,
# hard-wired for a reproducible benchmark rather than "whatever is smallest". Override at
# runtime with `MRT_BENCH_REAL_SOURCE` + `MRT_BENCH_REAL_ID`, or fall back to the
# smallest-matching search by setting `MRT_BENCH_REAL_ID=auto`.
const PINNED_DATASET = ("M4RAW", "multicoil_train/2022062402_T203")

# Larger cases (downloaded once, cached; ~1.5 GB / ~0.2 GB). See `load_real_case_3d` /
# `load_real_dynamic`.
#   3D knee   — Stanford fully-sampled 3D FSE knee, subject 1: 320×320 kx/ky, 256 kz partitions,
#               8-channel, 3 T. Reconstructed as a stack of 2D slices (IFFT along kz) so the
#               the task is split over the slice batch dim — the case where 8 threads should
#               beat 1.
#   dynamic   — OCMR fully-sampled cine fs_0001_1_5T: 15-channel, 19 cardiac phases, 208 PE,
#               1.5 T. Retrospectively 2×-undersampled for the low-rank / temporal-TV rows.
const PINNED_3D = ("MRIDATA", "52c2fd53-d233-4444-8bfd-7c454240d314")
const PINNED_DYNAMIC = ("OCMR", "fs_0001_1_5T")

export load_real_case, load_real_case_3d, load_real_dynamic
export real_data_available, real_data_source

# Match a filter needle against an entry id, ignoring separators and case.
_norm(s) = lowercase(replace(String(s), r"[\s\-_/]" => ""))

# Assemble one fully-sampled 2D Cartesian slice from a RawAcquisitionData into a dense
# (nkx, nky, ncoil) k-space array. Picks the middle slice of the first contrast / repetition /
# average and places each profile by its `kspace_encode_step_1` (phase-encode) counter.
function _assemble_cartesian_slice(raw)
    idx(p) = p.head.idx
    slices = sort(unique(Int(idx(p).slice) for p in raw.profiles))
    sl = slices[cld(length(slices), 2)]
    prof = [
        p for p in raw.profiles if
            Int(idx(p).slice) == sl && Int(idx(p).contrast) == 0 &&
            Int(idx(p).repetition) == 0 && Int(idx(p).average) == 0
    ]
    isempty(prof) && error("no imaging profiles for the selected slice")
    nsamp, ncoil = size(prof[1].data)
    pre = Int(prof[1].head.discard_pre)
    post = Int(prof[1].head.discard_post)
    nkx = nsamp - pre - post
    nky = maximum(Int(idx(p).kspace_encode_step_1) for p in prof) + 1
    ksp = zeros(ComplexF64, nkx, nky, ncoil)
    for p in prof
        row = Int(idx(p).kspace_encode_step_1) + 1
        ksp[:, row, :] .= ComplexF64.(@view p.data[(pre + 1):(pre + nkx), :])
    end
    return ksp
end

_img_from_ksp(ksp) = fftshift(ifft(ifftshift(ksp, (1, 2)), (1, 2)), (1, 2))
_ksp_from_img(img) = fftshift(fft(ifftshift(img, (1, 2)), (1, 2)), (1, 2))

_readout_range(p) = begin
    pre = Int(p.head.discard_pre)
    post = Int(p.head.discard_post)
    n = size(p.data, 1) - pre - post
    (pre + 1):(pre + n)
end

# Crop dimension 1 of a k-space array to `target` samples centred on the **actual** DC (the
# readout energy peak), not the array midpoint — asymmetric-echo / partial-Fourier acquisitions
# put DC well off centre (e.g. OCMR: sample 147 of 404). Zero-pads if the window runs past an
# edge. Returns the array unchanged when `target ≥ size(ksp, 1)` and DC is already centred.
function _center_readout(ksp, target)
    n = size(ksp, 1)
    prof = sum(abs2, reshape(ksp, n, :); dims = 2)[:, 1]
    dc = argmax(prof)
    half = target ÷ 2
    lo, hi = dc - half + 1, dc + half
    out = zeros(eltype(ksp), target, size(ksp)[2:end]...)
    src_lo, src_hi = max(1, lo), min(n, hi)
    dst_lo = src_lo - lo + 1
    out[dst_lo:(dst_lo + src_hi - src_lo), (Colon() for _ in 2:ndims(ksp))...] .=
        ksp[src_lo:src_hi, (Colon() for _ in 2:ndims(ksp))...]
    return out
end

# Assemble a full 3D Cartesian k-space (nkx, nky, nkz, ncoil) from a RawAcquisitionData, keying
# each profile by `idx.slice` (partition / kz) and `idx.kspace_encode_step_1` (ky), first
# contrast / repetition / average only.
function _assemble_cartesian_3d(raw)
    idx(p) = p.head.idx
    prof = [
        p for p in raw.profiles if
            Int(idx(p).contrast) == 0 && Int(idx(p).repetition) == 0 && Int(idx(p).average) == 0
    ]
    isempty(prof) && error("no imaging profiles")
    rr = _readout_range(prof[1])
    nkx = length(rr)
    _, ncoil = size(prof[1].data)
    nky = maximum(Int(idx(p).kspace_encode_step_1) for p in prof) + 1
    nkz = maximum(Int(idx(p).slice) for p in prof) + 1
    ksp = zeros(ComplexF64, nkx, nky, nkz, ncoil)
    for p in prof
        ky = Int(idx(p).kspace_encode_step_1) + 1
        kz = Int(idx(p).slice) + 1
        ksp[:, ky, kz, :] .= ComplexF64.(@view p.data[rr, :])
    end
    return ksp
end

# Assemble a dynamic 2D Cartesian k-space (nkx, nky, ncoil, nframes), keying each profile by
# `idx.kspace_encode_step_1` (ky) and `idx.phase` (cardiac frame), middle slice only.
function _assemble_cartesian_dynamic(raw)
    idx(p) = p.head.idx
    slices = sort(unique(Int(idx(p).slice) for p in raw.profiles))
    sl = slices[cld(length(slices), 2)]
    prof = [
        p for p in raw.profiles if
            Int(idx(p).slice) == sl && Int(idx(p).contrast) == 0 &&
            Int(idx(p).repetition) == 0 && Int(idx(p).average) == 0
    ]
    isempty(prof) && error("no imaging profiles for the selected slice")
    rr = _readout_range(prof[1])
    nkx = length(rr)
    _, ncoil = size(prof[1].data)
    nky = maximum(Int(idx(p).kspace_encode_step_1) for p in prof) + 1
    nfr = maximum(Int(idx(p).phase) for p in prof) + 1
    ksp = zeros(ComplexF64, nkx, nky, ncoil, nfr)
    for p in prof
        ky = Int(idx(p).kspace_encode_step_1) + 1
        fr = Int(idx(p).phase) + 1
        ksp[:, ky, :, fr] .= ComplexF64.(@view p.data[rr, :])
    end
    return ksp
end

_espirit(kslice; calib_size, kernel_size) = estimate_sensitivities(
    NamedDimsArray{(:kx, :ky, :coil)}(kslice);
    method = ESPIRiT(; calib_size = min(calib_size, size(kslice, 1), size(kslice, 2)), kernel_size),
)

"""
    load_real_case_3d(; nslices = 24, source = PINNED_3D, id = <pinned>,
                        calib_size = 24, kernel_size = 6)

Download (cached) the pinned Stanford 3D FSE knee, IFFT along the kz partition axis, take
`nslices` central slices, and return

    (; kspace, smaps, reference, image_size, label)

with `kspace :: NamedDimsArray{(:kx, :ky, :coil, :z)}`,
`smaps :: NamedDimsArray{(:x, :y, :coil, :z)}` (per-slice ESPIRiT),
`reference :: Array{Float64,3}` (per-slice RSS), `image_size == (nkx, nky, nslices)`.

Each slice is an independent 2D problem; `reconstruct` splits the task over `:z`.
"""
function load_real_case_3d(;
        nslices = 24,
        source = MRITestData.MRIDATA,
        id = get(ENV, "MRT_BENCH_REAL3D_ID", PINNED_3D[2]),
        calib_size = 24,
        kernel_size = 6,
    )
    raw = load_raw(dataset(source, id; offline = true))
    ksp = _assemble_cartesian_3d(raw)                     # (nkx, nky, nkz, ncoil)
    nkx, nky, nkz, ncoil = size(ksp)
    # kz -> image slice
    img_z = fftshift(ifft(ifftshift(ksp, 3), 3), 3)       # still k-space in kx, ky
    lo = max(1, (nkz - nslices) ÷ 2 + 1)
    sel = lo:(lo + min(nslices, nkz) - 1)
    slab = permutedims(img_z[:, :, sel, :], (1, 2, 4, 3)) # (kx, ky, coil, z)
    nz = length(sel)

    smaps = Array{ComplexF64}(undef, nkx, nky, ncoil, nz)
    reference = Array{Float64}(undef, nkx, nky, nz)
    for k in 1:nz
        ks = slab[:, :, :, k]
        smaps[:, :, :, k] = unname(_espirit(ks; calib_size, kernel_size))
        reference[:, :, k] = sqrt.(sum(abs2, _img_from_ksp(ks); dims = 3))[:, :, 1]
    end
    label = string(MRITestData.source_name(source), ":", id, " (3D, $nz slices)")
    return (;
        kspace = NamedDimsArray{(:kx, :ky, :coil, :z)}(ComplexF64.(slab)),
        smaps = NamedDimsArray{(:x, :y, :coil, :z)}(smaps),
        reference,
        image_size = (nkx, nky, nz),
        label,
    )
end

"""
    load_real_dynamic(; source = PINNED_DYNAMIC, id = <pinned>, R = 2, acs = 12,
                        calib_size = 24, kernel_size = 6)

Download (cached) the pinned OCMR fully-sampled cine, assemble `(kx, ky, coil, time)`,
retrospectively undersample phase-encode by `R` (keeping `2·acs+1` central lines), and return

    (; kspace, smaps, reference, subsampling, image_size, label)

`kspace :: NamedDimsArray{(:kx, :ky, :coil, :time)}` (only the sampled lines are non-zero),
`smaps` from the time-averaged k-space, `reference :: Array{Float64,3}` the per-frame RSS of the
*fully-sampled* data, `subsampling == (:, mask)`, `image_size == (nkx, nky, nframes)`.
"""
function load_real_dynamic(;
        source = MRITestData.OCMR_SOURCE,
        id = get(ENV, "MRT_BENCH_REALDYN_ID", PINNED_DYNAMIC[2]),
        R = 2, acs = 12, readout = 144, calib_size = 24, kernel_size = 6,
    )
    raw = load_raw(dataset(source, id; offline = true))
    ksp = _assemble_cartesian_dynamic(raw)               # (nkx, nky, ncoil, nframes)
    # Crop the (often 2× oversampled / asymmetric-echo) readout to `readout` samples about the
    # true DC — a lower-resolution but much cheaper *and* correctly-centred problem.
    readout !== nothing && size(ksp, 1) > readout && (ksp = _center_readout(ksp, readout))
    nkx, nky, ncoil, nfr = size(ksp)

    reference = Array{Float64}(undef, nkx, nky, nfr)
    for f in 1:nfr
        reference[:, :, f] = sqrt.(sum(abs2, _img_from_ksp(ksp[:, :, :, f]); dims = 3))[:, :, 1]
    end
    # Calibrate from a single frame, not the time average: cardiac motion across a full cine
    # smears a temporally-averaged calibration region and corrupts ESPIRiT (measured: a fully
    # sampled, unregularized CG-SENSE recon against averaged-calibration maps was NRMSE 0.63
    # from the coil-independent RSS reference; frame-1 calibration is the fix).
    smaps = _espirit(ksp[:, :, :, 1]; calib_size, kernel_size)

    mask = falses(nky)
    mask[1:R:nky] .= true
    mask[max(1, nky ÷ 2 - acs):min(nky, nky ÷ 2 + acs)] .= true
    # MRT wants the k-space compacted to the sampled phase-encode lines, with `subsampling`
    # recording which lines those were (matches the synthetic `Dynamic` group).
    ksp_us = ksp[:, mask, :, :]

    label = string(MRITestData.source_name(source), ":", id, " (cine, $nfr frames, R=$R)")
    return (;
        kspace = NamedDimsArray{(:kx, :ky, :coil, :time)}(ComplexF64.(ksp_us)),
        smaps,
        reference,
        subsampling = (:, mask),
        image_size = (nkx, nky, nfr),
        label,
    )
end

"""
    real_data_source()

The `MRITestData` source used by [`load_real_case`], from `ENV["MRT_BENCH_REAL_SOURCE"]`
(`"M4RAW"` / `"MRIDATA"` / `"OCMR"`), defaulting to [`PINNED_DATASET`](@ref)'s source.
"""
function real_data_source()
    s = uppercase(get(ENV, "MRT_BENCH_REAL_SOURCE", PINNED_DATASET[1]))
    s == "M4RAW" && return MRITestData.M4RAW
    s == "MRIDATA" && return MRITestData.MRIDATA
    s == "OCMR" && return MRITestData.OCMR_SOURCE
    error("Unknown MRT_BENCH_REAL_SOURCE=$s (expected M4RAW, MRIDATA or OCMR)")
end

"""
    real_data_available(; source = real_data_source()) -> Bool

Whether the offline catalog lists at least one fully-sampled Cartesian entry for `source`.
"""
function real_data_available(; source = real_data_source())
    try
        return !isempty(_candidates(source))
    catch
        return false
    end
end

function _candidates(source)
    entries = list_datasets(source; offline = true, fully_sampled = true)
    return [e for e in entries if e.trajectory === :cartesian || e.trajectory === nothing]
end

"""
    load_real_case(; source = real_data_source(), id = <pinned>, filter = nothing,
                     combine_coils = false, calib_size = 24, kernel_size = 6)

Download (cached) one fully-sampled Cartesian dataset, assemble its middle slice, and return

    (; kspace, smaps, reference, image_size, label)

with `kspace :: NamedDimsArray{(:kx, :ky, :coil)}` (ComplexF64),
`smaps :: NamedDimsArray{(:x, :y, :coil)}` from ESPIRiT, `reference` the RSS magnitude image,
`image_size == (nkx, nky)` and `label` a `"SOURCE:id"` string.

By default `id` is [`PINNED_DATASET`](@ref)'s id (`ENV["MRT_BENCH_REAL_ID"]` overrides it).
Set `id = "auto"` (or `ENV["MRT_BENCH_REAL_ID"] = "auto"`) to instead take the smallest
matching entry, optionally narrowed by `filter` — an id substring, separators / case ignored,
e.g. `filter = "T2"` or `filter = "knee"`.

`combine_coils = true` collapses the multi-coil member to one virtual channel: a SENSE-optimal
combine (`∑ conj(sᵢ)·xᵢ / ∑|sᵢ|²`) to a single complex image, FFT back to a 1-channel k-space,
`smaps ≡ 1`, `reference` the combined magnitude. The catalog has no native single-channel data
(see the survey above), so this is the way to a real-anatomy single-coil benchmark point.
"""
function load_real_case(;
        source = real_data_source(),
        id = get(ENV, "MRT_BENCH_REAL_ID", PINNED_DATASET[2]),
        filter = get(ENV, "MRT_BENCH_REAL_FILTER", nothing),
        combine_coils = false,
        calib_size = 24,
        kernel_size = 6,
    )
    entry = if id == "auto"
        entries = _candidates(source)
        isempty(entries) && error(
            "No offline fully-sampled Cartesian entries for $source. Populate its map or set ",
            "MRT_BENCH_REAL_SOURCE — see https://hakkelt.github.io/MRITestData.jl/dev/.",
        )
        if filter !== nothing
            needle = _norm(filter)
            entries = [e for e in entries if occursin(needle, _norm(string(e.id)))]
            isempty(entries) && error("filter=$(repr(filter)) matched no $source entry")
        end
        sort!(entries; by = e -> something(e.approx_size_bytes, typemax(Int)))
        first(entries)
    else
        dataset(source, id; offline = true)
    end
    # `dataset` hands back a `DatasetHandle` (wrapping `.entry`); `list_datasets` a bare
    # `DatasetEntry`. Normalise for the label; `load_raw` takes either.
    meta = entry isa MRITestData.DatasetHandle ? entry.entry : entry

    raw = load_raw(entry)
    ksp = _assemble_cartesian_slice(raw)                  # (nkx, nky, ncoil) ComplexF64
    nkx, nky, _ = size(ksp)

    smaps_full = estimate_sensitivities(
        NamedDimsArray{(:kx, :ky, :coil)}(ksp);
        method = ESPIRiT(; calib_size = min(calib_size, nkx, nky), kernel_size),
    )
    label = string(MRITestData.source_name(meta.source), ":", meta.id)

    if combine_coils
        s = unname(smaps_full)
        coil_imgs = _img_from_ksp(ksp)
        comb = sum(conj(s) .* coil_imgs; dims = 3) ./ (sum(abs2, s; dims = 3) .+ eps())
        ksp1 = _ksp_from_img(comb)                        # (nkx, nky, 1)
        kspace = NamedDimsArray{(:kx, :ky, :coil)}(ComplexF64.(ksp1))
        smaps = NamedDimsArray{(:x, :y, :coil)}(ones(ComplexF64, nkx, nky, 1))
        reference = abs.(comb)[:, :, 1]
        return (; kspace, smaps, reference, image_size = (nkx, nky), label = label * " (1ch)")
    end

    kspace = NamedDimsArray{(:kx, :ky, :coil)}(ksp)
    coil_imgs = _img_from_ksp(ksp)
    reference = sqrt.(sum(abs2, coil_imgs; dims = 3))[:, :, 1]
    return (; kspace, smaps = smaps_full, reference, image_size = (nkx, nky), label)
end

end
