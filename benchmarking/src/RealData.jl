module RealData

# Real scanner k-space for the benchmark / comparison suites, via MRITestData.jl
# (https://github.com/hakkelt/MRITestData.jl — not yet registered, pulled in as a `[sources]`
# url by benchmark/, benchmarking/ and comparison/).
#
# `load_real_case` downloads (cached, on first use) one fully-sampled Cartesian dataset,
# assembles its middle slice into `(:kx, :ky, :coil)` k-space, estimates ESPIRiT sensitivity
# maps and a root-sum-of-squares reference image — the same shape the synthetic
# `generate_multicoil_brain` produces, so it drops straight into the existing cases.
#
# The default source is M4Raw (Zenodo, CC-BY): 0.3 T low-field brain, 4-channel, ~12 MB per
# member, fully sampled — the cheapest real Cartesian data to pull on a fresh machine. Larger
# multi-vendor options: `MRIDATA` (mridata.org knee/brain ISMRMRD, ~1 GB) and, once the
# access forms are done, `FASTMRI`. Each provider's own licence and citation terms apply —
# see https://hakkelt.github.io/MRITestData.jl/stable/legal/.

using NamedDims: NamedDimsArray
using FFTW: ifft, fftshift, ifftshift
using MRITestData: MRITestData, list_datasets, load_raw
using MriReconstructionToolbox: estimate_sensitivities, ESPIRiT

export load_real_case, real_data_available, real_data_source

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

"""
    real_data_source()

The `MRITestData` source used by [`load_real_case`], from `ENV["MRT_BENCH_REAL_SOURCE"]`
(`"M4RAW"` / `"MRIDATA"` / `"FASTMRI"`), defaulting to `M4RAW`.
"""
function real_data_source()
    s = uppercase(get(ENV, "MRT_BENCH_REAL_SOURCE", "M4RAW"))
    s == "M4RAW" && return MRITestData.M4RAW
    s == "MRIDATA" && return MRITestData.MRIDATA
    s == "FASTMRI" && return MRITestData.FASTMRI
    error("Unknown MRT_BENCH_REAL_SOURCE=$s (expected M4RAW, MRIDATA or FASTMRI)")
end

"""
    real_data_available(; source = real_data_source()) -> Bool

Whether the offline catalog lists at least one fully-sampled Cartesian entry for `source`
(the FASTMRI / MRIDATA maps can be empty until their access step is done).
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
    load_real_case(; source = real_data_source(), filter = nothing,
                     calib_size = 24, kernel_size = 6)

Download (cached) the smallest matching fully-sampled Cartesian dataset, assemble its middle
slice, and return

    (; kspace, smaps, reference, image_size, label)

with `kspace :: NamedDimsArray{(:kx, :ky, :coil)}` (ComplexF64),
`smaps :: NamedDimsArray{(:x, :y, :coil)}` from ESPIRiT, `reference` the RSS magnitude image,
`image_size == (nkx, nky)` and `label` a `"SOURCE:id"` string.

`filter`, when given, keeps only entries whose id contains it (separators / case ignored),
e.g. `filter = "T2"` or `filter = "knee"`.
"""
function load_real_case(;
        source = real_data_source(),
        filter = get(ENV, "MRT_BENCH_REAL_FILTER", nothing),
        calib_size = 24,
        kernel_size = 6,
    )
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
    entry = first(entries)

    raw = load_raw(entry)
    ksp = _assemble_cartesian_slice(raw)                  # (nkx, nky, ncoil) ComplexF64
    nkx, nky, _ = size(ksp)

    kspace = NamedDimsArray{(:kx, :ky, :coil)}(ksp)
    smaps = estimate_sensitivities(
        kspace;
        method = ESPIRiT(; calib_size = min(calib_size, nkx, nky), kernel_size),
    )
    coil_imgs = fftshift(ifft(ifftshift(ksp, (1, 2)), (1, 2)), (1, 2))
    reference = sqrt.(sum(abs2, coil_imgs; dims = 3))[:, :, 1]

    label = string(MRITestData.source_name(entry.source), ":", entry.id)
    return (; kspace, smaps, reference, image_size = (nkx, nky), label)
end

end
