module MriReconstructionToolboxMRIBaseExt

using MriReconstructionToolbox
using MriReconstructionToolbox: CartesianAcquisitionInfo, NonCartesianAcquisitionInfo
using ArgCheck: @argcheck
using NamedDims: NamedDimsArray
using MRIBase: MRIBase, RawAcquisitionData, Limit, kspaceNodes

# ISMRMRD flag bit for `ACQ_IS_NOISE_MEASUREMENT` (1-based bit index 19, per the ISMRMRD spec /
# MRIBase.jl's `Flags.jl`). Noise-calibration profiles carry no image k-space and must be
# excluded before the profile list is turned into an array.
const _NOISE_MEASUREMENT_BIT = UInt64(1) << (19 - 1)
_is_noise_profile(p) = (p.head.flags & _NOISE_MEASUREMENT_BIT) != 0

function _image_profiles(raw::RawAcquisitionData)
    profiles = [p for p in raw.profiles if !_is_noise_profile(p)]
    @argcheck !isempty(profiles) "RawAcquisitionData contains no image (non-noise) profiles"
    return profiles
end

"""
    AcquisitionInfo(raw::MRIBase.RawAcquisitionData; sensitivity_maps=nothing)

Build an `AcquisitionInfo` directly from an ISMRMRD `RawAcquisitionData` (as returned by
`MRIFiles.RawAcquisitionData`/`MRITestData.load_raw`), without hand-assembling k-space arrays
from `raw.profiles`.

Derived automatically:
- **Encoding matrix / image size** from `raw.params["encodedSize"]`.
- **Cartesian vs. non-Cartesian dispatch**, from `raw.params["trajectory"]`: `"cartesian"``
  (case-insensitive) builds a [`CartesianAcquisitionInfo`](@ref); anything else builds a
  [`NonCartesianAcquisitionInfo`](@ref) from `MRIBase.trajectory`/`MRIBase.rawdata`.
- **`is3D`**, from whether `kspace_encode_step_2` actually varies (`enc_lim_kspace_encoding_step_2`)
  — a multi-slice 2D acquisition has `encodedSize[3] > 1` but a *singleton* `kspace_encode_step_2`,
  so it stays 2D with slices as a `:z` batch dimension (see below), not a 3D encoding.
- **Coil dimension** (`:coil`), from `head.active_channels`.
- **The subsampling pattern actually present in the profiles** (not merely "fully sampled"): the
  readout, phase-encode and (if 3D) partition-encode axes are each reduced to the tightest
  `AcquisitionInfo` subsampling spec that reproduces them — `Colon()` when everything in
  `encodedSize` is present, an `OrdinalRange` when the present samples form one contiguous block
  (e.g. an asymmetric-echo/partial-Fourier readout), or a `BitArray` mask otherwise (e.g. an
  accelerated ky pattern). `kspace_data` itself stores only the samples that were actually
  acquired — it is not zero-padded to `encodedSize`.
- **Slices/contrasts/repetitions/etc. as batch dimensions**: any of `slice`, `contrast`,
  `phase`, `repetition`, `set`, `average` that takes more than one distinct value across the
  profiles becomes a batch dimension, named respectively `:z`, `:contrast`, `:time`,
  `:repetition`, `:set`, `:average` (`:z` first, matching the convention in
  `docs/src/high-level/nameddims.md`: "Multi-slice data should include `:z` as the first batch
  dimension"). A counter that never varies contributes no dimension. `is3D` acquisitions with
  more than one slice (multi-slab 3D) are not supported (`:z` cannot coexist with 3D k-space) —
  filter `raw.profiles` yourself and construct a `CartesianAcquisitionInfo` slab-by-slab instead.

Not derivable, so it must be passed explicitly if needed:
- `sensitivity_maps`: `RawAcquisitionData` carries no coil sensitivity information.

## FFT-shift convention

`AcquisitionInfo` expects k-space with the DC component at the geometric centre of each Fourier
axis — index `N ÷ 2 + 1` for an axis of length `N` — which is what `fftshift` produces and what
[`get_fourier_operator`](@ref)'s internal `ifftshift`/`fftshift` pair assumes (see "FFT Shift
Conventions" in `docs/src/high-level/acquisition_info.md`). A profile's `kspace_encode_step_1`
(and, if 3D, `kspace_encode_step_2`) is an index into the *encoded* matrix starting at 0, with the
true k=0 line at `encoding_limits.center` — not necessarily `encodedSize ÷ 2` (partial-Fourier /
asymmetric acquisitions in particular need not centre their limits). Likewise a profile's readout
samples are indexed from 0 with the true k=0 sample at `head.center_sample`, again not necessarily
the middle of the acquired samples (asymmetric echo). This constructor places every sample at
`raw_index - center + N ÷ 2` (1-based: `+1`) along its axis before handing anything to
`CartesianAcquisitionInfo`, so the result is already centred: no `fftshift` call is needed in user
code afterwards. Ignoring these offsets — placing sample/line `i` at raw position `i + 1` — is
exactly the bug this constructor exists to avoid: it silently shifts the reconstructed image by
`center - N ÷ 2` samples along the affected axis.
"""
function MriReconstructionToolbox.AcquisitionInfo(raw::RawAcquisitionData; sensitivity_maps = nothing)
    trajectory_name = lowercase(String(get(raw.params, "trajectory", "cartesian")))
    if trajectory_name == "cartesian"
        return _cartesian_acquisition_info(raw; sensitivity_maps)
    end
    return _noncartesian_acquisition_info(raw; sensitivity_maps)
end

# Sorted-unique raw ids -> 1-based compact index (preserves ascending order, so a monotonic
# offset such as the FFT-centring shift below cannot reorder them).
function _compact_index_map(ids::AbstractVector{<:Integer})
    uids = sort(unique(ids))
    return Dict(id => i for (i, id) in enumerate(uids)), uids
end

# The most specific `AcquisitionInfo` subsampling spec that reproduces a sorted set of 1-based
# positions along a length-`n` axis: `Colon()` when everything is present, a range when the
# present positions form one contiguous block, else a boolean mask.
function _positions_to_subsampling(n::Integer, positions::AbstractVector{<:Integer})
    if length(positions) == n
        return Colon()
    elseif length(positions) == last(positions) - first(positions) + 1
        return first(positions):last(positions)
    end
    mask = falses(n)
    mask[positions] .= true
    return mask
end

function _cartesian_acquisition_info(raw::RawAcquisitionData; sensitivity_maps = nothing)
    profiles = _image_profiles(raw)

    enc = Int.(raw.params["encodedSize"])
    nkx, nky, nkz_full = enc[1], enc[2], enc[3]
    lim1 = raw.params["enc_lim_kspace_encoding_step_1"]::Limit
    lim2 = get(raw.params, "enc_lim_kspace_encoding_step_2", Limit(0, 0, 0))::Limit
    is3D = lim2.maximum > lim2.minimum
    nkz = is3D ? nkz_full : 1

    p1 = first(profiles)
    nsamples = size(p1.data, 1)
    ncoil = Int(p1.head.active_channels)
    pre = Int(p1.head.discard_pre)
    post = Int(p1.head.discard_post)
    center_sample = Int(p1.head.center_sample)
    T = eltype(p1.data)
    for p in profiles
        @argcheck size(p.data, 1) == nsamples "profiles have differing readout lengths; assemble manually"
        @argcheck Int(p.head.discard_pre) == pre && Int(p.head.discard_post) == post "profiles have differing discard_pre/discard_post; assemble manually"
        @argcheck Int(p.head.center_sample) == center_sample "profiles have differing center_sample; assemble manually"
        @argcheck Int(p.head.active_channels) == ncoil "profiles have differing active_channels; assemble manually"
    end

    rows = (pre + 1):(nsamples - post)
    row_offset = nkx ÷ 2 - center_sample
    kx_lo, kx_hi = first(rows) + row_offset, last(rows) + row_offset
    @argcheck 1 <= kx_lo && kx_hi <= nkx "readout centering places samples outside the encoded matrix ($(kx_lo):$(kx_hi) vs 1:$(nkx)); assemble manually"
    kx_sub = _positions_to_subsampling(nkx, collect(kx_lo:kx_hi))

    step1_ids = [Int(p.head.idx.kspace_encode_step_1) for p in profiles]
    ky_map, ky_uids = _compact_index_map(step1_ids)
    ky_positions = sort(ky_uids .- lim1.center .+ (nky ÷ 2) .+ 1)
    @argcheck first(ky_positions) >= 1 && last(ky_positions) <= nky "phase-encode centering places samples outside the encoded matrix; check enc_lim_kspace_encoding_step_1"
    ky_sub = _positions_to_subsampling(nky, ky_positions)

    kz_map = nothing
    kz_uids = nothing
    kz_sub = nothing
    if is3D
        step2_ids = [Int(p.head.idx.kspace_encode_step_2) for p in profiles]
        kz_map, kz_uids = _compact_index_map(step2_ids)
        kz_positions = sort(kz_uids .- lim2.center .+ (nkz ÷ 2) .+ 1)
        @argcheck first(kz_positions) >= 1 && last(kz_positions) <= nkz "kz centering places samples outside the encoded matrix; check enc_lim_kspace_encoding_step_2"
        kz_sub = _positions_to_subsampling(nkz, kz_positions)
    end

    # Batch dimensions, named per docs/src/high-level/nameddims.md ("Multi-slice data should
    # include :z as the first batch dimension"). Only counters that actually vary contribute a
    # dimension.
    batch_candidates = (
        (:z, p -> Int(p.head.idx.slice)),
        (:contrast, p -> Int(p.head.idx.contrast)),
        (:time, p -> Int(p.head.idx.phase)),
        (:repetition, p -> Int(p.head.idx.repetition)),
        (:set, p -> Int(p.head.idx.set)),
        (:average, p -> Int(p.head.idx.average)),
    )
    batch_names = Symbol[]
    batch_maps = Dict{Int, Int}[]
    batch_getters = Function[]
    batch_sizes = Int[]
    for (name, getter) in batch_candidates
        ids = [getter(p) for p in profiles]
        if length(unique(ids)) > 1
            @argcheck !is3D || name !== :z "3D k-space with more than one slice (multi-slab 3D) is not supported; filter raw.profiles and construct CartesianAcquisitionInfo slab-by-slab"
            map, uids = _compact_index_map(ids)
            push!(batch_names, name)
            push!(batch_maps, map)
            push!(batch_getters, getter)
            push!(batch_sizes, length(uids))
        end
    end

    kx_count = length(rows)
    ky_count = length(ky_uids)
    fourier_size = is3D ? (kx_count, ky_count, length(kz_uids)) : (kx_count, ky_count)
    ksp = zeros(T, fourier_size..., ncoil, batch_sizes...)

    for p in profiles
        ky_idx = ky_map[Int(p.head.idx.kspace_encode_step_1)]
        batch_idx = ntuple(j -> batch_maps[j][batch_getters[j](p)], length(batch_names))
        data = @view p.data[rows, :]
        if is3D
            kz_idx = kz_map[Int(p.head.idx.kspace_encode_step_2)]
            @views ksp[:, ky_idx, kz_idx, :, batch_idx...] .= data
        else
            @views ksp[:, ky_idx, :, batch_idx...] .= data
        end
    end

    fourier_names = is3D ? (:kx, :ky, :kz) : (:kx, :ky)
    dimnames_full = (fourier_names..., :coil, batch_names...)
    kspace_data = NamedDimsArray{dimnames_full}(ksp)

    subsampling = is3D ? (kx_sub, ky_sub, kz_sub) : (kx_sub, ky_sub)
    if all(s -> s isa Colon, subsampling)
        subsampling = nothing
    end

    return CartesianAcquisitionInfo(
        kspace_data;
        is3D,
        image_size = is3D ? (nkx, nky, nkz) : (nkx, nky),
        sensitivity_maps,
        subsampling,
    )
end

# Minimal non-Cartesian support: MRIBase's own `trajectory`/`rawdata` already assemble a single
# slice/contrast's samples in a consistent order, so this reuses them rather than re-deriving the
# (vendor-dependent) trajectory layout from individual profiles. Multiple slices/contrasts/
# repetitions are not collected into batch dimensions here (unlike the Cartesian path) — call
# this once per slice/contrast and combine the results yourself if you need more.
function _noncartesian_acquisition_info(
        raw::RawAcquisitionData; sensitivity_maps = nothing, slice::Integer = 1, contrast::Integer = 1,
    )
    tr = MRIBase.trajectory(raw; slice, contrast)
    nodes = kspaceNodes(tr) # (D, samples_per_profile, numProfiles), already in [-0.5, 0.5)
    D = size(nodes, 1)
    is3D = D == 3
    enc = Int.(raw.params["encodedSize"])
    image_size = is3D ? (enc[1], enc[2], enc[3]) : (enc[1], enc[2])
    trajectory = reshape(nodes, D, :)

    kspace_data = MRIBase.rawdata(raw; slice, contrast)

    return NonCartesianAcquisitionInfo(
        kspace_data;
        trajectory,
        sensitivity_maps,
        image_size,
    )
end

end # module MriReconstructionToolboxMRIBaseExt
