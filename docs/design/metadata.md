# Design note: metadata header and `ReconImage` (roadmap item 7)

Status: draft for review, 2026-10-09. Nothing here is implemented yet.

## Goals

1. `AcquisitionInfo` carries acquisition metadata: geometry, sequence parameters, and whatever
   else the raw data recorded.
2. `reconstruct` returns an `AbstractArray` that carries the geometry and the metadata of the
   acquisition it came from, so export (item 8) needs nothing else.
3. Users can attach their own tags to both.
4. Nothing about it depends on where the array lives (CPU or GPU).

## The header

A field `header::H` on both `CartesianAcquisitionInfo` and `NonCartesianAcquisitionInfo`, where
`H` is a `NamedTuple` of known keys plus one `extra::Dict{String, Any}` for everything else:

| key | type | meaning |
|---|---|---|
| `image_size` | `NTuple{N, Int}` | reconstruction matrix (moved here from its own field, see below) |
| `fov` | `NTuple{N, Float64}` or `nothing` | field of view in mm, per image axis |
| `voxel_size` | `NTuple{N, Float64}` or `nothing` | derived, `fov ./ image_size`, unless given |
| `orientation` | `SMatrix{3,3,Float64}` or `nothing` | columns: direction cosines of the image axes x, y, z (read, phase, slice) |
| `position` | `NTuple{3, Float64}` or `nothing` | centre of the volume, mm |
| `TE`, `TR`, `TI`, `flip_angle` | `Float64`, `Vector{Float64}` or `nothing` | ms, ms, ms, degrees |
| `field_strength` | `Float64` or `nothing` | T |
| `extra` | `Dict{String, Any}` | everything else (vendor parameters, protocol name, ...) |

Known keys are typed, so code that reads them is type-stable and JET-clean; unknown keys never
need a schema change. The constructor keyword is `header` and takes any `NamedTuple` (or
keywords); missing known keys are filled with `nothing`, an unknown key goes to `extra`.

`image_size` migrates into the header. Every read becomes the accessor `image_size(acq)` (`public`,
not exported); the field disappears and the constructor keyword `image_size = ...` stays as a
shorthand that writes the header entry. The package is unreleased, so this is a clean break with no
deprecation (`AGENTS.md`). About 150 uses in `src/` and `ext/` change mechanically.

`header(acq)` returns the header; `copy`-with-changes constructors (`acquisition_info_copy.jl`)
carry it over, and preprocessing that changes geometry (coil compression does not, cropping and
oversampling removal do) updates the affected keys. `adapt` leaves it untouched: it is host
metadata.

### Coordinate convention

KomaMRI writes MRD with identity direction cosines and does not model patient orientation;
MRIBase/MRIFiles pass the MRD `position` and `read_dir`/`phase_dir`/`slice_dir` through
unchanged. MRD and DICOM both use the patient coordinate system **LPS** (x towards the patient's
left, y posterior, z superior); NIfTI uses RAS. Ristretto stores **LPS**, as MRD gives it, and the
NIfTI export (item 8) flips the first two axes when it builds the affine. The docs state this once,
on the acquisition-data page.

## Tags

`settag!(x, key, value)`, `gettag(x, key[, default])` and `tags(x)` on both `AcquisitionInfo` and
`ReconImage`. Tags live in the header's `extra` dictionary under a `"tags"` sub-dictionary, so they
travel with the header from acquisition to image and into the exported files (as a JSON sidecar
for NIfTI, private tags or `ImageComments` for DICOM, user parameters for MRD).

`settag!` mutates the dictionary, which is shared by the acquisition and every image made from it
unless copied. `ReconImage` gets a copy of the header when it is created, so tagging an image never
changes its acquisition.

## `ReconImage`

```julia
struct ReconImage{T, N, A <: AbstractArray{T, N}, H <: NamedTuple} <: AbstractArray{T, N}
    data::A        # usually a NamedDimsArray; any array type, host or device
    header::H
end
```

- `AbstractArray` interface (`size`, `getindex`, `setindex!`, `IndexStyle`, `similar`) forwards to
  `data`, so it works wherever the plain image did.
- `parent(img)` returns `data`; `dimnames(img)` and `NamedDims.dim(img, :x)` forward to it.
- Keyword indexing as for a `NamedDimsArray`: `img[z = 5]`, `view(img; time = 1:10)`. A keyword
  index returns a `ReconImage` with the same header (the geometry of a sliced volume stays
  describable: `position` and `image_size` are updated for the axes that were indexed with a
  range or dropped with an integer). Positional indexing returns plain elements or arrays, as
  `NamedDimsArray` does.
- Broadcasting returns a plain array (`BroadcastStyle` of the data): arithmetic on images is not
  a reconstruction, so the result carries no claim about geometry.
- `Adapt.adapt_structure` adapts `data` and keeps `header`, so the GPU path needs no special case.
- `header(img)`, `image_size(img)`, `fov(img)`, ... read the header; `Array(img)` and
  `NamedDimsArray(img)` drop it.

### `DecomposedImage` as a special case

`DecomposedImage` becomes `ReconImage` with a `components::NamedTuple` header entry holding the
component images (each itself a `ReconImage` sharing the header). Its API stays: `img.total`,
`img.<component>`, `components(img)`, `total_image(img)`. `ReconImage` gets the `getproperty`
forwarding to components that `DecomposedImage` has today; with no `components` entry it only
exposes `header` and `data`.

Open choice for the review: keep a `DecomposedImage` alias (`const DecomposedImage = ReconImage{...}`
with a `components` header, for dispatch and docs) or drop the name. The note assumes the alias.

## What changes

- `src/acquisition_data/`: `header` field, constructors, `image_size(acq)` accessor, copy
  constructors, `show`.
- `src/reconstruction/reconstruct.jl`: `_present_image` wraps the result in `ReconImage` with the
  acquisition's header; `components.jl`: `DecomposedImage` rebuilt on `ReconImage`.
- `ext/RistrettoMRIBaseExt.jl` fills the header: `encodedFOV`/`reconFOV`, `encodedSize`/`reconSize`,
  sequence parameters (`TE`, `TR`, `TI`, `flipAngle_deg`), `H1resonanceFrequency_Hz` for the field,
  and `position`/`read_dir`/`phase_dir`/`slice_dir` of the first imaging profile of each slice.
  Multi-slice data gets one `position` per slice, stored as an `NTuple{3}` vector along `:z`.
- `simulate_acquisition` writes `fov` and `voxel_size` from the phantom.
- Tests in `test/test_metadata.jl`; docs on the acquisition-data and reconstruction pages.

## Questions for the review

1. Per-slice geometry: a vector of positions along `:z` (above), or one position plus a slice
   spacing? The vector is exact for non-equidistant slices; the pair matches NIfTI.
2. `DecomposedImage` alias kept or dropped?
3. Does keyword indexing need to update `position` (above), or is it acceptable for a slice of an
   image to carry the volume's geometry until export?
