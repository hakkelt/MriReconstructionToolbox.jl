# Design note: metadata header and `ReconImage` (roadmap item 7)

Status: reviewed 2026-10-09; decisions below are final. Delete this file once items 7 and 8 are
implemented and documented.

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
| `spacing` | `NTuple{N, Float64}` or `nothing` | centre-to-centre distance of neighbouring voxels in mm, per image axis; along the slice axis this is the slice spacing (gap included). Derived as `fov ./ image_size` unless given |
| `slice_thickness` | `Float64` or `nothing` | excited slice thickness in mm, when it differs from the slice spacing |
| `orientation` | `SMatrix{3,3,Float64}` or `nothing` | columns: unit direction of the image axes x, y, z (read, phase, slice) |
| `offset` | `NTuple{3, Float64}` or `nothing` | position of the centre of the first voxel, index `(1, 1, 1)`, in mm |
| `TE`, `TR`, `TI`, `flip_angle` | `Float64`, `Vector{Float64}` or `nothing` | ms, ms, ms, degrees |
| `field_strength` | `Float64` or `nothing` | T |
| `extra` | `Dict{String, Any}` | everything else (vendor parameters, protocol name, tags, ...) |

The geometry is stored once per volume: `offset`, `orientation` and `spacing` give every voxel's
position, `offset + orientation * ((i .- 1) .* spacing)`, for multi-slice data as well (slices
are assumed equidistant and parallel; the MRIBase extension warns when the recorded slice
positions are not). This is the DICOM (`ImagePositionPatient`, `ImageOrientationPatient`,
`PixelSpacing` plus `SpacingBetweenSlices`) and NIfTI (affine) model, so export is a direct map.

Known keys are typed, so code that reads them is type-stable and JET-clean; unknown keys never
need a schema change. The constructor keyword is `header` and takes any `NamedTuple` (or
keywords); missing known keys are filled with `nothing`, an unknown key goes to `extra`.

`image_size` migrates into the header. Every read becomes the accessor `image_size(acq)` (`public`,
not exported); the field disappears and the constructor keyword `image_size = ...` stays as a
shorthand that writes the header entry. The package is unreleased, so this is a clean break with no
deprecation (`AGENTS.md`). About 150 uses in `src/` and `ext/` change mechanically.

`header(acq)` returns the header; copy-with-changes constructors (`acquisition_info_copy.jl`)
carry it over, and preprocessing that changes geometry (cropping, oversampling removal) updates the
affected keys. `adapt` leaves it untouched: it is host metadata.

### Coordinate convention

KomaMRI writes MRD with identity direction cosines and does not model patient orientation;
MRIBase/MRIFiles pass the MRD `position` and `read_dir`/`phase_dir`/`slice_dir` through
unchanged. MRD and DICOM both use the patient coordinate system **LPS** (x towards the patient's
left, y posterior, z superior); NIfTI uses RAS. Ristretto stores **LPS**, as MRD gives it, and the
NIfTI export (item 8) flips the first two axes when it builds the affine. MRD records the centre of
each slice; the extension converts that to `offset` (the first voxel's centre). The docs state the
convention once, on the acquisition-data page.

## Tags

`settag!(x, key, value)`, `gettag(x, key[, default])` and `tags(x)` on both `AcquisitionInfo` and
`ReconImage`. Tags live in the header's `extra` dictionary under a `"tags"` sub-dictionary, so they
travel with the header from acquisition to image and into the exported files (a JSON sidecar for
NIfTI, `ImageComments` or private tags for DICOM, user parameters for MRD).

`ReconImage` gets a copy of the header when it is created, so tagging an image never changes its
acquisition.

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
  index returns a `ReconImage` whose geometry describes the result: for every spatial axis indexed
  with a range starting at `k`, `offset` moves by `orientation[:, axis] * (k - 1) * spacing[axis]`
  and `image_size`/`fov` shrink to the range; an axis selected with an integer is dropped from
  `image_size`, `fov` and `spacing` while `offset` moves to that slice. Non-spatial axes
  (`:time`, `:coil`, ...) leave the geometry alone. Positional indexing returns plain elements or
  arrays, as `NamedDimsArray` does.
- Broadcasting returns a plain array: arithmetic on images is not a reconstruction, so the result
  carries no claim about geometry.
- `Adapt.adapt_structure` adapts `data` and keeps `header`, so the GPU path needs no special case.
- `header(img)`, `image_size(img)`, `fov(img)`, ... read the header; `Array(img)` and
  `NamedDimsArray(img)` drop it.

### Decomposed reconstructions

`DecomposedImage` is removed. A reconstruction with `Component`s returns a `ReconImage` whose
`data` is the total image and whose header holds a `components::NamedTuple` entry of the component
images (each a `ReconImage` with the same geometry). `img.lowrank` keeps working through
`getproperty` forwarding to `components`, and `components(img)` and `total_image(img)` stay. With
no `components` entry, `img.<name>` throws as for any struct.

## What changes

- `src/acquisition_data/`: `header` field, constructors, `image_size(acq)` accessor, copy
  constructors, `show`.
- `src/reconstruction/reconstruct.jl`: `_present_image` wraps the result in `ReconImage` with the
  acquisition's header; `components.jl`: `DecomposedImage` removed, components stored in the header.
- `ext/RistrettoMRIBaseExt.jl` fills the header: `encodedFOV`/`reconFOV`, `encodedSize`/`reconSize`,
  sequence parameters (`TE`, `TR`, `TI`, `flipAngle_deg`), the field strength from
  `H1resonanceFrequency_Hz`, and `offset`/`orientation`/slice spacing from the profiles'
  `position`/`read_dir`/`phase_dir`/`slice_dir`.
- `simulate_acquisition` writes `fov` and `spacing` from the phantom.
- Tests in `test/test_metadata.jl`; docs on the acquisition-data and reconstruction pages; every
  test, doc page, notebook and example that names `DecomposedImage`.
