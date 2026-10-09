# Design note: metadata header and `ReconImage` (roadmap item 7)

Status: reviewed 2026-10-09; the decisions below are final. Delete this file once items 7 and 8
are implemented and documented.

## Goals

1. `AcquisitionInfo` can carry acquisition metadata: geometry, sequence parameters, and whatever
   else the raw data recorded. Metadata is optional and may be incomplete.
2. `reconstruct` returns an `AbstractArray` that carries the geometry and the metadata of the
   acquisition it came from, so export (item 8) needs nothing else.
3. Users can attach their own tags to both.
4. Nothing about it depends on where the array lives (CPU or GPU).

## `Header`

```julia
mutable struct Header <: AbstractDict{Symbol, Any}
    fov::Union{Nothing, NTuple{2, Float64}, NTuple{3, Float64}}
    spacing::Union{Nothing, NTuple{2, Float64}, NTuple{3, Float64}}
    slice_spacing::Union{Nothing, Float64}
    slice_thickness::Union{Nothing, Float64}
    orientation::Union{Nothing, Matrix{Float64}}
    offset::Union{Nothing, NTuple{3, Float64}}
    TE::Union{Nothing, Float64, Vector{Float64}}
    TR::Union{Nothing, Float64}
    TI::Union{Nothing, Float64, Vector{Float64}}
    flip_angle::Union{Nothing, Float64, Vector{Float64}}
    field_strength::Union{Nothing, Float64}
    tags::Dict{String, Any}
    extra::Dict{Symbol, Any}
end
```

- **Construction:** `Header(; kwargs...)` takes any keywords. A known key fills its field, after
  conversion and checking (below). Any other key goes to `extra`, stored as given.
  `Header(pairs)` and `Header(nt::NamedTuple)` do the same.
- **Incomplete by default:** every known field may be `nothing`; nothing is required.
- **Dictionary interface:** `Header` is an `AbstractDict{Symbol, Any}` over the known fields that
  are set plus `extra`:
  - `h[:fov]` and `h[:protocol]` both work and throw a `KeyError` when the key is absent;
  - `get(h, :k, default)`, `haskey`, `keys`, `length` and iteration see only keys that are set;
  - `h[:k] = v` sets a known field (with checks) or an `extra` entry;
  - `delete!` resets a known field to `nothing` or removes an `extra` entry.
- **Property access:** only the known keys are properties. `h.fov` is plain field access and is
  `nothing` when unset. `h.fov = v` converts and checks like `h[:fov] = v`. A misspelled or
  unknown property throws.
- **Checks on set:**
  - `fov`, `spacing` and `offset` become `Float64` tuples, and `offset` must have 3 entries;
  - `orientation` must be 3×3, its columns the unit directions of the image axes x, y and z
    (read, phase, slice);
  - scalars become `Float64`; `tags` becomes `Dict{String, Any}`.
- **Consistency warning:** when `fov`, `spacing` and the image size are all known, a mismatch
  between them is warned about once, not rejected, because a scanner's `fov` may describe the
  oversampled grid.
- **`show`:** compact, listing only the keys that are set.
- **`copy(h)`:** deep-copies `tags` and `extra`.

### Units and coordinates

Lengths are in mm, times in ms, angles in degrees and the field in T.

- `spacing` is the centre-to-centre distance of neighbouring voxels along each image axis.
- `slice_spacing` is the distance between the slices of a multi-slice 2D acquisition, along
  `orientation[:, 3]`.
- `offset` is the centre of the first voxel, index `(1, 1, 1)`.
- The geometry is stored once per volume: the position of voxel `i` is
  `offset + orientation * ((i .- 1) .* spacing)`. Slices are assumed equidistant and parallel; the
  MRIBase extension warns when the recorded slice positions are not.

KomaMRI writes MRD with identity direction cosines and does not model patient orientation.
MRIBase/MRIFiles pass the MRD `position` and `read_dir`/`phase_dir`/`slice_dir` through
unchanged. MRD and DICOM both use the patient coordinate system **LPS** (x towards the patient's
left, y posterior, z superior), while NIfTI uses RAS.

- Ristretto stores **LPS**, as MRD gives it.
- The NIfTI export (item 8) flips the first two axes when it builds the affine.
- MRD records the centre of each slice; the extension converts that to `offset`.
- The docs state the convention once, on the acquisition-data page.

## On `AcquisitionInfo`

`image_size` stays a field of its own. It is a reconstruction parameter: required, inferred from
the k-space, the subsampling or the maps, and checked against them, while the header is optional
descriptive metadata. `image_size(acq)` (`public`, not exported) reads the field.

A `header` field is added to both acquisition types, with the constructor keyword
`header = nothing`.

- **No header given:** an empty `Header()` is created.
- **`NamedTuple` or another dictionary given:** it is converted to a `Header`.
- **`Header` given:** it is stored as given, not copied, as the k-space array is. Tagging the
  user's `Header` object afterwards is therefore visible on the acquisition.

Copy-with-changes constructors (`AcquisitionInfo(acq; ...)`) and preprocessing share the input's
header unless a new one is passed: a prewhitened or coil-compressed copy describes the same scan.
Preprocessing that changes geometry (cropping, oversampling removal) passes an updated copy.
`adapt` leaves the header untouched, since it is host metadata.

`header(acq)` returns the header.

## Tags

`settag!(x, key, value)`, `gettag(x, key[, default])` and `tags(x)` work on both `AcquisitionInfo`
and `ReconImage`. Tags live in the header's `tags` field, so they travel with the header from
acquisition to image and into the exported files:

- a JSON sidecar for NIfTI;
- `ImageComments` or private tags for DICOM;
- user parameters for MRD.

## `ReconImage`

```julia
struct ReconImage{T, N, A <: AbstractArray{T, N}, C <: Union{Nothing, NamedTuple}} <: AbstractArray{T, N}
    data::A          # usually a NamedDimsArray; any array type, host or device
    header::Header   # always its own copy
    components::C    # the component images of a decomposition, or nothing
    spatial_ndims::Int   # the leading axes of `data` that are image axes
end
```

`spatial_ndims` is needed because the dimension names cannot tell a 3D image from a multi-slice 2D
one (both are `(:x, :y, :z)`). `reconstruct` sets it from the acquisition; otherwise it defaults to
the length of `spacing` or `fov`, or the number of leading axes named `:x`, `:y`, `:z`.

- `reconstruct` returns one carrying `copy(header(acq))`. Its image size is the shape of `data`'s
  spatial axes, and `image_size(img)` reads it from there. When the header has a `fov` but no
  `spacing`, `spacing` is derived then.
- Tagging an image never changes its acquisition.
- **Array interface:** `size`, `getindex`, `setindex!`, `IndexStyle` and `similar` forward to
  `data`. `Array(img)` and `unname(img)` return a plain array. Broadcasting acts on `data` and
  returns a plain (possibly named) array: arithmetic on images is not a reconstruction, so the
  result makes no claim about geometry.
- **Wrapped data:** `parent(img)` returns `data`; `dimnames(img)` and `NamedDims.dim(img, :x)`
  forward to it.
- **Keyword indexing,** as for a `NamedDimsArray`: `img[z = 5]`, `view(img; time = 1:10)`. A
  keyword index returns a `ReconImage` whose geometry describes the result:
  - a spatial axis indexed with a range starting at `k` moves `offset` by
    `orientation[:, axis] * (k - 1) * spacing[axis]`, and `fov` shrinks to the range (a stepped
    range also scales `spacing`);
  - a spatial axis selected with an integer is dropped from `fov` and `spacing`. `offset` moves to
    that slice, the dropped spacing becomes `slice_thickness`, and the orientation's columns are
    reordered so the kept axes come first;
  - `z` of a multi-slice 2D image moves `offset` by `slice_spacing`;
  - non-spatial axes (`:time`, `:coil`, ...) leave the geometry alone;
  - the original image is never modified.

  Positional indexing returns plain elements or arrays, as `NamedDimsArray` does.
- **Device moves:** `Adapt.adapt_structure` adapts `data` and the components and keeps `header`,
  so the GPU path needs no special case.

### Decomposed reconstructions

`DecomposedImage` is removed. A reconstruction with `Component`s returns a `ReconImage` whose
`data` is the total image. Its `components` field is a `NamedTuple` of the component images, each
a `ReconImage` with its own copy of the header. `img.lowrank` keeps working, through
`getproperty` forwarding to `components`, and is concretely typed. `components(img)` and
`total_image(img)` stay. A component may not be named `data`, `header` or `components`.

## What changes

- **`src/acquisition_data/`:**
  - `header.jl` (`Header`);
  - the `header` field and keyword on both acquisition types;
  - the `image_size(acq)` accessor;
  - copy constructors;
  - `show`.
- **`src/reconstruction/`:**
  - `recon_image.jl` (`ReconImage`);
  - `reconstruct.jl`, where `reconstruct` wraps its result;
  - `components.jl` and `task_splitting/stacking.jl`, where `DecomposedImage` is replaced.
- **`ext/RistrettoMRIBaseExt.jl`** fills the header:
  - `fov` from `encodedFOV`/`reconFOV`;
  - `TE`, `TR`, `TI` and `flip_angle` from the sequence parameters;
  - `field_strength` from `H1resonanceFrequency_Hz`;
  - `offset`, `orientation` and `slice_spacing` from the profiles' `position`, `read_dir`,
    `phase_dir` and `slice_dir`.
- `simulate_acquisition` writes nothing: a phantom array carries no physical size.
- **Tests:** `test/test_metadata.jl`.
- **Docs:** the acquisition-data and reconstruction pages; every test, doc page, notebook and
  example that names `DecomposedImage`.
- **Exports:** `Header`, `ReconImage`, `settag!`, `gettag`, `tags`; `header` and `image_size` are
  `public`.
