# Writing a `ReconImage` to NIfTI, DICOM and MRD. The writers live in package extensions; what
# they share, the layout of the volume and its voxel-to-patient affine, is here.

"""
    write_nifti(path, img::ReconImage; sidecar = true) -> path

Write `img` to the NIfTI-1 file `path` (`.nii`, or `.nii.gz` to compress). Needs `using NIfTI`.

The first three axes of the file are the image axes x, y and z: a 2D image gets a z axis of
length one, or its slice axis when the axis after the image axes is named `:z` or `:slice`. Any
further axes (time, echoes, ...) follow in order. Complex images are written as `complex64` or
`complex128`; write `abs.(img)` for a magnitude image.

The affine (`sform`) maps voxel indices to the scanner's RAS coordinates, converted from the
header's LPS geometry (see [`Header`](@ref)). With `sidecar = true`, a JSON file next to `path`
holds the sequence parameters under their BIDS names (`EchoTime`, `RepetitionTime`, ... in
seconds), the other header entries, and the tags.
"""
function write_nifti end

"""
    write_dicom(dir, img::ReconImage; series_description = "Ristretto", series_number = 1) -> files

Write `img` as one DICOM MR image series to the directory `dir`, one file per slice and per
index of any non-image axis, and return the file names. Needs `using DICOM`.

The pixel data is the magnitude of the image, stored as 16-bit integers with a rescale slope, so
`RescaleSlope * stored` recovers it to 1 part in 65535 of the maximum. Each file carries the
slice position and orientation (`ImagePositionPatient`, `ImageOrientationPatient`, LPS like the
header), `PixelSpacing`, `SliceThickness`, the sequence parameters, and the tags as JSON in
`ImageComments`. Patient and study fields are left empty except for generated UIDs; fill them in
with a DICOM tool before archiving.
"""
function write_dicom end

"""
    write_mrd(path, img::ReconImage; group = "image_0") -> path

Write `img` as MRD (ISMRMRD HDF5) images to `path`, under `/dataset/<group>`, with an
`ismrmrdHeader` XML holding the field of view, the matrix size and the sequence parameters.
Needs `using MRIFiles`.

A 3D image is one MRD image; a 2D image is one MRD image per slice. Each non-image axis index is
another image, numbered by `image_index`, with a `:time` index also in the `phase` counter.
Geometry goes to `position` (the centre of each image) and `read_dir`/`phase_dir`/`slice_dir`;
tags go to each image's meta attributes.
"""
function write_mrd end

for (f, pkg) in ((:write_nifti, "NIfTI"), (:write_dicom, "DICOM"), (:write_mrd, "MRIFiles"))
    @eval $f(args...; kwargs...) = throw(ArgumentError(string($(string(f)), " needs the ", $pkg, " package: run `using ", $pkg, "` first")))
end

# ---------------------------------------------------------------- layout and geometry

# `img` as a host `Array` whose first three axes are x, y and z, the names of its other axes, and
# whether a 2D image's z axis holds its slices (`true`) or was added (`false`; length one).
function _export_volume(img::ReconImage)
    data = Array(img)
    nd = _spatial_ndims(img)
    @argcheck nd in (2, 3) "only 2D and 3D images can be exported, this one has $nd image axes"
    names = parent(img) isa NamedDimsArray ? collect(Symbol, dimnames(img)) : fill(:_, ndims(data))
    nd == 3 && return data, names[4:end], false
    if ndims(data) >= 3 && names[3] in (:z, :slice)
        return data, names[4:end], true
    end
    vol = reshape(data, size(data, 1), size(data, 2), 1, size(data)[3:end]...)
    return vol, names[3:end], false
end

# The 4×4 map from 0-based voxel indices (x, y, z) to LPS coordinates in mm. What the header
# lacks is filled in so that a file can still be written: identity orientation, 1 mm voxels, a
# slice spacing equal to the slice thickness, and the image centre at the origin.
function _lps_affine(h::Header, vol_size::NTuple{3, Int}, nd::Integer)
    R = something(h.orientation, Matrix{Float64}(I, 3, 3))
    sp = _export_spacing(h, vol_size, nd)
    M = R * Diagonal(collect(sp))
    offset = isnothing(h.offset) ? -(M * collect(Float64, vol_size .÷ 2)) : collect(h.offset)
    return [M offset; 0 0 0 1]
end

function _export_spacing(h::Header, vol_size, nd)
    sp = h.spacing
    if isnothing(sp) || length(sp) != nd
        fov = h.fov
        sp = !isnothing(fov) && length(fov) == nd ? fov ./ vol_size[1:nd] : nothing
    end
    if isnothing(sp)
        @warn "the header has no spacing or fov; exporting with 1 mm voxels" maxlog = 1
        sp = ntuple(_ -> 1.0, nd)
    end
    nd == 3 && return Tuple(Float64.(sp))
    return (Float64(sp[1]), Float64(sp[2]), something(h.slice_spacing, h.slice_thickness, 1.0))
end

# ---------------------------------------------------------------- metadata as text

# Sequence parameters under their BIDS sidecar names, in seconds where BIDS uses seconds.
function _bids_parameters(h::Header)
    d = Dict{String, Any}()
    seconds(ms) = ms .* 1.0e-3
    TE, TR, TI, flip, B0, thickness = h.TE, h.TR, h.TI, h.flip_angle, h.field_strength, h.slice_thickness
    isnothing(TE) || (d["EchoTime"] = seconds(TE))
    isnothing(TR) || (d["RepetitionTime"] = seconds(TR))
    isnothing(TI) || (d["InversionTime"] = seconds(TI))
    isnothing(flip) || (d["FlipAngle"] = flip)
    isnothing(B0) || (d["MagneticFieldStrength"] = B0)
    isnothing(thickness) || (d["SliceThickness"] = thickness)
    for (k, v) in h.extra
        d[string(k)] = v
    end
    isempty(h.tags) || (d["Tags"] = h.tags)
    return d
end

# A minimal JSON writer for the values a header holds; anything else is written as its string.
_json(x) = sprint(_json, x)
_json(io::IO, x::AbstractString) = (print(io, '"'); _json_escape(io, x); print(io, '"'))
_json(io::IO, x::Symbol) = _json(io, string(x))
_json(io::IO, x::Bool) = print(io, x)
_json(io::IO, x::Integer) = print(io, x)
_json(io::IO, x::AbstractFloat) = isfinite(x) ? print(io, x) : print(io, "null")
_json(io::IO, ::Nothing) = print(io, "null")
_json(io::IO, x::Union{AbstractVector, Tuple}) =
    (print(io, '['); join(io, (_json(v) for v in x), ", "); print(io, ']'))
function _json(io::IO, x::AbstractDict)
    print(io, '{')
    join(io, (_json(string(k)) * ": " * _json(v) for (k, v) in sort!(collect(x); by = p -> string(first(p)))), ", ")
    return print(io, '}')
end
_json(io::IO, x) = _json(io, string(x))

function _json_escape(io::IO, s::AbstractString)
    for c in String(s)::String
        if c == '"' || c == '\\'
            print(io, '\\', c)
        elseif c < ' '
            print(io, "\\u", string(UInt16(c); base = 16, pad = 4))
        else
            print(io, c)
        end
    end
    return nothing
end
