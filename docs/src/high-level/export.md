# Export

A [`ReconImage`](@ref) carries its geometry and metadata in its [`Header`](@ref), so writing it
to a file format needs nothing else. Each format is a package extension, loaded with its package:

| Function | Format | Load |
|---|---|---|
| [`write_nifti`](@ref) | NIfTI-1 (`.nii`, `.nii.gz`) and a BIDS-style JSON sidecar | `using NIfTI` |
| [`write_dicom`](@ref) | DICOM MR image series, one file per slice and frame | `using DICOM` |
| [`write_mrd`](@ref) | MRD (ISMRMRD HDF5) images | `using MRIFiles` |

```julia
using Ristretto, NIfTI

img = reconstruct(acq, IterativeReconstruction(TotalVariation2D(1e-3)))
settag!(img, :reader, "A")
write_nifti("recon.nii.gz", img)      # also writes recon.json
```

## Geometry

The header stores positions in the patient coordinate system LPS, as MRD and DICOM do (see
[Metadata header](@ref)). DICOM and MRD files receive it unchanged; the NIfTI affine maps voxel
indices to RAS, negating the first two coordinates. A 2D image is written as a volume with one
slice, or with its slices when the axis after `x` and `y` is named `:z` or `:slice`; any further
axes (time, echoes) follow.

What the header lacks is filled in so that a file can always be written: an identity
orientation, 1 mm voxels (with a warning), a slice spacing equal to the slice thickness, and the
image centre at the scanner origin. An acquisition read from MRD through the MRIBase extension
has all of it.

## What goes where

| Header | NIfTI | DICOM | MRD |
|---|---|---|---|
| `spacing`, `orientation`, `offset` | `sform` affine, `pixdim` | `PixelSpacing`, `ImageOrientationPatient`, `ImagePositionPatient` | `field_of_view`, `read_dir`/`phase_dir`/`slice_dir`, `position` |
| `slice_spacing`, `slice_thickness` | third axis of the affine | `SpacingBetweenSlices`, `SliceThickness` | slice `position` |
| `TE`, `TR`, `TI`, `flip_angle`, `field_strength` | sidecar, BIDS names, in seconds | `EchoTime`, `RepetitionTime`, ... | `ismrmrdHeader` XML |
| other keys | sidecar | — | — |
| tags | sidecar, `Tags` | `ImageComments` as JSON | meta attributes of each image |

NIfTI and MRD keep complex images complex; DICOM stores the magnitude as 16-bit integers with a
rescale slope.

```@docs
write_nifti
write_dicom
write_mrd
```
