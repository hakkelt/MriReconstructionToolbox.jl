module RistrettoNIfTIExt

using Ristretto
using Ristretto: ReconImage, header, _spatial_ndims, _export_volume, _lps_affine, _bids_parameters, _json
using NIfTI: NIfTI, NIVolume, niwrite

# LPS to RAS: the first two patient axes point the other way.
const _LPS_TO_RAS = [-1.0 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 1]

function Ristretto.write_nifti(path::AbstractString, img::ReconImage; sidecar::Bool = true)
    vol, _, _ = _export_volume(img)
    h = header(img)
    affine = _LPS_TO_RAS * _lps_affine(h, size(vol)[1:3], _spatial_ndims(img))
    voxel_size = Tuple(Float32.(sqrt.(vec(sum(abs2, affine[1:3, 1:3]; dims = 1)))))
    nii = NIVolume(vol; voxel_size, orientation = Matrix{Float32}(affine[1:3, :]), descrip = "Ristretto")
    # Only the sform is meaningful; a qform code with a zero quaternion would claim no rotation.
    nii.header.qform_code = Int16(0)
    niwrite(path, nii)
    if sidecar
        write(_sidecar_path(path), _json(_bids_parameters(h)), "\n")
    end
    return path
end

_sidecar_path(path) = replace(path, r"\.nii(\.gz)?$" => "") * ".json"

end
