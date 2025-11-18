"""
    shepp_logan(nx::Int, ny::Int; oversample=3) -> Array{ComplexF32, 2}
    shepp_logan(nx::Int, ny::Int, nz::Int; oversample=3) -> Array{ComplexF32, 3}

Generate a Shepp-Logan phantom for MRI simulation and reconstruction testing.

The Shepp-Logan phantom is a standard test image widely used in medical imaging research,
particularly for MRI and CT reconstruction algorithm validation. This implementation uses
the Toft modification of the original Shepp-Logan phantom, which provides better contrast
for modern reconstruction algorithms.

# Arguments
- `nx::Int`: Number of pixels/voxels in the x-dimension
- `ny::Int`: Number of pixels/voxels in the y-dimension
- `nz::Int`: (3D only) Number of voxels in the z-dimension

# Keywords
- `oversample::Int=3`: Oversampling factor for anti-aliasing. Higher values produce smoother
  edges but take longer to compute. Default of 3 provides good quality for most applications.

# Returns
- **2D**: A `ComplexF32` matrix of size `(nx, ny)` representing the phantom image
- **3D**: A `ComplexF32` array of size `(nx, ny, nz)` representing the phantom volume

# Details

## 2D Phantom
Uses the Toft modification of the classical Shepp-Logan phantom, consisting of 10 ellipses
with varying intensities representing different tissue types in a head cross-section.

## 3D Phantom
Generates a 3D extension using ellipsoids with:
- Field-of-view (FOV): 24 cm × 24 cm × 20 cm
- Modified intensities (×10) for inner structures to improve visibility
- Toft's intensity modifications for the head (1.0) and skull/brain (-0.8) regions

# References
- Shepp, L. A., & Logan, B. F. (1974). "The Fourier reconstruction of a head section."
  IEEE Transactions on Nuclear Science, 21(3), 21-43.
- Toft, P. (1996). "The Radon Transform - Theory and Implementation."
  PhD thesis, Technical University of Denmark.

# See Also
- [`simulate_acquisition`](@ref): Generate k-space data from a phantom
- [`coil_sensitivities`](@ref): Generate sensitivity maps for multi-coil simulation
- [`AcquisitionInfo`](@ref): Container for acquisition parameters
"""
function shepp_logan(dims...)
    if length(dims) == 2
        return shepp_logan_2d(dims...)
    elseif length(dims) == 3
        return shepp_logan_3d(dims...)
    else
        error("Shepp-Logan phantom is only defined for 2D and 3D.")
    end
end

function shepp_logan_2d(nx::Int, ny::Int; oversample=3)
    params = ImagePhantoms.ellipse_parameters(ImagePhantoms.SheppLoganToft())
    ob = ImagePhantoms.ellipse(params)
    x = range(-0.5, 0.5, nx)
    y = range(0.5 * (ny/nx), -0.5 * (ny/nx), ny)
    return ComplexF32.(ImagePhantoms.phantom(x, y, ob, oversample))
end

function shepp_logan_3d(nx::Int, ny::Int, nz::Int; oversample=3)
    fovs = (24, 24, 20)

    # Get parameters of the original Shepp-Logan phantom
    params = ImagePhantoms.ellipsoid_parameters( ; fovs)
    # Patch with Toft's modification
    params[1] = tuple(params[1][1:end-1]..., 1.0) # Change the intensity of the first ellipsoid (head)
    params[2] = tuple(params[2][1:end-1]..., -0.8) # Change the intensity of the second ellipsoid (skull/brain)
    for i in 3:length(params)
        params[i] = tuple(params[i][1:end-1]..., params[i][end]*10) # Change the intensity of the other ellipsoids (inner structures)
    end

    ob = ImagePhantoms.ellipsoid(params) # Vector of Ellipsoid objects
    dims = (nx, ny, nz)
    ig = ImageGeom( ; dims, deltas = fovs ./ dims )
    ax = [axes(ig)[1], reverse(axes(ig)[2]), axes(ig)[3]]
    return ComplexF32.(ImagePhantoms.phantom(ax..., ob, oversample))
end
