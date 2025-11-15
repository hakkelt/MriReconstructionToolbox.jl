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
