# Phantoms and coil sensitivity maps for the benchmark case catalog (`cases.jl`).
#
# Everything here is built from GeometricMedicalPhantoms and closed-form maps only -- no MRT
# simulation code -- so two MRT checkouts measured against each other receive byte-identical
# inputs. Images and maps are `ComplexF32`, the precision every toolkit reconstructs in.

"""
    SHEPP_LOGAN_FOV

Field of view `create_shepp_logan_phantom` uses by default, in its own units; the coil maps are
laid out on the same grid.
"""
const SHEPP_LOGAN_FOV = 20.0

"""
    shepp_logan_2d(n) -> Matrix{ComplexF32}

An `n × n` axial Shepp-Logan slice with MRI intensities (`MRISheppLoganIntensities`).
"""
shepp_logan_2d(n::Int) = create_shepp_logan_phantom(n, n, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)

"""
    shepp_logan_volume(n) -> Array{ComplexF32, 3}

An `n × n × n` Shepp-Logan volume with MRI intensities. The multislice case takes its slices from
this volume, so the two share one object.
"""
shepp_logan_volume(n::Int) = create_shepp_logan_phantom(n, n, n; ti = MRISheppLoganIntensities(), eltype = ComplexF32)

"""
    coil_maps_2d(nx, ny, ncoils) -> Array{ComplexF32, 3}

`(nx, ny, ncoils)` Gaussian sensitivities centred on a ring at 80% of the half field of view, each
with a constant phase `2π(c - 1)/ncoils`. Deliberately **not** normalised (`Σ|Sᶜ|² ≠ 1`): a
normalised map set gives MRT a known operator norm and so a free step size, while MRIReco's
`SensitivityOp` and BART's `pics` normalise nothing.
"""
function coil_maps_2d(nx::Int, ny::Int, ncoils::Int)
    fov = SHEPP_LOGAN_FOV
    x = range(-fov / 2, fov / 2; length = nx)
    y = range(-fov / 2, fov / 2; length = ny)
    maps = zeros(ComplexF32, nx, ny, ncoils)
    for c in 1:ncoils
        phase = 2π * (c - 1) / ncoils
        cx, cy = 0.8 * fov / 2 * cos(phase), 0.8 * fov / 2 * sin(phase)
        for j in 1:ny, i in 1:nx
            d2 = (x[i] - cx)^2 + (y[j] - cy)^2
            maps[i, j, c] = exp(-d2 / (fov^2 * 0.25)) * cis(phase)
        end
    end
    return maps
end

"""
    coil_maps_3d(nx, ny, nz, ncoils; rings = 2) -> Array{ComplexF32, 4}

`(nx, ny, nz, ncoils)` Gaussian sensitivities on `rings` rings stacked along `z` (at ±40% of the
half field of view for two rings), `ncoils ÷ rings` coils per ring, each ring rotated by half a
coil spacing against the previous one.

These are genuinely 3D: sensitivity varies along `z` as much as in-plane, so a 3D encode has coil
information along its third axis. MRT's own `coil_sensitivities` repeats 2D maps along `z`, which
gives a 3D parallel-imaging problem no leverage along `z` at all.
"""
function coil_maps_3d(nx::Int, ny::Int, nz::Int, ncoils::Int; rings::Int = 2)
    ncoils % rings == 0 || throw(ArgumentError("ncoils = $ncoils is not divisible by rings = $rings"))
    per_ring = ncoils ÷ rings
    fov = SHEPP_LOGAN_FOV
    x = range(-fov / 2, fov / 2; length = nx)
    y = range(-fov / 2, fov / 2; length = ny)
    z = range(-fov / 2, fov / 2; length = nz)
    zc = rings == 1 ? [0.0] : collect(range(-0.4 * fov / 2, 0.4 * fov / 2; length = rings))
    maps = zeros(ComplexF32, nx, ny, nz, ncoils)
    for r in 1:rings, k in 1:per_ring
        c = (r - 1) * per_ring + k
        phase = 2π * (k - 1) / per_ring + π * (r - 1) / per_ring
        cx, cy = 0.8 * fov / 2 * cos(phase), 0.8 * fov / 2 * sin(phase)
        for l in 1:nz, j in 1:ny, i in 1:nx
            d2 = (x[i] - cx)^2 + (y[j] - cy)^2 + (z[l] - zc[r])^2
            maps[i, j, l, c] = exp(-d2 / (fov^2 * 0.25)) * cis(2π * (c - 1) / ncoils)
        end
    end
    return maps
end

"""
    torso_cine(n, nframes; heart_rate = 75.0, respiratory_rate = 15.0) -> Array{ComplexF32, 3}

`(n, n, nframes)` coronal torso slice over **one cardiac cycle**, from GeometricMedicalPhantoms'
torso phantom driven by its cardiac and respiratory signals, sampled at `nframes` evenly spaced
instants of that cycle. The breathing over the same 0.8 s is slow, so the motion is dominated by the
beating heart, as in a breath-held cine.

A smooth, slowly varying phase is added: a real cine has one, and a purely real phantom lets a
magnitude prior do work no real reconstruction could rely on.
"""
function torso_cine(n::Int, nframes::Int; heart_rate::Real = 75.0, respiratory_rate::Real = 15.0)
    cycle = 60 / heart_rate
    fs = nframes / cycle
    _, chambers = generate_cardiac_signals(cycle, fs, heart_rate)
    _, lung = generate_respiratory_signal(cycle, fs, respiratory_rate)
    frames = create_torso_phantom(
        n, n, :coronal;
        respiratory_signal = lung[1:nframes], cardiac_volumes = map(v -> v[1:nframes], chambers),
        eltype = ComplexF32,
    )
    u = range(-1, 1; length = n)
    phase = [cis(Float32(0.3π * ux + 0.2π * uy^2)) for ux in u, uy in u]
    return frames .* phase
end

"""
    centred_fft(x, dims) / centred_ifft(x, dims)

The orthonormally unscaled DFT with the origin at the array centre on both sides,
`fftshift(fft(ifftshift(x, dims), dims), dims)`; the convention MRT's `shifted_image_dims` and
every competitor's layout converter assume.
"""
centred_fft(x, dims) = fftshift(fft(ifftshift(x, dims), dims), dims)
centred_ifft(x, dims) = fftshift(ifft(ifftshift(x, dims), dims), dims)
