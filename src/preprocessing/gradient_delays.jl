"""
    GradientDelayMethod

Abstract type representing trajectory gradient delay correction methods in non-Cartesian MRI.
"""
abstract type GradientDelayMethod end

"""
    OpposingSpokes <: GradientDelayMethod

Classical gradient delay estimation using cross-correlation / peak shifting of opposing radial spokes
(Peters et al. 2003, Block & Uecker 2011). Returns `(dx, dy)`.
"""
struct OpposingSpokes <: GradientDelayMethod end

"""
    RING <: GradientDelayMethod

Radial Intersections for Navigation and Gradient delay estimation (Rosenzweig et al. 2019):
estimates the full anisotropic 2×2 delay tensor from spoke trajectory intersections.
Returns a NamedTuple `(dx = Sxx, dy = Syy, dxy = Sxy)`.
"""
struct RING <: GradientDelayMethod end

"""
    correct_gradient_delays(acq::NonCartesianAcquisitionInfo; method = OpposingSpokes())

Estimates trajectory gradient delays from non-Cartesian k-space data and returns an updated
`NonCartesianAcquisitionInfo` with the corrected sampling trajectory.
"""
function correct_gradient_delays(
        acq::NonCartesianAcquisitionInfo;
        method::GradientDelayMethod = OpposingSpokes(),
    )
    delays = estimate_gradient_delays(acq; method)
    traj_corr = _apply_gradient_delays(acq.trajectory, delays)
    return NonCartesianAcquisitionInfo(acq; trajectory = traj_corr)
end

function correct_gradient_delays(acq::CartesianAcquisitionInfo; kwargs...)
    throw(ArgumentError("Gradient delay correction is only applicable to non-Cartesian acquisitions (got CartesianAcquisitionInfo)."))
end

"""
    estimate_gradient_delays(acq::NonCartesianAcquisitionInfo; method = OpposingSpokes())

Estimates the gradient delay parameters. Returns `(dx, dy)` for `OpposingSpokes()` or
`(dx = Sxx, dy = Syy, dxy = Sxy)` for `RING()`.
"""
function estimate_gradient_delays(
        acq::NonCartesianAcquisitionInfo;
        method::GradientDelayMethod = OpposingSpokes(),
    )
    traj = unname(acq.trajectory)
    ksp = unname(acq.kspace_data)
    @argcheck size(traj, 1) >= 2 "Trajectory must have at least 2 spatial dimensions (got $(size(traj, 1)))"
    return _estimate_delays_core(traj, ksp, method)
end

"""
    _spoke_angles(traj::AbstractArray)

Azimuthal angle of each spoke, from its first-to-last sample displacement in the kx/ky plane.
"""
function _spoke_angles(traj::AbstractArray)
    Nsamples = size(traj, 2)
    Nspokes = size(traj, 3)
    kx_end = traj[1, Nsamples, 1:Nspokes] .- traj[1, 1, 1:Nspokes]
    ky_end = traj[2, Nsamples, 1:Nspokes] .- traj[2, 1, 1:Nspokes]
    return atan.(ky_end, kx_end)
end

function _extract_spoke_angles_and_shifts(traj::AbstractArray, ksp::AbstractArray)
    Nsamples = size(traj, 2)
    Nspokes = size(traj, 3)

    angles = _spoke_angles(traj)

    ksp_mag = if ndims(ksp) >= 3
        coil_dims = Tuple(3:ndims(ksp))
        dropdims(sqrt.(sum(abs2, ksp; dims = coil_dims)); dims = coil_dims)
    else
        abs.(ksp)
    end
    ksp_mag_2d = reshape(ksp_mag, Nsamples, Nspokes)

    r0 = range(-0.5, 0.5, length = Nsamples)
    dr = step(r0)

    shifts = zeros(Float64, Nspokes)
    for s in 1:Nspokes
        peak_idx = argmax(ksp_mag_2d[:, s])
        if 1 < peak_idx < Nsamples
            y1 = Float64(ksp_mag_2d[peak_idx - 1, s])
            y2 = Float64(ksp_mag_2d[peak_idx, s])
            y3 = Float64(ksp_mag_2d[peak_idx + 1, s])
            denom = y1 - 2.0 * y2 + y3
            delta = abs(denom) > 1.0e-12 ? 0.5 * (y1 - y3) / denom : 0.0
            shifts[s] = r0[peak_idx] + delta * dr
        else
            shifts[s] = r0[peak_idx]
        end
    end
    return angles, shifts
end

function _estimate_delays_core(traj::AbstractArray, ksp::AbstractArray, ::OpposingSpokes)
    angles, shifts = _extract_spoke_angles_and_shifts(traj, ksp)
    A = [cos.(angles) sin.(angles)]
    delay_vec = A \ shifts
    return (delay_vec[1], delay_vec[2])
end

function _estimate_delays_core(traj::AbstractArray, ksp::AbstractArray, ::RING)
    angles, shifts = _extract_spoke_angles_and_shifts(traj, ksp)
    A = [cos.(angles) .^ 2 sin.(angles) .^ 2 (2.0 .* cos.(angles) .* sin.(angles))]
    p = A \ shifts
    return (dx = p[1], dy = p[2], dxy = p[3])
end

function _apply_gradient_delays(traj::AbstractArray, delays::Tuple{Real, Real})
    dx, dy = delays
    traj_corr = copy(traj)
    Nspokes = size(traj, 3)

    angles = _spoke_angles(traj)

    for s in 1:Nspokes
        shift_x = dx * cos(angles[s])
        shift_y = dy * sin(angles[s])
        traj_corr[1, :, s] .-= shift_x
        traj_corr[2, :, s] .-= shift_y
    end

    if traj isa NamedDimsArray
        return NamedDimsArray{dimnames(traj)}(traj_corr)
    else
        return traj_corr
    end
end

function _apply_gradient_delays(traj::AbstractArray, delays::NamedTuple)
    traj_corr = copy(traj)
    Nspokes = size(traj, 3)

    angles = _spoke_angles(traj)

    for s in 1:Nspokes
        θ = angles[s]
        shift = delays.dx * cos(θ)^2 + delays.dy * sin(θ)^2 + 2.0 * delays.dxy * cos(θ) * sin(θ)
        traj_corr[1, :, s] .-= shift * cos(θ)
        traj_corr[2, :, s] .-= shift * sin(θ)
    end

    if traj isa NamedDimsArray
        return NamedDimsArray{dimnames(traj)}(traj_corr)
    else
        return traj_corr
    end
end
