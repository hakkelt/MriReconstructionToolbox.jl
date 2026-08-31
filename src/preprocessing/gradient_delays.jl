"""
    GradientDelayMethod

Abstract type representing trajectory gradient delay correction methods in non-Cartesian MRI.
"""
abstract type GradientDelayMethod end

"""
    OpposingSpokes <: GradientDelayMethod

Classical gradient delay estimation using cross-correlation / peak shifting of opposing radial spokes
(Peters et al. 2003, Block & Uecker 2011).
"""
struct OpposingSpokes <: GradientDelayMethod end

"""
    RING <: GradientDelayMethod

Radial Intersections for Navigation and Gradient delay estimation (Rosenzweig et al. 2019):
estimates the full anisotropic 2×2 delay tensor from spoke trajectory intersections.

!!! warning
    Not implemented yet — `estimate_gradient_delays` / `correct_gradient_delays` throw for this
    method. Use [`OpposingSpokes`](@ref) for an isotropic estimate.
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

Estimates the 2D gradient delay vector `(dx, dy)` in trajectory units.
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

function _estimate_delays_core(traj::AbstractArray, ksp::AbstractArray, ::OpposingSpokes)
    # traj has size (D, Nsamples, Nspokes, ...)
    Nsamples = size(traj, 2)
    Nspokes = size(traj, 3)

    # Compute spoke angles from endpoint or trajectory vector
    kx_end = traj[1, Nsamples, 1:Nspokes] .- traj[1, 1, 1:Nspokes]
    ky_end = traj[2, Nsamples, 1:Nspokes] .- traj[2, 1, 1:Nspokes]
    angles = atan.(ky_end, kx_end)

    # Combine multi-coil k-space by root-sum-of-squares over the coil dimension(s), which
    # for radial data are all dims past (samples, spokes). Reducing over the spoke axis
    # instead would collapse the very axis the per-spoke peak fit needs.
    ksp_mag = if ndims(ksp) >= 3
        coil_dims = Tuple(3:ndims(ksp))
        dropdims(sqrt.(sum(abs2, ksp; dims = coil_dims)); dims = coil_dims)
    else
        abs.(ksp)
    end
    ksp_mag_2d = reshape(ksp_mag, Nsamples, Nspokes)

    # Sample coordinates along readout
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

    # Fit dx * cos(θ) + dy * sin(θ) = shifts
    A = [cos.(angles) sin.(angles)]
    delay_vec = A \ shifts
    return (delay_vec[1], delay_vec[2])
end

function _estimate_delays_core(::AbstractArray, ::AbstractArray, ::RING)
    throw(ArgumentError("RING gradient-delay estimation is not implemented yet; use OpposingSpokes()."))
end

function _apply_gradient_delays(traj::AbstractArray, delays::Tuple{Real, Real})
    dx, dy = delays
    traj_corr = copy(traj)
    Nsamples = size(traj, 2)
    Nspokes = size(traj, 3)

    kx_end = traj[1, Nsamples, 1:Nspokes] .- traj[1, 1, 1:Nspokes]
    ky_end = traj[2, Nsamples, 1:Nspokes] .- traj[2, 1, 1:Nspokes]
    angles = atan.(ky_end, kx_end)

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
