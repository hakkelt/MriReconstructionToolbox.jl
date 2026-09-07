"""
    DensityCompensation

Abstract base type for k-space density compensation methods.
"""
abstract type DensityCompensation end

"""
    PipeMenonDCF(; maxit::Int = 20) <: DensityCompensation

Iterative sample density compensation factor (DCF) estimation based on the algorithm of
Pipe & Menon (1999) using NFFT operators.
"""
Base.@kwdef struct PipeMenonDCF <: DensityCompensation
    maxit::Int = 20
end

"""
    VoronoiDCF(; bounds = nothing) <: DensityCompensation

Geometric sample density compensation calculating Voronoi cell areas for 2D k-space trajectories.
Points are clipped within `bounds = (xmin, xmax, ymin, ymax)` (defaulting to `(-0.5, 0.5, -0.5, 0.5)`).
"""
Base.@kwdef struct VoronoiDCF{B} <: DensityCompensation
    bounds::B = nothing
end

"""
    density_compensation(acq::AcquisitionInfo; method::DensityCompensation = PipeMenonDCF())

Compute the sample density compensation factors (DCF) for a non-Cartesian acquisition and return
a new `NonCartesianAcquisitionInfo` with the computed `dcf`.

Throws an `ArgumentError` if called on Cartesian acquisition data, as Cartesian data does not use
a density compensation factor.
"""
function density_compensation(
        acq::CartesianAcquisitionInfo;
        method::DensityCompensation = PipeMenonDCF(),
    )
    throw(
        ArgumentError(
            "Density compensation is only defined for non-Cartesian acquisitions. Cartesian acquisitions do not use a density compensation factor.",
        )
    )
end

function density_compensation(
        acq::NonCartesianAcquisitionInfo;
        method::DensityCompensation = PipeMenonDCF(),
    )
    dcf = compute_dcf(acq.trajectory, acq.image_size, method)
    return NonCartesianAcquisitionInfo(
        acq.kspace_data;
        trajectory = acq.trajectory,
        dcf = dcf,
        sensitivity_maps = acq.sensitivity_maps,
        image_size = acq.image_size,
        shifted_kspace_dims = acq.shifted_kspace_dims,
        shifted_image_dims = acq.shifted_image_dims,
    )
end

"""
    compute_dcf(trajectory::AbstractArray, image_size::Tuple, method::DensityCompensation)

Compute density compensation factor weights for the given trajectory and Cartesian image grid size.
"""
function compute_dcf(
        trajectory::AbstractArray{T},
        image_size::Tuple,
        method::PipeMenonDCF,
    ) where {T <: Real}
    mod = parentmodule(NFFTOp)
    NFFT = getfield(mod, :NFFT)
    NFFTTools = getfield(mod, :NFFTTools)

    traj_raw = trajectory isa NamedDimsArray ? unname(trajectory) : trajectory
    coord_dim = size(traj_raw, 1)
    ksp_shape = size(traj_raw)[2:end]
    traj_flat = reshape(traj_raw, coord_dim, :)

    plan = NFFT.plan_nfft(traj_flat, image_size)
    raw_dcf = NFFTTools.sdc(plan; iters = method.maxit)
    dcf_arr = reshape(raw_dcf, ksp_shape)

    if trajectory isa NamedDimsArray
        return NamedDimsArray{dimnames(trajectory)[2:end]}(dcf_arr)
    end
    return dcf_arr
end

function compute_dcf(
        trajectory::AbstractArray{T},
        image_size::Tuple,
        method::VoronoiDCF,
    ) where {T <: Real}
    traj_raw = trajectory isa NamedDimsArray ? unname(trajectory) : trajectory
    coord_dim = size(traj_raw, 1)
    @argcheck coord_dim == 2 "VoronoiDCF currently supports 2D trajectories (coord_dim == 2)"

    ksp_shape = size(traj_raw)[2:end]
    traj_flat = reshape(traj_raw, 2, :)
    bounds = method.bounds !== nothing ? method.bounds : (-0.5, 0.5, -0.5, 0.5)

    raw_dcf = _compute_voronoi_2d(traj_flat; bounds)
    dcf_arr = reshape(raw_dcf, ksp_shape)

    if trajectory isa NamedDimsArray
        return NamedDimsArray{dimnames(trajectory)[2:end]}(dcf_arr)
    end
    return dcf_arr
end

# 2D Voronoi polygon clipping helpers

function _clip_polygon_2d(poly::Vector{Tuple{Float64, Float64}}, n::Tuple{Float64, Float64}, c::Float64)
    out = Tuple{Float64, Float64}[]
    isempty(poly) && return out
    len = length(poly)
    for i in 1:len
        p1 = poly[i]
        p2 = poly[i == len ? 1 : i + 1]
        d1 = n[1] * p1[1] + n[2] * p1[2] - c
        d2 = n[1] * p2[1] + n[2] * p2[2] - c
        if d1 <= 1.0e-12
            push!(out, p1)
        end
        if (d1 < -1.0e-12 && d2 > 1.0e-12) || (d1 > 1.0e-12 && d2 < -1.0e-12)
            t = d1 / (d1 - d2)
            push!(out, (p1[1] + t * (p2[1] - p1[1]), p1[2] + t * (p2[2] - p1[2])))
        end
    end
    return out
end

function _polygon_area_2d(poly::Vector{Tuple{Float64, Float64}})
    len = length(poly)
    len < 3 && return 0.0
    area = 0.0
    for i in 1:len
        p1 = poly[i]
        p2 = poly[i == len ? 1 : i + 1]
        area += p1[1] * p2[2] - p2[1] * p1[2]
    end
    return abs(area) / 2.0
end

function _compute_voronoi_2d(pts::AbstractMatrix{T}; bounds = (-0.5, 0.5, -0.5, 0.5)) where {T <: Real}
    M = size(pts, 2)
    areas = zeros(T, M)
    xmin, xmax, ymin, ymax = Float64.(bounds)

    coords = [(Float64(pts[1, i]), Float64(pts[2, i])) for i in 1:M]

    Threads.@threads for i in 1:M
        p_i = coords[i]
        poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        dists = [(hypot(coords[j][1] - p_i[1], coords[j][2] - p_i[2]), j) for j in 1:M if j != i]
        sort!(dists; by = first)

        for (d, j) in dists
            isempty(poly) && break
            max_r = maximum(hypot(v[1] - p_i[1], v[2] - p_i[2]) for v in poly)
            d > 2.0 * max_r && break

            p_j = coords[j]
            n = (p_j[1] - p_i[1], p_j[2] - p_i[2])
            c = (p_j[1]^2 + p_j[2]^2 - (p_i[1]^2 + p_i[2]^2)) / 2.0
            poly = _clip_polygon_2d(poly, n, c)
        end
        areas[i] = T(_polygon_area_2d(poly))
    end
    return areas
end
