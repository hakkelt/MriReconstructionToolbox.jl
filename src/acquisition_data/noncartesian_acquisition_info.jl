"""
    NonCartesianAcquisitionInfo(
            kspace_data;
            trajectory,
            dcf=nothing,
            sensitivity_maps=nothing,
            image_size,
            shifted_kspace_dims::Tuple=(),
            shifted_image_dims::Tuple=(),
    )

Container for non-Cartesian MRI acquisition settings.

The trajectory stores coordinate axes in its first dimension. Its remaining
dimensions must match the non-coil k-space sample layout.
"""
struct NonCartesianAcquisitionInfo{K,T,D,S,I,SD,ID} <: AcquisitionInfo
    kspace_data::K
    trajectory::T
    dcf::D
    sensitivity_maps::S
    image_size::I
    shifted_kspace_dims::SD
    shifted_image_dims::ID
    is3D::Bool

    function NonCartesianAcquisitionInfo(ksp, traj, dcf, smaps, img_size, sK, sI)
        @argcheck !isnothing(traj) "trajectory must be provided"
        nd = ndims(traj)
        @argcheck nd > 1 "trajectory must have at least 2 dimensions"
        coord_dim = size(traj, 1)
        @argcheck coord_dim == 2 || coord_dim == 3 "the first dimension of trajectory must be 2 or 3"
        is3D = coord_dim == 3

        if traj isa NamedDimsArray
            first_name = dimnames(traj)[1]
            @argcheck first_name == :dim || first_name == :coord "trajectory first dimension must be :dim or :coord"
        end

        @argcheck !isnothing(img_size) "image_size must be provided"
        @argcheck length(img_size) == (is3D ? 3 : 2) "image_size length must match trajectory dimensionality"

        if !isnothing(ksp)
            fourier_dims = ndims(traj) - 1
            @argcheck size(ksp)[1:fourier_dims] == size(traj)[2:end] "k-space data dimensions must match trajectory sample dimensions"
            if ksp isa NamedDimsArray
                @argcheck traj isa NamedDimsArray "trajectory must be a NamedDimsArray when k-space data is a NamedDimsArray"
                @argcheck dimnames(ksp)[1:fourier_dims] == dimnames(traj)[2:end] "k-space data dimension names must match trajectory sample dimension names"
                if !isnothing(smaps)
                    @argcheck :coil ∈ dimnames(ksp) "k-space must have :coil dimension when sensitivity maps are provided"
                end
            end
        end

        if !isnothing(dcf)
            @argcheck size(dcf) == size(traj)[2:end] "dcf shape must match trajectory sample dimensions"
            @argcheck eltype(dcf) <: Real "dcf must be real-valued"
        end

        if !isnothing(smaps) && !isnothing(ksp)
            @argcheck eltype(ksp) == eltype(smaps) "k-space and sensitivity maps eltype mismatch"
            if is3D
                @argcheck ndims(smaps) == 4 "sensitivity maps must be 4D for 3D acquisition"
                @argcheck size(smaps)[1:3] == img_size "sensitivity maps spatial size must match image_size"
            elseif ndims(smaps) == 4
                @argcheck size(smaps)[1:2] == img_size "sensitivity maps spatial size must match image_size"
            else
                @argcheck ndims(smaps) == 3 "sensitivity maps must be 3D for 2D acquisition"
                @argcheck size(smaps)[1:2] == img_size "sensitivity maps spatial size must match image_size"
            end
        end

        return new{typeof(ksp),typeof(traj),typeof(dcf),typeof(smaps),typeof(img_size),typeof(sK),typeof(sI)}(
            ksp, traj, dcf, smaps, img_size, sK, sI, is3D
        )
    end
end

NonCartesianAcquisitionInfo(
    kspace_data;
    trajectory,
    dcf=nothing,
    sensitivity_maps=nothing,
    image_size,
    shifted_kspace_dims::Union{Tuple,Integer,Symbol}=(),
    shifted_image_dims::Union{Tuple,Integer,Symbol}=(),
) = NonCartesianAcquisitionInfo(kspace_data, trajectory, dcf, sensitivity_maps, image_size, shifted_kspace_dims, shifted_image_dims)

function NonCartesianAcquisitionInfo(config::NonCartesianAcquisitionInfo; kwargs...)
    new_kwargs = Dict{Symbol,Any}()
    for fn in fieldnames(NonCartesianAcquisitionInfo)
        if fn == :is3D
            continue
        elseif haskey(kwargs, fn)
            new_kwargs[fn] = kwargs[fn]
        else
            new_kwargs[fn] = getfield(config, fn)
        end
    end
    return NonCartesianAcquisitionInfo(
        new_kwargs[:kspace_data],
        new_kwargs[:trajectory],
        new_kwargs[:dcf],
        new_kwargs[:sensitivity_maps],
        new_kwargs[:image_size],
        new_kwargs[:shifted_kspace_dims],
        new_kwargs[:shifted_image_dims],
    )
end

AcquisitionInfo(config::NonCartesianAcquisitionInfo; kwargs...) =
    NonCartesianAcquisitionInfo(config; kwargs...)

function Base.show(io::IO, info::NonCartesianAcquisitionInfo)
    meta = String[]
    if !isnothing(info.kspace_data)
        push!(meta, "kspace_data=Array{$(eltype(info.kspace_data))}<$(join(size(info.kspace_data), "×"))>")
    end
    push!(meta, "trajectory=Array{$(eltype(info.trajectory))}<$(join(size(info.trajectory), "×"))>")
    if !isnothing(info.dcf)
        push!(meta, "dcf=Array{$(eltype(info.dcf))}<$(join(size(info.dcf), "×"))>")
    end
    push!(meta, "encoding=" * (info.is3D ? "3D" : "2D"))
    push!(meta, "image_size=$(join(info.image_size, "×"))")
    if !isnothing(info.sensitivity_maps)
        push!(meta, "sensitivity_maps=$(eltype(info.sensitivity_maps))<$(join(size(info.sensitivity_maps), "×"))>")
    end
    print(io, "NonCartesianAcquisitionInfo(", join(meta, ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", info::NonCartesianAcquisitionInfo)
    meta = String[]
    if !isnothing(info.kspace_data)
        push!(meta, "kspace_data=Array{$(eltype(info.kspace_data))}<$(join(size(info.kspace_data), "×"))>")
    end
    push!(meta, "trajectory=Array{$(eltype(info.trajectory))}<$(join(size(info.trajectory), "×"))>")
    if !isnothing(info.dcf)
        push!(meta, "dcf=Array{$(eltype(info.dcf))}<$(join(size(info.dcf), "×"))>")
    end
    push!(meta, "encoding=" * (info.is3D ? "3D" : "2D"))
    push!(meta, "image_size=$(join(info.image_size, "×"))")
    if !isnothing(info.sensitivity_maps)
        push!(meta, "sensitivity_maps=$(eltype(info.sensitivity_maps))<$(join(size(info.sensitivity_maps), "×"))>")
    end
    println(io, "NonCartesianAcquisitionInfo:")
    for (i, m) in enumerate(meta)
        m = replace(m, "=" => " = ", "_" => " ")
        print(io, "  - $m")
        if i < length(meta)
            println(io)
        end
    end
    return nothing
end

function get_encoding_operator(info::NonCartesianAcquisitionInfo; threaded::Bool=true, fast_planning::Bool=false)
    error("Non-Cartesian encoding operator not implemented yet")

function get_subsampling_operator(::NonCartesianAcquisitionInfo)
    error("Subsampling operator is not applicable to non-Cartesian trajectories")
end