"""
Abstract type for MRI acquisition information.

Subtypes:
- `CartesianAcquisitionInfo`: Cartesian (regular grid) acquisitions
- `NonCartesianAcquisitionInfo`: Non-Cartesian (trajectory-based) acquisitions
"""
abstract type AcquisitionInfo end

"""
    AcquisitionInfo(
            kspace_data=nothing;
            trajectory=nothing,
            is3D::Union{Bool,Nothing}=nothing,
            sensitivity_maps=nothing,
            image_size=nothing,
            subsampling=nothing,
            dcf=nothing,
            shifted_kspace_dims::Union{Tuple,Integer,Symbol}=(),
            shifted_image_dims::Union{Tuple,Integer,Symbol}=(),
    )

Smart constructor that dispatches to either `CartesianAcquisitionInfo` or
`NonCartesianAcquisitionInfo`.

- If `trajectory` is `nothing`, constructs `CartesianAcquisitionInfo`
- If `trajectory` is provided, constructs `NonCartesianAcquisitionInfo`

For non-Cartesian acquisitions, `dcf` may be provided as an optional density
compensation array and `subsampling` is not allowed.

`kspace_data` is normally an `AbstractArray` (plain or `NamedDimsArray`). For a Cartesian
acquisition whose frames select *different numbers of samples* it is a [`PartitionedKSpace`](@ref)
instead, one array per frame — a dense array cannot hold a ragged sample axis.
"""
function AcquisitionInfo(
        kspace_data = nothing;
        trajectory = nothing,
        is3D::Union{Bool, Nothing} = nothing,
        sensitivity_maps = nothing,
        image_size = nothing,
        subsampling = nothing,
        dcf = nothing,
        shifted_kspace_dims::Union{Tuple, Integer, Symbol} = (),
        shifted_image_dims::Union{Tuple, Integer, Symbol} = (),
    )
    if isnothing(trajectory)
        @argcheck isnothing(dcf) "dcf can only be used with trajectory-based acquisitions"
        return CartesianAcquisitionInfo(
            kspace_data;
            is3D,
            sensitivity_maps,
            image_size,
            subsampling,
            shifted_kspace_dims,
            shifted_image_dims,
        )
    end
    @argcheck isnothing(subsampling) "subsampling cannot be used with trajectory-based acquisitions"
    return NonCartesianAcquisitionInfo(
        kspace_data;
        trajectory,
        dcf,
        sensitivity_maps,
        image_size,
        shifted_kspace_dims,
        shifted_image_dims,
    )
end
