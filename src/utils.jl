function get_full_kspace(acq_info::CartesianAcquisitionInfo)
    @argcheck !isnothing(acq_info.kspace_data) "AcquisitionInfo must include k-space data"
    𝒫 = get_subsampling_operator(acq_info)
    return 𝒫' * acq_info.kspace_data
end

ensure_tuple(x::Tuple) = x
ensure_tuple(x) = (x,)
