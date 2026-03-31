# Copy constructors for AcquisitionInfo subtypes — included after both subtypes are defined.
AcquisitionInfo(config::CartesianAcquisitionInfo; kwargs...) =
    CartesianAcquisitionInfo(config; kwargs...)
AcquisitionInfo(config::NonCartesianAcquisitionInfo; kwargs...) =
    NonCartesianAcquisitionInfo(config; kwargs...)
