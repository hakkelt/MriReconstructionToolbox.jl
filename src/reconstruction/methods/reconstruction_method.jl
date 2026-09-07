"""
	ReconstructionMethod

Abstract root of the reconstruction method taxonomy.
"""
abstract type ReconstructionMethod end

"""
	IterativeMethod <: ReconstructionMethod

Abstract supertype for iterative reconstruction methods.
"""
abstract type IterativeMethod <: ReconstructionMethod end

"""
	DirectMethod <: ReconstructionMethod

Abstract supertype for direct (non-iterative) reconstruction methods.
"""
abstract type DirectMethod <: ReconstructionMethod end

"""
	DEFAULT_ALGORITHMS

Default solver tuple dispatched when no specific algorithm is provided.
"""
const DEFAULT_ALGORITHMS = (CG(), CGNR(), FISTA(), ADMM(), DouglasRachford())

"""
	lower(method::ReconstructionMethod)
	lower(method::ReconstructionMethod, acq::AcquisitionInfo)

Lowers high-level method specifications into canonical reconstruction methods.
"""
lower(m::ReconstructionMethod) = m
lower(m::ReconstructionMethod, ::AcquisitionInfo) = lower(m)

"""
	check_applicable(method::ReconstructionMethod, acq::AcquisitionInfo)

Validates that the reconstruction method is applicable to the given acquisition data.
Throws an error or returns `nothing`.
"""
check_applicable(::ReconstructionMethod, ::AcquisitionInfo) = nothing

"""
	variable_dims(method::ReconstructionMethod, acq::AcquisitionInfo)

Returns the dimension names/order of the reconstruction optimization variable.
"""
variable_dims(::ReconstructionMethod, acq::AcquisitionInfo) = get_image_dims(acq)

"""
	variable_size(method::ReconstructionMethod, acq::AcquisitionInfo)

Returns the expected size of the reconstruction optimization variable.
"""
variable_size(::ReconstructionMethod, acq::AcquisitionInfo) = get_image_size(acq)

"""
	output_dims(method::ReconstructionMethod, acq::AcquisitionInfo)

Returns the dimension names of the output reconstructed image.
"""
output_dims(::ReconstructionMethod, acq::AcquisitionInfo) = get_image_dims(acq)
