"""
	AbstractReconstructionMethod

Abstract root of the reconstruction method taxonomy.
"""
abstract type AbstractReconstructionMethod end

"""
	AbstractIterativeMethod <: AbstractReconstructionMethod

Abstract supertype for iterative reconstruction methods.
"""
abstract type AbstractIterativeMethod <: AbstractReconstructionMethod end

"""
	AbstractDirectMethod <: AbstractReconstructionMethod

Abstract supertype for direct (non-iterative) reconstruction methods.
"""
abstract type AbstractDirectMethod <: AbstractReconstructionMethod end

"""
	DEFAULT_ALGORITHMS

Default solver tuple dispatched when no specific algorithm is provided.
"""
const DEFAULT_ALGORITHMS = (CG(), CGNR(), FISTA(), ADMM(), DouglasRachford())

"""
	lower(method::AbstractReconstructionMethod)

Lowers high-level method specifications into canonical reconstruction methods.
"""
lower(m::AbstractReconstructionMethod) = m

"""
	check_applicable(method::AbstractReconstructionMethod, acq::AcquisitionInfo)

Validates that the reconstruction method is applicable to the given acquisition data.
Throws an error or returns `nothing`.
"""
check_applicable(::AbstractReconstructionMethod, ::AcquisitionInfo) = nothing

"""
	variable_dims(method::AbstractReconstructionMethod, acq::AcquisitionInfo)

Returns the dimension names/order of the reconstruction optimization variable.
"""
variable_dims(::AbstractReconstructionMethod, acq::AcquisitionInfo) = get_image_dims(acq)

"""
	variable_size(method::AbstractReconstructionMethod, acq::AcquisitionInfo)

Returns the expected size of the reconstruction optimization variable.
"""
variable_size(::AbstractReconstructionMethod, acq::AcquisitionInfo) = get_image_size(acq)

"""
	output_dims(method::AbstractReconstructionMethod, acq::AcquisitionInfo)

Returns the dimension names of the output reconstructed image.
"""
output_dims(::AbstractReconstructionMethod, acq::AcquisitionInfo) = get_image_dims(acq)
