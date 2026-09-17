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

`POGM` rather than `FISTA` carries the proximal-gradient slot: it has the same problem shape and
per-iteration cost, and a better worst-case convergence rate, so it is the better default wherever
either would be picked. `FISTA` remains available as an explicit `algorithm`.

The tuple holds one entry per *problem shape*, not one per algorithm: a candidate whose
`get_assumptions` duplicates an earlier entry's could never be reached, so `FISTA` and `ISTA` —
which declare exactly what `POGM` declares — are reached through `algorithm = FISTA()` /
`algorithm = ISTA()` instead of sitting here as unreachable fallbacks.
"""
const DEFAULT_ALGORITHMS = (CG(), CGNR(), POGM(), ADMM(), DouglasRachford())

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
