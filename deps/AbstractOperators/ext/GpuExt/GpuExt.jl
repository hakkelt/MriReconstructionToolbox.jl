module GpuExt

using GPUArrays
using KernelAbstractions
using AbstractOperators
import LinearAlgebra: mul!
import AbstractOperators:
    _should_thread, array_type_display_string, check,
    AdjointOperator, Variation, NoOperatorBroadCast,
    domain_type, allocate_in_domain, allocate_in_codomain,
    GetIndex, ZeroPad, OperatorWrapper
using RecursiveArrayTools: ArrayPartition

include("properties.jl")
include("linearoperators/getindex.jl")
include("linearoperators/zeropad.jl")
include("linearoperators/variation.jl")
include("cpuwrapper.jl")
include("guards.jl")

end # module GpuExt
