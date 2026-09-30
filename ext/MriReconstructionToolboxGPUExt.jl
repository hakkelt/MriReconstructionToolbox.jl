module MriReconstructionToolboxGPUExt

# The operator and proximal packages are submodules of MriReconstructionToolbox, and a submodule
# cannot load package extensions, so their GPU extensions are included here instead. Each goes
# into a module of its own that binds the package it extends, because the extensions import it
# relatively (`..AbstractOperators`) and two of them are called `GpuExt`.

module AbstractOperatorsGPU
using MriReconstructionToolbox: AbstractOperators
include(joinpath(@__DIR__, "..", "deps", "AbstractOperators", "ext", "GpuExt", "GpuExt.jl"))
end

module FFTWOperatorsGPU
using MriReconstructionToolbox: AbstractOperators, FFTWOperators
include(
    joinpath(@__DIR__, "..", "deps", "AbstractOperators", "FFTWOperators", "ext", "GpuExt", "GpuExt.jl")
)
end

module NFFTOperatorsGPU
using MriReconstructionToolbox: NFFTOperators
include(
    joinpath(
        @__DIR__, "..", "deps", "AbstractOperators", "NFFTOperators", "ext", "NFFTOperatorsGPUArraysExt.jl"
    )
)
end

module ProximalOperatorsGPU
using MriReconstructionToolbox: ProximalOperators
include(joinpath(@__DIR__, "..", "deps", "ProximalOperators", "ext", "GpuExt", "GpuExt.jl"))
end

end # module MriReconstructionToolboxGPUExt
