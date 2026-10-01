module NFFTGPUArraysExt

using ..NFFT, ..NFFT.AbstractNFFTs
using ..NFFT.SparseArrays, ..NFFT.LinearAlgebra, ..NFFT.FFTW
using GPUArrays, Adapt
using GPUArrays.KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, get_backend

include("implementation.jl")

end
