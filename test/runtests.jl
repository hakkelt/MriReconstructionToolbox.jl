using Test
using MriReconstructionToolbox
using AbstractOperators
using FFTW
using NamedDims
using LinearAlgebra
using Wavelets

include("test_encoding_op.jl")
include("test_regularizations.jl")
include("test_minimizer.jl")
include("test_reconstruction_integration.jl")
