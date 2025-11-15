module MriReconstructionToolbox
using Reexport

using LinearAlgebra
using Base.Cartesian
using ProximalOperators
using ProximalAlgorithms
@reexport using AbstractOperators
@reexport using NamedDims
@reexport using StructuredOptimization

using AbstractOperators: @enable_full_threading, @restrict_threading
@reexport using WaveletOperators: WaveletOp, WT, wavelet
@reexport using FFTWOperators: FFTWOperators, DFT, fftshift_op, ifftshift_op, alternate_sign!
using FFTW: FFTW
using ArgCheck: @argcheck
using Printf: @sprintf
using Statistics: quantile, median, mean
using Base.Threads: @threads, @spawn, nthreads
using StatsBase: sample, ProbabilityWeights
using ImagePhantoms: ImagePhantoms
using ImageGeoms: ImageGeom

const ISTA = ProximalAlgorithms.ForwardBackward
const FISTA = ProximalAlgorithms.FastForwardBackward
const ADMM = ProximalAlgorithms.ADMM
const CG = ProximalAlgorithms.CG
const CGNR = ProximalAlgorithms.CGNR

export get_operator, get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator
export Tikhonov, L1Image, L1Wavelet2D, L1Wavelet3D, TotalVariation2D, TotalVariation3D, TemporalFourier, LowRank
export calculate, build_model, reconstruct, Config, SequentialExecutor, MultiThreadingExecutor
export ISTA, FISTA, ADMM, CG, CGNR
export AcquisitionInfo
export simulate_acquisition, shepp_logan, coil_sensitivities
export UniformRandomSampling, VariableDensitySampling, PoissonDiskSampling, GaussianDistribution, PolynomialDistribution
export create_sampling_pattern, to_displayable_mask

include("acquisition_data/acquisition_info.jl")
include("acquisition_data/dimension_utils.jl")

include("scaling.jl")
include("utils.jl")

include("encoding/named_dims_op.jl")
include("encoding/fourier_operators.jl")
include("encoding/sensitivity_map_operators.jl")
include("encoding/subsampling_operators.jl")
include("encoding/encoding_operators.jl")

include("regularization/regularization.jl")
include("regularization/image_domain_reg.jl")
include("regularization/wavelet_reg.jl")
include("regularization/total_variation_reg.jl")
include("regularization/temporal_fourier_reg.jl")
include("regularization/low_rank_reg.jl")

include("reconstruction/decomposition.jl")
include("reconstruction/config.jl")
include("reconstruction/build_model.jl")
include("reconstruction/progress_utils.jl")
include("reconstruction/reconstruct.jl")

include("simulation/subsampling.jl")
include("simulation/phantoms.jl")
include("simulation/sensitivities.jl")
include("simulation/simulate_acquisition.jl")

end # module MriReconstructionToolbox
