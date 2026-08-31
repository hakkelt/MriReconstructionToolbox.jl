module MriReconstructionToolbox
using Reexport

using LinearAlgebra
using Random: Random, AbstractRNG
using Base.Cartesian
using ProximalOperators
using ProximalCore
using ProximalAlgorithms
@reexport using AbstractOperators
using AbstractOperators: Sum  # resolve ambiguity with ProximalOperators.Sum
@reexport using NamedDims
@reexport using StructuredOptimization
using NFFTOperators: NFFTOp

using NestedThreading: @budgeted_threads, with_full_threads, with_restricted_threads
@reexport using WaveletOperators: WaveletOp, WT, wavelet
@reexport using FFTWOperators: FFTWOperators, DFT, fftshift_op, ifftshift_op, alternate_sign!
using FFTW: FFTW
using ArgCheck: @argcheck
using Printf: @sprintf
using Statistics: quantile, median, mean
using Base.Threads: @threads, @spawn, nthreads
using StatsBase: sample, ProbabilityWeights

const ISTA = ProximalAlgorithms.ForwardBackward
const FISTA = ProximalAlgorithms.FastForwardBackward
const ADMM = ProximalAlgorithms.ADMM
const DouglasRachford = ProximalAlgorithms.DouglasRachford
const CG = ProximalAlgorithms.CG
const CGNR = ProximalAlgorithms.CGNR

export get_operator, get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator
export Tikhonov, L1Image, L1Wavelet2D, L1Wavelet3D, TotalVariation2D, TotalVariation3D, TemporalFourier, LowRank, RankLimit
export TemporalTotalVariation, JointSparsity, LocallyLowRank, ReferencePrior, NonNegative, BoxConstraint
export SecondOrderTotalVariation2D, SecondOrderTotalVariation3D, MultiScaleLowRank
export EdgePreservingRoughness2D, EdgePreservingRoughness3D, TotalGeneralizedVariation2D
export HardThreshold, SparsityLimit, PlugAndPlay
export calculate, build_model, reconstruct, Config, SequentialExecutor, MultiThreadingExecutor
export AbstractReconstructionMethod, AbstractIterativeMethod, AbstractDirectMethod
export DirectReconstruction, IterativeReconstruction, DEFAULT_ALGORITHMS
export ReconstructionDomain, ImageDomain, KSpaceDomain
export CoilCombination, AdjointSensitivity, RootSumSquares, NoCoilCombination
export DataFidelity, L2Loss, HardConsistency, NoFidelity, HardConsistencyProx
export lower, check_applicable, variable_dims, variable_size, output_dims
export Component, DecomposedImage, components, total
export BartScaling, FixedScaling, MeasurementBasedScaling, NoScaling
export ISTA, FISTA, ADMM, DouglasRachford, CG, CGNR
export AcquisitionInfo, CartesianAcquisitionInfo, NonCartesianAcquisitionInfo
export simulate_acquisition, coil_sensitivities
export UniformRandomSampling, VariableDensitySampling, PoissonDiskSampling, GaussianDistribution, PolynomialDistribution
export create_sampling_pattern, to_displayable_mask

include("acquisition_data/acquisition_info.jl")
include("acquisition_data/cartesian_acquisition_info.jl")
include("acquisition_data/noncartesian_acquisition_info.jl")
include("acquisition_data/acquisition_info_copy.jl")
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
include("regularization/temporal_total_variation_reg.jl")
include("regularization/second_order_total_variation_reg.jl")
include("regularization/edge_preserving_reg.jl")
include("regularization/total_generalized_variation_reg.jl")
include("regularization/low_rank_reg.jl")
include("regularization/locally_low_rank_reg.jl")
include("regularization/multi_scale_low_rank_reg.jl")
include("regularization/joint_sparsity_reg.jl")
include("regularization/hard_threshold_reg.jl")
include("regularization/constraint_reg.jl")
include("regularization/reference_prior_reg.jl")
include("regularization/plug_and_play_reg.jl")

include("reconstruction/components.jl")
include("reconstruction/methods/domains.jl")
include("reconstruction/methods/reconstruction_method.jl")
include("reconstruction/methods/direct_reconstruction.jl")
include("reconstruction/methods/iterative_reconstruction.jl")
include("reconstruction/decomposition.jl")
include("reconstruction/config.jl")
include("reconstruction/hard_consistency.jl")
include("reconstruction/build_model.jl")
include("reconstruction/progress_utils.jl")
include("reconstruction/initial_guess.jl")
include("reconstruction/direct.jl")
include("reconstruction/solve_core.jl")
include("reconstruction/reconstruct.jl")

include("simulation/subsampling.jl")
include("simulation/sensitivities.jl")
include("simulation/simulate_acquisition.jl")

end # module MriReconstructionToolbox
