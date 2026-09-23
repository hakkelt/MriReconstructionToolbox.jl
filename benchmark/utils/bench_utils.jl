"""
    BenchUtils

Everything the MRT harness (`benchmark/run.jl`) and the comparison suite (`benchmark/comparison/`)
share: site configuration, the case catalog with its phantoms, sampling patterns, noise and real
data, MRT's reconstruction of each method, the timing function and the result store.

Depends only on MRT, GeometricMedicalPhantoms, FFTW, NamedDims, MRITestData, JSON and standard
libraries. It never assumes BART, SigPy or MRIReco are available; those live in the comparison
suite's own bridges.

    include(joinpath(<repo>, "benchmark", "utils", "bench_utils.jl"))
    using .BenchUtils
"""
module BenchUtils

using Dates: Dates, now, @dateformat_str
using FFTW: fft, ifft, fftshift, ifftshift
using GeometricMedicalPhantoms: create_shepp_logan_phantom, create_torso_phantom, MRISheppLoganIntensities,
    generate_cardiac_signals, generate_respiratory_signal
using JSON: JSON
using LinearAlgebra: norm
using NamedDims: NamedDimsArray, unname
using Printf: @sprintf
using Random: AbstractRNG, MersenneTwister, randperm
using SHA: sha1
using Serialization: serialize, deserialize
using Statistics: median

using MriReconstructionToolbox
using MriReconstructionToolbox: CartesianAcquisitionInfo, NonCartesianAcquisitionInfo

include("config.jl")
include("phantoms.jl")
include("sampling.jl")
include("noise.jl")
include("cases.jl")
include("real_data.jl")
include("mrt_methods.jl")
include("harness.jl")
include("results_store.jl")

export load_site_env!, env_flag, ensure_download_path!
export BenchCase, get_case, case_ids, filter_case_ids, SYNTHETIC_CASES, REAL_CASES, small_mode, cine_frames
export ncoils, acceleration, zero_filled, mrt_acquisition, applicable_methods, METHODS
export norm_ksp, add_noise, nrmse, mag_nrmse, centred_fft, centred_ifft
export mrt_reconstructor, mrt_regularizer, mrt_algorithm, DEFAULT_LAMBDA, OUTER_ITERATIONS, CG_ITERATIONS, ADMM_RHO
export time_run, timed_runs, git_ref, tree_hash, node_class, recorded_env, ResultsStore

end
