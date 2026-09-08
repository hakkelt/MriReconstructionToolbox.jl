using Documenter
using MriReconstructionToolbox
using ProximalAlgorithms
using ProximalOperators
using ContourletOperators

# The extension surface is `public` but not exported (see NAMING.md §6), so bring the names
# documented here into scope for the `@docs` and `@ref` blocks that reference them unqualified.
using MriReconstructionToolbox: Regularization, ReconstructionMethod, IterativeMethod,
    DirectMethod, Scaling, CoilCombination, DataFidelity, Verbosity,
    ReconstructionExecutor, Subsampling, VariableDensityDistribution, PartialFourierFilter,
    DensityCompensation, CoilCompression, SensitivityEstimation, GradientDelay,
    get_operator, materialize, materialize_with_auxiliaries, materialize_all, get_affected_dims,
    scale_regularization, bind_dimensions, calculate, check_applicable,
    get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator,
    build_encoding_operator, signal_model_operator, NamedDimsOp, DEFAULT_ALGORITHMS

# Internals whose docstrings are rendered on the low-level pages, or that other docstrings link to
# with `@ref`. Cross-references resolve in the page's module, so these have to be in scope too.
using MriReconstructionToolbox: with_serial_blas, serial_blas_threshold_bytes,
    set_serial_blas_threshold_bytes!, uses_blas3, maybe_disable_unsplit_threading,
    model_encoding_operator, StackedNSCTOp, BlockNuclearNorm, DenoiserProx

makedocs(;
    modules = [MriReconstructionToolbox, ProximalAlgorithms, ProximalOperators, ContourletOperators],
    authors = "Tamás Hakkel <hakkelt@gmail.com>",
    sitename = "MriReconstructionToolbox.jl",
    format = Documenter.HTML(
        assets = [asset("assets/favicon.svg", class = :ico, islocal = true)]
    ),
    pages = [
        "Home" => "index.md",
        "Theoretical Background" => "theory.md",
        "High-level Interface" => [
            "AcquisitionInfo" => "high-level/acquisition_info.md",
            "Preprocessing" => "high-level/preprocessing.md",
            "Simulation Tools" => "high-level/simulation.md",
            "Reconstruction Methods" => "high-level/methods.md",
            "Reconstruction" => "high-level/reconstruction.md",
            "Regularization" => "high-level/regularization.md",
            "Optimization Algorithms" => "high-level/algorithms.md",
            "Named Dimensions" => "high-level/nameddims.md",
            "Task Splitting" => "high-level/task_splitting.md",
            "Image Decomposition" => "high-level/image_decomposition.md",
            "Noise & Analysis" => "high-level/analysis.md",
            "Performance & Threading" => "high-level/performance.md",
        ],
        "Low-Level Interface" => [
            "MRI Operators" => "low-level/operators.md",
            "Custom Reconstruction" => "low-level/custom_reconstruction.md",
            "AbstractOperators.jl" => "low-level/abstract_operators.md",
            "ProximalOperators.jl" => "low-level/proximal_operators.md",
        ],
    ],
    checkdocs = :none,
    doctest = false
)

deploydocs(
    repo = "github.com/hakkelt/MriReconstructionToolbox.jl.git"
)
