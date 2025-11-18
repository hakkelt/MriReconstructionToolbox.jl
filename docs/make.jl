using Documenter
using MriReconstructionToolbox
using ProximalAlgorithms

makedocs(;
    modules = [MriReconstructionToolbox, ProximalAlgorithms],
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
            "Simulation Tools" => "high-level/simulation.md",
            "Reconstruction" => "high-level/reconstruction.md",
            "Regularization" => "high-level/regularization.md",
            "Optimization Algorithms" => "high-level/algorithms.md",
            "Named Dimensions" => "high-level/nameddims.md",
            "Problem Decomposition" => "high-level/decomposition.md",
        ],
        "Low-Level Interface" => [
            "MRI Operators" => "low-level/operators.md",
            "Custom Reconstruction" => "low-level/custom_reconstruction.md",
            "AbstractOperators.jl" => "low-level/abstract_operators.md",
            "ProximalOperators.jl" => "low-level/proximal_operators.md",
        ]
    ],
    checkdocs = :none,
    doctest = false
)

deploydocs(
    repo = "github.com/hakkelt/MriReconstructionToolbox.jl.git"
)
