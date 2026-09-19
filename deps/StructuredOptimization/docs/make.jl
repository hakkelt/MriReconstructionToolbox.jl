using Documenter, StructuredOptimization,
    LinearAlgebra, DSP, FFTW, AbstractOperators, ProximalAlgorithms
using Literate
using Random

# Every doctest runs with these bindings in scope and a fixed seed, so a docstring example
# does not have to repeat the `using` lines and random data is reproducible.
DocMeta.setdocmeta!(
    StructuredOptimization,
    :DocTestSetup,
    :(
        using StructuredOptimization, ProximalAlgorithms, ProximalOperators,
            AbstractOperators, LinearAlgebra, Random;
        Random.seed!(0)
    );
    recursive = true,
)

# Literate sources live outside `src/` so the generated Markdown is never mistaken for
# hand-written documentation. Each is executed during the build, so the examples are tests:
# a page that stops working fails CI.
const EXAMPLES_IN = joinpath(@__DIR__, "examples")
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")

const EXAMPLE_PAGES = [
    "Lasso and warm starting" => "lasso.jl",
    "Total variation denoising" => "tv_denoising.jl",
    "Audio declipping" => "audio_declipping.jl",
    "A multi-variable problem" => "multivariable.jl",
    "A non-convex problem" => "rosenbrock.jl",
    "FFT deconvolution" => "fft_deconvolution.jl",
    "When parsing fails" => "when_parsing_fails.jl",
]

isdir(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive = true)
for (_, file) in EXAMPLE_PAGES
    Literate.markdown(joinpath(EXAMPLES_IN, file), EXAMPLES_OUT; documenter = true)
end

example_pages = [title => joinpath("examples", replace(file, ".jl" => ".md")) for (title, file) in EXAMPLE_PAGES]

makedocs(
    # Only this package's symbols are coverage-checked; ProximalAlgorithms docstrings are
    # still rendered where referenced, but we don't require documenting its entire internal
    # API here. `:all` (rather than `:exports`) means a docstring that is not referenced from
    # any page fails the build — including the internal ones, which is why `internals.md`
    # exists.
    modules = [StructuredOptimization],
    checkdocs = :all,
    format = Documenter.HTML(),
    # Phase 0.3: run every docstring/doc code block as a doctest in CI.
    doctest = true,
    # Float output differs in the last digits between machines and BLAS versions, and object
    # printing carries type parameters that are not the point of any example. Filter both so
    # the doctests test behaviour rather than formatting.
    doctestfilters = [
        r"[0-9]+\.[0-9]{6,}e?-?[0-9]*",     # long floats
        r"\{[A-Za-z0-9_, \.\{\}\<\:]+\}",   # type parameters in printed types
        r"@ StructuredOptimization .*",     # method locations
    ],
    sitename = "StructuredOptimization",
    authors = "Niccolò Antonello and Lorenzo Stella",
    pages = [
        "Home" => "index.md",
        "Quick Tutorial Guide" => "tutorial.md",
        "Theory" => [
            "Problem form & algorithms" => "theory/problem_form.md",
            "How problems are parsed" => "theory/parsing.md",
            "Matrix-free operators" => "theory/matrix_free.md",
        ],
        "Expressions" => "expressions.md",
        "Functions" => "functions.md",
        "Solvers" => "solvers.md",
        "Examples" => example_pages,
        "FAQ / Troubleshooting" => "faq.md",
        "Internals" => "internals.md",
        "Demos" => "demos.md",
    ],
)

deploydocs(
    repo = "github.com/hakkelt/StructuredOptimization.jl.git",
    target = "build",
)
