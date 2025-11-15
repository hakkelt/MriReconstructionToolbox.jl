using Documenter
using MriReconstructionToolbox

makedocs(;
    modules = [MriReconstructionToolbox],
    authors = "Tamás Hakkelt <tamas.hakkelt@gmail.com>",
    sitename = "MriReconstructionToolbox.jl",
    remotes = nothing,  # Disable remote source links for local builds
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Encoding Operator" => "encoding.md",
    ],
    doctest = false,
    clean = true,
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/hakkelt/MriReconstructionToolbox.jl.git",
    push_preview = true
)
