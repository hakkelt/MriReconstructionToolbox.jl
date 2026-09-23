# Back-compat driver: run every comparison section as its own subprocess (each section
# `include`s `_setup.jl` and defines `const`s, so they must not share a process). Each section
# records its own rows to `results/runs/` (see `ResultsStore.jl`) as it finishes -- there is no
# merge step: every run is an immutable file, and `query_results.jl` reads the whole directory at
# query time.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_all.jl --threads=N [--use-mkl]
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_all.jl --threads=N --sections=sparsity,dynamic
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_all.jl --threads=N --cases="Sparsity|Total Variation (20 it)"
# --sections restricts which of ALL_SECTIONS to run; --cases restricts further, to specific
# (category, method) pairs within whichever sections run (see `_setup.jl`'s `should_run`). Both
# exist to re-verify a suspect result without paying for the full matrix.
# For real parallelism submit the sections as separate SLURM jobs instead (see the SLURM array
# script) -- concurrent writers are safe by construction, so no merge coordination is needed either
# way.

const ALL_SECTIONS = ("base", "noncart", "cgsense", "sparsity", "dynamic", "kspace", "real", "accuracy_race")
let i = findfirst(a -> startswith(a, "--sections="), ARGS)
    global const SECTIONS = i === nothing ? ALL_SECTIONS : Tuple(split(split(ARGS[i], "=")[2], ","))
end
issubset(SECTIONS, ALL_SECTIONS) || error("unknown section in $SECTIONS; known: $ALL_SECTIONS")
pass = filter(a -> a != "run_all.jl" && !startswith(a, "--sections="), ARGS)
proj = Base.active_project()
jl = Base.julia_cmd()

for s in SECTIONS
    script = joinpath(@__DIR__, "run_$s.jl")
    @info "=== section $s ==="
    cmd = `$jl --project=$proj -t $(Threads.nthreads()) $script $pass`
    try
        run(cmd)
    catch e
        @warn "section $s failed (exit $(e))"
    end
end
