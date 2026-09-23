# Back-compat driver: run every comparison section as its own subprocess (each section
# `include`s `_setup.jl` and defines `const`s, so they must not share a process), then merge.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_all.jl --threads=N [--use-mkl]
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_all.jl --threads=N --sections=sparsity,dynamic
# --sections restricts which of ALL_SECTIONS to run (e.g. to re-verify a suspect result without
# paying for the full matrix); default is all of them. merge_benchmarks.jl still merges every
# fragment found on disk, so a partial run's merged JSON keeps whatever was already there for the
# sections skipped this time.
# For real parallelism submit the sections as separate SLURM jobs instead (see the SLURM array
# script), then run merge_benchmarks.jl.

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

run(`$jl --project=$proj $(joinpath(@__DIR__, "merge_benchmarks.jl"))`)
