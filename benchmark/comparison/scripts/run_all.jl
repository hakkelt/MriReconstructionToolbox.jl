# Back-compat driver: run every comparison section as its own subprocess (each section
# `include`s `_setup.jl` and defines `const`s, so they must not share a process), then merge.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_all.jl --threads=N [--use-mkl]
# For real parallelism submit the sections as separate SLURM jobs instead (see the SLURM array
# script), then run merge_benchmarks.jl.

const SECTIONS = ("base", "noncart", "cgsense", "sparsity", "dynamic", "kspace", "real", "accuracy_race")
pass = filter(a -> a != "run_all.jl", ARGS)
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

run(`$jl --project=$proj $(joinpath(@__DIR__, "merge_benchmarks.jl")) $pass`)
