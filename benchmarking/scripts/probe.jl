# What the process actually sees of its own core budget. Run under whatever launcher /
# allocation you are debugging (taskset, srun, bare).
#
# usage: julia --project=benchmarking -t N benchmarking/scripts/probe.jl

using LinearAlgebra, Printf

cpus_allowed = let line = ""
    for l in eachline("/proc/self/status")
        startswith(l, "Cpus_allowed_list:") && (line = strip(split(l, ":")[2]))
    end
    line
end
n_allowed = sum(
    r -> (p = split(r, "-"); length(p) == 1 ? 1 : parse(Int, p[2]) - parse(Int, p[1]) + 1),
    split(cpus_allowed, ","),
)

@printf("host                       = %s\n", gethostname())
@printf("Sys.CPU_THREADS            = %d   (whole node, NOT cgroup-aware)\n", Sys.CPU_THREADS)
@printf("Cpus_allowed_list          = %s  (%d cpus)\n", cpus_allowed, n_allowed)
@printf("Threads.nthreads()         = %d\n", Threads.nthreads())
@printf("Threads.threadpoolsize()   = %d\n", Threads.threadpoolsize())
@printf("BLAS.get_num_threads()     = %d   <- LinearAlgebra default (from jl_effective_threads)\n", BLAS.get_num_threads())
@printf("SLURM_CPUS_PER_TASK        = %s\n", get(ENV, "SLURM_CPUS_PER_TASK", "<unset>"))
@printf("OMP_NUM_THREADS            = %s\n", get(ENV, "OMP_NUM_THREADS", "<unset>"))
@printf("KMP_BLOCKTIME              = %s\n", get(ENV, "KMP_BLOCKTIME", "<unset>"))
flush(stdout)
