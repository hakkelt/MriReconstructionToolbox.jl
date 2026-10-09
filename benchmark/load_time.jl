# Load time and time to first solve of Ristretto, each measured in a fresh process:
#
#   julia --project=benchmark --threads=N benchmark/load_time.jl [reps]
#
# For each of `reps` rounds and each workload, a child process (same project, same thread count)
# times `using Ristretto`, the first solve of the workload (compilation included) and a second,
# warm solve. One more child prints the 15 slowest `@time_imports` entries. Precompilation is
# done before the first round, so no round pays for it.
#
# Run it through benchmark/slurm/load_time.sh, which sweeps the thread counts.

using Printf, Statistics

const WORKLOADS = ("cartesian_wavelet", "radial_tv", "cine_lowrank")

if length(ARGS) >= 1 && ARGS[1] == "child"
    const t_using = @elapsed @eval using Ristretto
    using Random
    Random.seed!(0)

    function problem(name)
        n, nc = 64, 4
        img = ComplexF32.([hypot(i - n / 2, j - n / 3) < n / 4 for i in 1:n, j in 1:n])
        smaps = coil_sensitivities(n, n, nc)
        if name == "cartesian_wavelet"
            pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1), (n, n))
            acq = AcquisitionInfo(is3D = false, image_size = (n, n), subsampling = pattern, sensitivity_maps = smaps)
            return simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true), IterativeReconstruction(L1Wavelet2D(0.005f0); algorithm = FISTA(), maxit = 20)
        elseif name == "radial_tv"
            traj = Float32.(radial_trajectory(2n, 32))
            acq = AcquisitionInfo(; trajectory = traj, image_size = (n, n), sensitivity_maps = smaps)
            return simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true), IterativeReconstruction(TotalVariation2D(0.001f0); maxit = 20)
        elseif name == "cine_lowrank"
            nt = 8
            cine = NamedDimsArray{(:x, :y, :time)}(stack(circshift(img, (0, t)) for t in 1:nt))
            pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1), (n, n))
            acq = AcquisitionInfo(
                is3D = false, image_size = (n, n), subsampling = pattern,
                sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(smaps),
            )
            return simulate_acquisition(cine, acq; inverse_crime_check = false, keep_sensitivity_maps = true), IterativeReconstruction(LowRank(0.01f0; time_dim = :time); maxit = 20)
        end
        error("unknown workload $name")
    end

    data, method = problem(ARGS[2])
    t_first = @elapsed reconstruct(data, method; verbosity = Silent())
    t_second = @elapsed reconstruct(data, method; verbosity = Silent())
    distributed = any(m -> nameof(m) === :Distributed, values(Base.loaded_modules))
    println("RESULT ", t_using, " ", t_first, " ", t_second, " ", distributed)
    exit(0)
end

if length(ARGS) >= 1 && ARGS[1] == "imports"
    InteractiveUtils = Base.require(Base.PkgId(Base.UUID("b77e0a4c-d291-57a0-90e8-8db25a27a240"), "InteractiveUtils"))
    @eval InteractiveUtils.@time_imports using Ristretto
    exit(0)
end

const REPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const JULIA = joinpath(Sys.BINDIR, Base.julia_exename())
const PROJECT = Base.active_project()
const FLAGS = `--startup-file=no --project=$PROJECT --threads=$(Threads.nthreads())`

run(`$JULIA $FLAGS -e 'using Ristretto'`)

println("threads = ", Threads.nthreads(), ", julia ", VERSION, ", ", gethostname())
println()
println("slowest imports:")
lines = readlines(`$JULIA $FLAGS $(@__FILE__) imports`)
ms(l) = (m = match(r"^\s*([\d.]+) ms", l); m === nothing ? 0.0 : parse(Float64, m[1]))
foreach(println, first(sort(filter(l -> ms(l) > 0, lines); by = ms, rev = true), 15))
println()

results = Dict(w => NTuple{3, Float64}[] for w in WORKLOADS)
for _ in 1:REPS, w in WORKLOADS
    out = read(`$JULIA $FLAGS $(@__FILE__) child $w`, String)
    m = match(r"RESULT (\S+) (\S+) (\S+) (\S+)", out)
    m === nothing && error("child $w printed no result:\n$out")
    m[4] == "true" && @warn "Distributed is loaded in the $w child"
    push!(results[w], (parse(Float64, m[1]), parse(Float64, m[2]), parse(Float64, m[3])))
end

@printf("%-18s %16s %16s %16s\n", "workload", "using [s]", "first solve [s]", "warm solve [s]")
for w in WORKLOADS
    r = results[w]
    cell(k) = @sprintf("%6.2f ± %5.2f", median(getindex.(r, k)), std(getindex.(r, k)))
    @printf("%-18s %16s %16s %16s\n", w, cell(1), cell(2), cell(3))
end
println("(median ± std over $REPS fresh processes)")
