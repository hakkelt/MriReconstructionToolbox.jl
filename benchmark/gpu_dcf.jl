# Pipe-Menon density compensation of a host acquisition against the same acquisition on a CUDA
# device, for a 2D radial slice and a radial cine with one trajectory per frame. The two variants
# of a case are timed round-robin so that drift on a shared node hits them alike.
#
#   julia --project=test benchmark/gpu_dcf.jl [reps]
#
# Needs a CUDA device; run it through benchmark/slurm/gpu_refs.sh on an A100.

using GPUEnv
GPUEnv.activate(; include_jlarrays = false)
using CUDA, Printf, Statistics
using Ristretto
using Ristretto: NonCartesianAcquisitionInfo
using NamedDims: unname

CUDA.functional() || error("no functional CUDA device")
const REPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10

function acquisition(nsample, nspoke, ncoil, nframe; n = 128)
    traj = Float32.(unname(radial_trajectory(nsample, nspoke * nframe; ordering = GoldenAngle())))
    if nframe == 1
        traj = reshape(traj, 2, nsample, nspoke)
        ksp = randn(ComplexF32, nsample, nspoke, ncoil)
    else
        traj = reshape(traj, 2, nsample, nspoke, nframe)
        ksp = randn(ComplexF32, nsample, nspoke, ncoil, nframe)
    end
    return NonCartesianAcquisitionInfo(ksp; trajectory = traj, image_size = (n, n))
end

function timed(f)
    t = @elapsed (f(); CUDA.synchronize())
    return t
end

const CASES = (
    ("radial 2D 256x64, 8 coils", acquisition(256, 64, 8, 1)),
    ("radial cine 256x34x24, 8 coils", acquisition(256, 34, 8, 24)),
)

@printf("%-34s %10s %10s %8s\n", "case", "host [ms]", "cuda [ms]", "cuda/host")
for (name, host) in CASES
    dev = Ristretto.Adapt.adapt(CuArray, host)
    variants = (() -> density_compensation(host), () -> density_compensation(dev))
    foreach(f -> f(), variants)  # compile
    times = [Float64[] for _ in variants]
    for _ in 1:REPS, (i, f) in enumerate(variants)
        push!(times[i], timed(f))
    end
    th, td = minimum.(times) .* 1.0e3
    @printf("%-34s %10.2f %10.2f %8.3f\n", name, th, td, td / th)
end
