# Where "serial BLAS wins" stops being true: sweep work-item size and batch width for a
# BLAS-1 (CG-shaped) loop and, separately, a BLAS-3 (SVD-shaped) loop of the kind the
# locally-low-rank / multi-scale-low-rank prox steps run. This is the evidence behind
# SERIAL_BLAS_THRESHOLD_BYTES and maybe_disable_undecomposed_threading.
#
# usage: julia --project=benchmark/hpc -t N benchmark/hpc/scripts/threading_sweep.jl <openblas|mkl> <blas_threads> <cg|svd>
#   cg  : env N, NCOIL, NSLICE, NITER
#   svd : env SVD_M, SVD_N, SVD_BLOCKS, NITER

const BACKEND = ARGS[1]
const BLAS_THREADS = parse(Int, ARGS[2])
const KIND = ARGS[3]

BACKEND == "mkl" && (@eval using MKL)
using LinearAlgebra, FFTW, Printf

BLAS.set_num_threads(BLAS_THREADS)
FFTW.set_num_threads(1)   # parallelism comes from the Julia-level batch loop

envi(k, d) = parse(Int, get(ENV, k, string(d)))

cpu_seconds() = begin
    f = split(read("/proc/self/stat", String))
    (parse(Float64, f[14]) + parse(Float64, f[15])) / 100
end

const N = envi("N", 128)
const NCOIL = envi("NCOIL", 8)
const NSLICE = envi("NSLICE", 32)
const NITER = envi("NITER", 300)

struct Slab{P, IP}
    x::Array{ComplexF32, 3}
    r::Array{ComplexF32, 3}
    p::Array{ComplexF32, 3}
    Ap::Array{ComplexF32, 3}
    plan::P
    iplan::IP
end

function Slab()
    x = randn(ComplexF32, N, N, NCOIL)
    plan = plan_fft!(similar(x), (1, 2); flags = FFTW.ESTIMATE)
    iplan = plan_bfft!(similar(x), (1, 2); flags = FFTW.ESTIMATE)
    return Slab(x, similar(x), copy(x), similar(x), plan, iplan)
end

function cg_like!(s::Slab)
    acc = 0.0f0
    for _ in 1:NITER
        copyto!(s.Ap, s.p)
        s.plan * s.Ap
        s.Ap .*= 0.5f0
        s.iplan * s.Ap
        pAp = real(dot(vec(s.p), vec(s.Ap)))
        α = 1.0f-3 / (abs(pAp) + 1.0f-6)
        axpy!(α, vec(s.p), vec(s.x))
        axpy!(-α, vec(s.Ap), vec(s.r))
        acc += norm(s.r)
        @. s.p = s.r + 1.0f-3 * s.p
    end
    return acc
end

const SVD_M = envi("SVD_M", 256)
const SVD_N = envi("SVD_N", 64)
const SVD_BLOCKS = envi("SVD_BLOCKS", 32)

struct Block
    A::Matrix{ComplexF32}
end
Block() = Block(randn(ComplexF32, SVD_M, SVD_N))

function svd_like!(b::Block)
    acc = 0.0f0
    for _ in 1:NITER
        F = svd(b.A)
        acc += sum(F.S)
        mul!(b.A, F.U, Diagonal(F.S) * F.Vt)
    end
    return acc
end

if KIND == "cg"
    items = [Slab() for _ in 1:NSLICE]
    work!(v) = Threads.@threads for s in v
        cg_like!(s)
    end
    label = @sprintf("cg  %d^2x%-3d x%-3d slabs", N, NCOIL, NSLICE)
    bytes = N * N * NCOIL * 8
else
    items = [Block() for _ in 1:SVD_BLOCKS]
    work!(v) = Threads.@threads for b in v
        svd_like!(b)
    end
    label = @sprintf("svd %dx%-4d x%-3d blocks", SVD_M, SVD_N, SVD_BLOCKS)
    bytes = SVD_M * SVD_N * 8
end

work!(items)   # warmup
best_wall, best_cpu = Inf, Inf
for _ in 1:3
    c0 = cpu_seconds()
    w = @elapsed work!(items)
    global best_wall = min(best_wall, w)
    global best_cpu = min(best_cpu, cpu_seconds() - c0)
end

@printf(
    "%-26s %-9s jl=%-2d blas=%-2d  %6.2f MiB/item  wall=%8.3f s  cpu/wall=%5.2f\n",
    label, BACKEND, Threads.nthreads(), BLAS.get_num_threads(),
    bytes / 2^20, best_wall, best_cpu / best_wall
)
