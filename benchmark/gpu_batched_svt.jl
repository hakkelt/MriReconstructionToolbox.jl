# Batched singular value thresholding on a GPU: the path `ProximalOperators.batched_svt!` takes
# (device Gram matrices, host eigendecompositions in Float64) against CUSOLVER's batched
# routines, for the slice shapes locally low-rank regularization produces (block voxels × frames).
# The variants of a case are timed round-robin so that drift on a shared node hits them alike.
#
#   julia --project=test benchmark/gpu_batched_svt.jl [reps]
#
# Needs a CUDA device; run it through benchmark/slurm/gpu_batched_svt.sh on an A100.

using GPUEnv
GPUEnv.activate(; include_jlarrays = false)
using CUDA, LinearAlgebra, Printf, Statistics, Random
using MriReconstructionToolbox
const MRT = MriReconstructionToolbox
const CUSOLVER = CUDA.CUSOLVER

CUDA.functional() || error("no functional CUDA device")
const REPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20

# Rank-deficient slices plus noise, so thresholding keeps a few singular values per slice.
function blocks(m, n, nb; rank = 3)
    Random.seed!(0)
    A = zeros(ComplexF32, m, n, nb)
    for b in 1:nb
        A[:, :, b] = randn(ComplexF32, m, rank) * randn(ComplexF32, rank, n) .+ 0.05f0 .* randn(ComplexF32, m, n)
    end
    return CuArray(A)
end

shrink(s, τ) = ifelse(s > τ, 1 - τ / s, zero(s))

# Current path.
gram_host(A, τ) = MRT.ProximalOperators.batched_svt!(A, τ)

# Gram matrices and their eigendecompositions both on the device (`heevjBatched`, n ≤ 32).
function gram_device(A, τ)
    m, n, nb = size(A)
    G = CUDA.zeros(eltype(A), n, n, nb)
    CUBLAS.gemm_strided_batched!('C', 'N', one(eltype(A)), A, A, zero(eltype(A)), G)
    w, V = CUSOLVER.heevjBatched!('V', 'U', G)
    s = sqrt.(max.(w, 0))
    total = sum(max.(s .- τ, 0))
    W = V .* reshape(shrink.(s, τ), 1, n, nb)
    Wf = CUDA.zeros(eltype(A), n, n, nb)
    CUBLAS.gemm_strided_batched!('N', 'C', one(eltype(A)), W, V, zero(eltype(A)), Wf)
    Y = similar(A)
    CUBLAS.gemm_strided_batched!('N', 'N', one(eltype(A)), A, Wf, zero(eltype(A)), Y)
    copyto!(A, Y)
    return total
end

# Full batched Jacobi SVD (m, n ≤ 32).
function svdj_batched(A, τ)
    m, n, nb = size(A)
    U, S, V = CUSOLVER.gesvdj!('V', copy(A))
    k = min(m, n)
    total = sum(max.(S .- τ, 0))
    Us = view(U, :, 1:k, :) .* reshape(max.(S .- τ, 0), 1, k, nb)
    Vk = V[:, 1:k, :]
    CUBLAS.gemm_strided_batched!('N', 'C', one(eltype(A)), Us, Vk, zero(eltype(A)), A)
    return total
end

# Strided batched approximate SVD (`gesvda`, m ≥ n, no size limit).
function svda_batched(A, τ)
    m, n, nb = size(A)
    U, S, V = CUSOLVER.gesvda!('V', copy(A))
    total = sum(max.(S .- τ, 0))
    Us = U .* reshape(max.(S .- τ, 0), 1, n, nb)
    CUBLAS.gemm_strided_batched!('N', 'C', one(eltype(A)), Us, V, zero(eltype(A)), A)
    return total
end

# One device `svd!` per slice.
function svd_loop(A, τ)
    total = 0.0f0
    for b in axes(A, 3)
        F = svd!(A[:, :, b])
        d = max.(F.S .- τ, 0)
        total += sum(d)
        A[:, :, b] = F.U * Diagonal(d) * F.Vt
    end
    return total
end

const VARIANTS = [
    ("gram+host eig", gram_host, (m, n) -> true),
    ("gram+heevj", gram_device, (m, n) -> n <= 32),
    ("gesvdj batched", svdj_batched, (m, n) -> m <= 32 && n <= 32),
    ("gesvda batched", svda_batched, (m, n) -> m >= n),
    ("svd! loop", svd_loop, (m, n) -> true),
]

function timed(f, A0, τ)
    A = copy(A0)
    CUDA.synchronize()
    t = @elapsed begin
        f(A, τ)
        CUDA.synchronize()
    end
    return t, A
end

cases = [
    # (block voxels m, frames n, blocks nb)  — 4² and 8² blocks, 128² to 256² images, 3D
    (16, 10, 1024), (16, 30, 1024), (16, 30, 4096),
    (64, 10, 256), (64, 30, 256), (64, 30, 1024), (64, 30, 16384),
    (64, 8, 1024),
]

println(CUDA.name(CUDA.device()), ", ", REPS, " reps, median ms (max rel. deviation from gram+host eig)")
@printf("%-16s", "m×n×nb")
for (name, _, _) in VARIANTS
    @printf(" %16s", name)
end
println()
for (m, n, nb) in cases
    A0 = blocks(m, n, nb)
    τ = 0.5f0 * sqrt(Float32(m))
    active = filter(v -> v[3](m, n), VARIANTS)
    reference = copy(A0)
    gram_host(reference, τ)
    times = Dict(name => Float64[] for (name, _, _) in active)
    deviation = Dict{String, Float64}()
    for (name, f, _) in active  # warm-up, and correctness against the current path
        _, A = timed(f, A0, τ)
        deviation[name] = Float64(norm(A - reference) / norm(reference))
    end
    for _ in 1:REPS, (name, f, _) in active
        push!(times[name], first(timed(f, A0, τ)))
    end
    @printf("%-16s", "$(m)×$(n)×$(nb)")
    for (name, _, _) in VARIANTS
        if haskey(times, name)
            @printf(" %7.2f (%.0e)", 1000 * median(times[name]), deviation[name])
        else
            @printf(" %16s", "-")
        end
    end
    println()
end
