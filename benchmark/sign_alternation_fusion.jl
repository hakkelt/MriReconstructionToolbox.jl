# A sign alternation fused into the pointwise run of a `Compose`, on the shape of a multi-coil
# Cartesian frame: coil expansion, sensitivity weighting, 2-D DFT, sign alternation and a k-space
# mask (`M S F C B`). Three ways to apply it:
#
#   fused     the whole chain as it runs now, the sign alternation in the DFT's epilogue run;
#   separate  the chain without the sign alternation, fused, plus the sign alternation as a pass
#             of its own: how the chain ran before the sign alternation could join a run;
#   unfused   every operator a pass of its own.
#
#   julia --project=benchmark --threads=N benchmark/sign_alternation_fusion.jl [reps]

using Ristretto
using Ristretto: AbstractOperators, FFTWOperators
using Ristretto.AbstractOperators: DiagOp, BroadCast, Eye, GetIndex, Compose, get_normal_op
using Ristretto.FFTWOperators: DFT, SignAlternation
using LinearAlgebra, Printf, Random

const AO = AbstractOperators
const REPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
const T = ComplexF32

function unfused!(y, C, x)
    AO._pw_sequential!(y, C.A, C.buf, x)
    return y
end

function best_time(f!, y, C, x)
    f!(y, C, x)
    return minimum(@elapsed(f!(y, C, x)) for _ in 1:REPS) * 1.0e3
end

Random.seed!(0)
println("threads = ", Threads.nthreads())
@printf("%-14s %11s %14s %13s %16s\n", "size", "fused [ms]", "separate [ms]", "unfused [ms]", "separate/fused")
for (n, nc) in (((128, 128), 8), ((256, 256), 8), ((256, 256), 16))
    dims = (n..., nc)
    B = BroadCast(Eye(T, n), dims)
    C = DiagOp(randn(T, dims))
    F = DFT(T, dims, (1, 2))
    S = SignAlternation(T, dims, (1, 2))
    M = get_normal_op(GetIndex(T, dims, (:, rand(n[2]) .< 0.4, :)))
    A = M * S * F * C * B
    @assert A isa Compose && any(op -> op isa SignAlternation, A.A)
    x = randn(T, n)
    y = AO.allocate_in_codomain(A)
    @assert unfused!(similar(y), A, x) == mul!(similar(y), A, x)
    rest = M * F * C * B
    k = similar(y)
    tf = best_time((o, c, i) -> mul!(o, c, i), y, A, x)
    ts = best_time((o, c, i) -> (mul!(k, rest, i); mul!(o, S, k)), y, A, x)
    tu = best_time(unfused!, y, A, x)
    @printf("%-14s %11.3f %14.3f %13.3f %16.2f\n", join(dims, "×"), tf, ts, tu, ts / tf)
end
