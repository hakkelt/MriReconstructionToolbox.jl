#!/usr/bin/env julia
#
# benchmark/batch_op_threshold.jl
#
# Measures where threading a `SimpleBatchOp`'s batch loop starts to pay, so that
# `MIN_BATCH_WORK_FOR_PARALLEL` in `src/threading_policy.jl` is a transcription of a
# measurement instead of the guess its docstring has always admitted it was.
#
# `_should_thread(op::AbstractOperator)` only looks at the wrapped operator's own element
# count. It has no knowledge of how many times the batch `mul!` will be called -- and in a
# solver's inner loop (CG inside ADMM inside an outer loop) that can be hundreds of times.
# Each threaded call pays a fixed cost this proxy kernel does not: `Task` allocation,
# `with_thread_budget`'s NestedThreading scope guard, and the join. This script measures the
# batch `mul!` **as actually called** -- one full forward + adjoint pair per sample, at the
# batch count (8) real reconstructions use -- rather than a single isolated call, so that
# fixed per-call overhead is not amortised away by the harness itself.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=benchmark -t 8 benchmark/batch_op_threshold.jl
#
# Output: a table on stdout and in `.temp/batch_op_threshold.md`.

using AbstractOperators
using AbstractOperators: create_BatchOp
using BenchmarkTools
using FFTWOperators
using LinearAlgebra
using Printf

BLAS.set_num_threads(1)

const NS = [8, 16, 32, 64, 128, 256]
const BATCH = 8   # matches the dynamic section's Td and the multi-coil sections' Nc
const OUTDIR = joinpath(@__DIR__, "..", ".temp")

# Per-item operator: a 2D DFT, the dominant cost in the per-frame/per-coil `mul!` this
# threshold actually gates (`GetIndex . DFT . DiagOp` in the encoding operator). GetIndex and
# DiagOp are memory-bound and cheap next to the transform, so the DFT alone is a fair proxy
# for the wrapped operator's total element-and-cost profile without pulling in the whole
# acquisition-info machinery.
function make_case(N)
    op = DFT(ComplexF32, (N, N))
    x = randn(ComplexF32, N, N, BATCH)
    y = similar(x)
    return op, x, y
end

function bench_variant(op, x, y, threaded::Bool)
    B = create_BatchOp(op, (BATCH,), (false, false, true); threaded)
    return @belapsed begin
        mul!($y, $B, $x)
        mul!($x, $B', $y)
    end
end

function run()
    rows = NamedTuple[]
    for N in NS
        op, x, y = make_case(N)
        t_serial = bench_variant(op, x, y, false)
        t_thread = bench_variant(op, x, y, true)
        push!(rows, (; N, n_per_item = N * N, t_serial, t_thread, speedup = t_serial / t_thread))
        @printf(
            "N=%-4d n/item=%-8d serial=%10.2f us  threaded=%10.2f us  x=%.2fx\n",
            N, N * N, t_serial * 1.0e6, t_thread * 1.0e6, t_serial / t_thread
        )
    end

    crossover = nothing
    for i in eachindex(rows)
        if all(r -> r.speedup >= 1.0, rows[i:end])
            crossover = rows[i].n_per_item
            break
        end
    end

    mkpath(OUTDIR)
    open(joinpath(OUTDIR, "batch_op_threshold.md"), "w") do io
        println(io, "# SimpleBatchOp threading crossover (batch = $BATCH, forward+adjoint pair)")
        println(io, "| N | n/item | serial (us) | threaded (us) | speedup |")
        println(io, "|---|---|---|---|---|")
        for r in rows
            @printf(
                io, "| %d | %d | %.2f | %.2f | %.2fx |\n",
                r.N, r.n_per_item, r.t_serial * 1.0e6, r.t_thread * 1.0e6, r.speedup
            )
        end
        println(io)
        println(io, crossover === nothing ? "No crossover found in swept range." : "Crossover: n/item >= $crossover")
    end
    println()
    println(crossover === nothing ? "No crossover found in swept range." : "Crossover: n/item >= $crossover")
    return rows
end

run()
