"""
    @conditionally_enable_threading threaded expr

Run `expr` with every registered thread pool opened up when `threaded` is true, and with all
of them pinned to one thread when it is false.

The permitted branch raises every counted pool (BLAS included) to `NestedThreading.capacity()` —
that is `with_full_threads`' documented behaviour, and is the reason
`AbstractOperators._with_blas_threading` refuses to open a full-throttle scope of its own. Before
NestedThreading 0.1.1, `exclude` could not narrow a counted pool at all (only the guarded
`:polyester` pool), so the iterative solve had to narrow BLAS from inside its own scope instead —
see [`with_serial_blas`](@ref), which now does exactly that via `with_thread_default(f, 1; only =
(:blas, :mkl))`: NestedThreading 0.1.1's allowlist, and 0.1.2's soft default.
"""
macro conditionally_enable_threading(threaded, expr)
    return quote
        if $(esc(threaded))
            with_full_threads() do
                $(esc(expr))
            end
        else
            with_restricted_threads() do
                $(esc(expr))
            end
        end
    end
end

"""
    DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES

MRT's shipped value for [`serial_blas_threshold_bytes`](@ref): 16 MiB. See
[`with_serial_blas`](@ref) for the measurements it comes from, and
[`set_serial_blas_threshold_bytes!`](@ref) for why it is settable at all.
"""
const DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES = 16 * 2^20

const _SERIAL_BLAS_THRESHOLD_BYTES = Ref(DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES)

"""
    serial_blas_threshold_bytes() -> Int

Per-work-item size below which a threaded BLAS costs more than it returns, and below which one
work item cannot keep the machine busy on its own. Consulted by [`with_serial_blas`](@ref) and,
through `_should_thread_work_item`, by `suggest_executor` when it chooses between spreading
slices over threads and running them one at a time.

It is no longer consulted to decide whether an *operator* may thread: that is the operator's own
call, made per input by `AbstractOperators.threading_threshold` and
`ProximalOperators.should_thread`.

Defaults to [`DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES`](@ref), overridable per process with
[`set_serial_blas_threshold_bytes!`](@ref) or the `MRT_SERIAL_BLAS_THRESHOLD_BYTES`
environment variable.
"""
serial_blas_threshold_bytes() = _SERIAL_BLAS_THRESHOLD_BYTES[]

"""
    set_serial_blas_threshold_bytes!(bytes::Integer) -> Int

Set [`serial_blas_threshold_bytes`](@ref) for this process. Pass
[`DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES`](@ref) to restore the shipped value.

# Why this is a knob and not a constant

Re-fitted on real solves on an exclusive cluster node (`IMPLEMENTATION_PLAN.md` `C6.3`), the
crossover turned out to depend on the BLAS backend by more than the octave a single number
could bracket. `serial / threaded` wall time for a whole solve, 8 threads, one node:

| item  | OpenBLAS | MKL   |
|-------|----------|-------|
| ≤4 MiB| 1.00-1.02x | 1.00-1.02x |
| 8 MiB | 0.76x, 0.88x | 1.08x, 1.38x |
| 16 MiB| 1.13x    | 1.35x |
| 64 MiB| 1.66x    | —     |

Below 4 MiB the two paths are indistinguishable, so the exact value does not matter there.
Between 4 and 16 MiB they disagree in *direction*: threading an 8 MiB solve is a 1.3x loss on
OpenBLAS and a 1.4x win on MKL. The shipped 16 MiB is the value that is safe on both — it
gives up some MKL throughput between 4 and 16 MiB rather than risking an OpenBLAS
pessimisation — and a site that knows its backend and hardware can move it:

```julia
ENV["MRT_SERIAL_BLAS_THRESHOLD_BYTES"] = 4 * 2^20   # before `using MriReconstructionToolbox`
MriReconstructionToolbox.set_serial_blas_threshold_bytes!(4 * 2^20)   # or at any time
```
"""
function set_serial_blas_threshold_bytes!(bytes::Integer)
    @argcheck bytes >= 0 "the threshold is a byte count"
    return _SERIAL_BLAS_THRESHOLD_BYTES[] = Int(bytes)
end

# Read once at load, so a job script can set the threshold without touching the caller's code.
# A malformed value is a warning, not an error: it must not take down a reconstruction.
function _init_serial_blas_threshold!()
    raw = get(ENV, "MRT_SERIAL_BLAS_THRESHOLD_BYTES", nothing)
    raw === nothing && return nothing
    parsed = tryparse(Int, strip(raw))
    if parsed === nothing || parsed < 0
        @warn "ignoring malformed MRT_SERIAL_BLAS_THRESHOLD_BYTES" value = raw
        return nothing
    end
    set_serial_blas_threshold_bytes!(parsed)
    return nothing
end

"""
    with_serial_blas(f)

Run `f()` with BLAS at a single thread *by default*, restoring the previous count afterwards
(also on exception). Returns `f()`'s value, and opens no scope when BLAS is already serial.

This is a soft default (`NestedThreading.with_thread_default`), not a hard limit: a call inside
`f` that is known to be worth threading takes BLAS back for itself with
`NestedThreading.with_thread_grant`, up to whatever hard limit is open around it. The operator
stack does that for three kinds of call, each behind a size gate checked before any scope opens:

| call                                   | gate                                             | default         |
|----------------------------------------|--------------------------------------------------|-----------------|
| dense `svd!` / `eigen!` (low-rank prox)| `ProximalOperators.FACTORIZATION_THREAD_WORK`    | `m·n·min(m,n)` ≥ 2^22 |
| `MatrixOp` / `LMatrixOp` `gemm`        | `AbstractOperators.BLAS3_THREAD_WORK`            | `m·n·k` ≥ 2^25  |
| a CG step's `dot` / `axpy!`            | `ProximalAlgorithms.CG_BLAS_THREAD_BYTES`        | 8 MiB MKL, 16 MiB OpenBLAS |

Each is a `Ref`; set it to `typemax(Int)` to keep that kind of call serial. When a grant closes on
OpenBLAS, NestedThreading shuts the pool's workers down (`NestedThreading.park_openblas`), since
otherwise they spin for `OPENBLAS_THREAD_TIMEOUT` on the cores the next FFT or `@threads` region
needs.

Only `:blas` and `:mkl` are narrowed. FFTW, NFFT and Polyester are left alone deliberately: each
kernel that uses them decides for itself whether its own input is big enough to thread
(`AbstractOperators.threading_threshold`, `ProximalOperators.should_thread`), and BLAS is the one
pool with no equivalent per-call policy.

`_iterative_reconstruct_core` wraps every solve in this, with no size gate. A gated version used
to exist, applying it only below [`serial_blas_threshold_bytes`](@ref); the tables below are what
that gate was fitted to, and they are all *wide-batch* or whole-solve measurements. What they miss
is that the reconstruction path narrows BLAS for a single work item at a time, where a threaded
BLAS loses at every size measured. Real data, 256x256 single-coil, TV-ADMM 20 iterations, 8
threads: 584.0 ms with BLAS open against 233.3 ms pinned.

What that gate could not see at all is BLAS *level*, and level 3 responds to threading in the
opposite direction and by a larger margin. Measured on `x1001c4s3b0n1` under SLURM, MKL, a single
work item, serial → threaded:

| allocation | `-t` | level-1, 16 MiB item | level-3, 4096×1024 SVD |
|------------|------|----------------------|------------------------|
| 8          |  8   | 13.14 s → 12.64 s    | 8.14 s → 2.55 s        |
| 16         |  8   | 13.25 s → 12.62 s    | 8.55 s → 2.38 s        |
| 16         | 16   | 13.80 s → 12.58 s    | 8.46 s → 2.36 s        |
| 32         |  8   | 12.97 s → 13.05 s    | 8.49 s → 2.19 s        |

That is why the level-3 calls are granted per call rather than the whole solve being left
threaded, as a low-rank solve used to be: a solve-wide exemption also threads every level-1 call
around the SVDs (OpenBLAS, 8 threads, locally-low-rank cine: 533 ms threaded, 294 ms serial).

# Why

The BLAS calls on the iterative-reconstruction path are level 1 — the `dot` / `axpy!` / `norm`
of the CG inner loop, the residual norms of ADMM — over one slab's worth of elements. Threaded
BLAS pays a fork/join barrier per call, and below a few million elements that barrier dwarfs
the arithmetic. Above it, the extra memory bandwidth starts to pay.

Measured on the cluster `test` node, MKL, 8 Julia threads pinned to 8 cores, `ComplexF32`,
CG-shaped iterations. "wide" means enough slabs to saturate the cores, "narrow" means a single
slab, where a serial BLAS leaves seven cores idle (`cpu/wall = 0.99` on those rows):

| work item      | batch  | BLAS=1  | BLAS=8  |               |
|----------------|--------|---------|---------|---------------|
| 128²×8  (1 MiB)| wide   |  2.73 s | 14.91 s | serial 5.5x   |
| 256²×8  (4 MiB)| wide   | 14.80 s | 64.76 s | serial 4.4x   |
| 512²×8 (16 MiB)| wide   | 21.29 s | 26.35 s | serial 1.24x  |
| 128²×8  (1 MiB)| narrow |  0.51 s |  0.71 s | serial 1.4x   |
| 256²×8  (4 MiB)| narrow |  6.81 s |  7.02 s | serial 1.03x  |
| 512²×8 (16 MiB)| narrow | 21.21 s | 19.78 s | **threaded 1.07x** |
| 512²×32(64 MiB)| narrow | 43.94 s | 38.35 s | **threaded 1.15x** |

The same sweep over a BLAS-3 (SVD) workload, of the kind the locally-low-rank and multi-scale
low-rank prox steps run, shows the inversion far more sharply:

| block        | batch  | BLAS=1  | BLAS=8  |               |
|--------------|--------|---------|---------|---------------|
| 256×64       | wide   |  2.95 s |  3.98 s | serial 1.35x  |
| 1024×256     | wide   |  7.45 s | 25.28 s | serial 3.4x   |
| 2048×512     | narrow |  4.87 s |  2.20 s | **threaded 2.2x** |
| 4096×1024    | narrow | 18.51 s |  5.22 s | **threaded 3.5x** |

So: with batch width to saturate the cores, serial BLAS wins at every size tested — the outer
loop is already using the machine and BLAS's barriers are pure overhead. Without it, serial
BLAS runs one core out of eight, and past roughly 16 MiB per item that idle capacity is worth
more than the barriers cost. Hence the threshold, and hence why this cannot be an unconditional
scope.

## Core count

Repeating the decisive rows under SLURM on `x1001c4s3b0n1`, where the cgroup limit is real
rather than an affinity mask, across four allocations (serial → threaded):

| allocation | `-t` | wide batch, 1 MiB item | narrow batch, 16 MiB item |
|------------|------|------------------------|---------------------------|
| 8          |  8   | 2.09 s → 15.59 s       | 13.14 s → 12.64 s         |
| 16         |  8   | 2.03 s → 22.32 s       | 13.25 s → 12.62 s         |
| 16         | 16   | 2.06 s → 34.22 s       | 13.80 s → 12.58 s         |
| 32         |  8   | 2.03 s → 39.66 s       | 12.97 s → 13.05 s         |

Serial timings are flat across allocations; the threaded penalty on small items grows with the
allocation (7.5x → 19.5x), so more cores make serialising *more* valuable, not less. The
allocation-to-`-t` ratio itself is not a factor: rows 2 and 3 differ only in `-t` and their
narrow-batch numbers are identical, and the ratio-4 row (32 cores, `-t 8`, 24 cores the Julia
loop can never touch) is the one where threaded BLAS-1 stops winning at all.

## What the default budget actually is

`LinearAlgebra.__init__` derives its budget from `jl_effective_threads()`, never from `-t`, so
this cannot be inferred from `threaded` or `Threads.nthreads()`. It *is* allocation-aware: in
all four jobs above the default came out exactly equal to `--cpus-per-task` (8, 16, 16, 32)
while `Sys.CPU_THREADS` reported 64 throughout. So the budget tracks the allocation faithfully
— it just tracks the wrong quantity, since for a saturated batch loop the right answer is 1.
Note that a bare `taskset` window on the login node gives a different answer again (8 CPUs
allowed → a default of 4), so do not assume the two environments agree.

An operator that genuinely wants a threaded `gemm` (a large `MatrixOp`) is unaffected: it grants
itself BLAS's threads back, see the gate table above.

## Why the gate keys on the work item and not on batch width

The tables above say the decision really depends on two quantities — item size *and* whether
enough slabs are in flight to saturate the cores — but `_iterative_reconstruct_core` only ever
sees one work item; it cannot see how many slabs the caller has in flight, and the task-splitting
executor that does know is several frames up. The two quantities disagree only for a wide batch
of large items, and there the gate opens BLAS while the outer loop is already using the machine
— i.e. the budget wins over this scope in exactly the case the gate gets wrong. Preferring the
budget there is the conservative direction (a wide batch of 16 MiB+ items is the one row where
threaded BLAS-1 is within noise either way), so this is left keyed on the item alone rather than
plumbed through.
"""
function with_serial_blas(f::F) where {F}
    LinearAlgebra.BLAS.get_num_threads() == 1 && return f()
    return with_thread_default(f, 1; only = (:blas, :mkl))
end

"""
    _should_thread_work_item(config, bytes) -> Bool

Whether a work item of `bytes` bytes is large enough to occupy the machine by itself:
`config.threaded` must be on *and* the item must be at least
[`serial_blas_threshold_bytes`](@ref).

This is a question about **how to spread slices**, not about whether an individual operator
should thread. `suggest_executor` asks it to choose between spreading slices over threads and
running them one at a time, and `slice_threading` asks it to keep the inside of a slice serial
while a [`MultiThreadingExecutor`](@ref) already has every thread busy with whole slices.
Whether a given kernel is worth threading at a given size is not decided here at all: it belongs
to the operator, and `AbstractOperators.threading_threshold` / `ProximalOperators.should_thread`
decide it per operator, per input.
"""
_should_thread_work_item(config, bytes) =
    config.threaded && bytes >= serial_blas_threshold_bytes()
