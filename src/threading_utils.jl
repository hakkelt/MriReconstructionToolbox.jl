"""
    @conditionally_enable_threading threaded expr

Run `expr` with every registered thread pool opened up when `threaded` is true, and with all
of them pinned to one thread when it is false.

!!! warning "The permitted branch raises BLAS, and cannot currently be told not to"
    `with_full_threads` sets every *counted* pool to `NestedThreading.capacity()`
    (`Threads.threadpoolsize()`), BLAS included, even past a lower count the caller chose —
    documented behaviour, and the reason `AbstractOperators._with_blas_threading` refuses to
    open a full-throttle scope of its own.

    The obvious guard, `with_full_threads(exclude = (:blas, :mkl))`, **does not work**:
    `exclude` is consulted only in `NestedThreading._run_guarded`, which walks
    `GUARDED_POOLS`. Counted pools go through `_enter!` / `_apply!`, which take no `exclude`
    at all, so `:blas`, `:mkl`, `:fftw` and `:nfft` cannot be excluded from anything.
    `:polyester` is the only registered guarded pool, hence the only name `exclude` can
    actually name. Unknown names are accepted silently. Measured, `-t 8`, starting from
    `BLAS.set_num_threads(2)`:

    ```
    outside                                  = 2
    with_full_threads()                      = 8
    with_full_threads(exclude=(:blas,:mkl))  = 8   # no effect
    with_restricted_threads(exclude=(:blas,))= 1   # no effect
    ```

    So the BLAS budget has to be narrowed from inside instead — see [`with_serial_blas`](@ref),
    which the iterative solve opens around itself. If NestedThreading grows an allowlist
    (`only = (:blas, :mkl)`) or extends `exclude` to counted pools, this branch should use it
    and the inner scope can go away.
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

Per-work-item size below which a threaded BLAS — and, with it, threading the solve at all —
costs more than it returns. Consulted by [`with_serial_blas`](@ref),
`maybe_disable_undecomposed_threading` and `slice_threading`.

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
    with_serial_blas(f, x)

Run `f()` with BLAS pinned to a single thread, restoring the previous budget afterwards (also
on exception). Returns `f()`'s value, and skips the save/restore when BLAS is already serial.

The two-argument form applies that only when `x` — the work item the call is about, e.g. the
solver's image variable — is smaller than [`serial_blas_threshold_bytes`](@ref), and otherwise
runs `f()` untouched. **Prefer it.** The size gate is not a refinement; it is the difference
between a 20x win and a 3.9x loss.

The size gate is necessary but not sufficient: it cannot see BLAS *level*, and level 3 responds
to threading in the opposite direction and by a larger margin. Callers must also skip this scope
entirely for work that is level-3-dominated — see [`uses_blas3`](@ref), which
`_iterative_reconstruct_core` consults before reaching here.

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

An operator that genuinely wants a threaded `gemm` (a large `MatrixOp`) is unaffected: it
resolves its own `threaded` flag through `AbstractOperators._blas_threaded`, outside this scope.

## Why the gate keys on the work item and not on batch width

The tables above say the decision really depends on two quantities — item size *and* whether
enough slabs are in flight to saturate the cores — but `_iterative_reconstruct_core` only ever
sees one work item; it cannot see how many slabs the caller has in flight, and the decomposition
executor that does know is several frames up. The two quantities disagree only for a wide batch
of large items, and there the gate opens BLAS while the outer loop is already using the machine
— i.e. the budget wins over this scope in exactly the case the gate gets wrong. Preferring the
budget there is the conservative direction (a wide batch of 16 MiB+ items is the one row where
threaded BLAS-1 is within noise either way), so this is left keyed on the item alone rather than
plumbed through.
"""
function with_serial_blas(f::F) where {F}
    prev = LinearAlgebra.BLAS.get_num_threads()
    prev == 1 && return f()
    LinearAlgebra.BLAS.set_num_threads(1)
    try
        return f()
    finally
        LinearAlgebra.BLAS.set_num_threads(prev)
    end
end

function with_serial_blas(f::F, x) where {F}
    return _work_item_bytes(x) < serial_blas_threshold_bytes() ? with_serial_blas(f) : f()
end

"""
    with_restricted_threads_if_needed(f)

Run `f()` restricted to a single thread on every registered pool, like
`NestedThreading.with_restricted_threads`, but skip entering the scope at all when BLAS is
already serial — mirroring [`with_serial_blas`](@ref)'s own early-out.

Every counted pool (`BLAS`, `MKL`, `FFTW`, `NFFT`, ...) is always set to the same applied
budget by `NestedThreading._apply!`, so BLAS already being serial means the whole scope would
be a no-op: entering it would still pay the enter/exit bookkeeping and the Polyester guard for
every registered pool, just to re-set values that already match. That is dead weight when the
caller is already serial — the decomposed `MultiThreadingExecutor` path, or a `-t 1` process —
which is exactly when `_iterative_reconstruct_core` reaches for this instead of
`with_restricted_threads` directly.
"""
function with_restricted_threads_if_needed(f::F) where {F}
    LinearAlgebra.BLAS.get_num_threads() == 1 && return f()
    return with_restricted_threads(f)
end

"""
    _should_thread_work_item(config, bytes) -> Bool

Whether a work item of `bytes` bytes is worth threading, given `config`. The single predicate
behind both [`maybe_disable_undecomposed_threading`](@ref) (an undecomposed whole-problem
variable) and `decomposition/execution.jl`'s per-slice `slice_threading` (one slice of a decomposed
problem): `config.threaded` must be on *and* the item must be at least
[`serial_blas_threshold_bytes`](@ref). See `with_serial_blas`'s docstring for the measurements
the threshold comes from.
"""
_should_thread_work_item(config, bytes) =
    config.threaded && bytes >= serial_blas_threshold_bytes()

"""
    maybe_disable_undecomposed_threading(config, method, acq_data) -> ReconstructionConfig

When a reconstruction has no batch dimensions to decompose over, `config.threaded` would
otherwise open every thread pool (BLAS, FFTW, NFFT, Polyester) to full capacity for a single
problem. Below [`serial_blas_threshold_bytes`](@ref) per variable that is a net loss: no single
layer dominates (an FFT plan threads a transform too small to benefit, the CG inner loop is
BLAS-1, Polyester on the gradient stencils actually helps a little), but the accumulated
fork/join and budget enter/exit overhead of a few hundred small threaded ops per solve adds up
(measured: 128²×8 TV solve, 3.2 s threaded vs 1.2 s serial). Return a `config` with `threaded`
forced off in that case, leaving larger single-volume problems (where a threaded 3-D FFT
genuinely pays) untouched.
"""
function maybe_disable_undecomposed_threading(config, method, acq_data)
    config.threaded || return config
    bytes = prod(variable_size(method, acq_data)) * sizeof(eltype(acq_data.kspace_data))
    return _should_thread_work_item(config, bytes) ? config : ReconstructionConfig(config; threaded = false)
end

_work_item_bytes(x::AbstractArray) = length(x) * sizeof(eltype(x))
_work_item_bytes(xs::Tuple) = isempty(xs) ? 0 : maximum(_work_item_bytes, xs)
_work_item_bytes(::Any) = 0   # unknown shape: fall back to the serial branch
