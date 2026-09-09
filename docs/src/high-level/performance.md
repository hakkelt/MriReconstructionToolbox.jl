# Performance & Threading

Reconstruction speed on a multi-core machine depends on a handful of settings that are easy to
get wrong. This page covers what you need to do; MRT handles the rest.

## Quick checklist

```bash
export KMP_BLOCKTIME=0          # only if you use MKL
julia --project -t 8 recon.jl   # -t = number of cores you actually have
```

- Give Julia as many threads as you have physical cores — no more.
- On Slurm, make `--cpus-per-task` match `-t`, and always ask for memory explicitly.
- Don't call `BLAS.set_num_threads` yourself. MRT manages it during reconstruction.

That is usually all you need.

## Why threading can make things slower

Julia, BLAS, FFTW and NFFT each have their own thread pool, and by default each one sizes itself
from the machine without knowing about the others. Nest them and you get more threads than cores,
all paying synchronisation costs on work that was already busy.

MRI reconstruction makes this worse than usual, because the natural unit of work is small — a
128×128 slice is 1 MiB and fits in cache. Splitting *that* across 8 cores costs more in
coordination than it saves in arithmetic.

The right structure is: **parallelise across slices, keep each slice serial.** MRT does this for
you. Problems appear when something underneath the slice loop starts its own threads anyway.

Concretely, on an 8-core run of a typical iterative solve:

| BLAS threads | time    |
|--------------|---------|
| 1            | 2.7 s   |
| 8            | 15.6 s  |

Same code, same cores — nearly 6x slower just from letting BLAS thread. OpenBLAS and MKL behave
the same way here; this is not a bug in either.

## What MRT does for you

- **Splits work across slices** and tells the libraries underneath to stay single-threaded.
- **Skips threading for small FFTs**, where it would cost more than it saves.
- **Pins BLAS to one thread during the iterative solve** — except for large problems, and except
  for low-rank regularizers (`LowRank`, `LocallyLowRank`, `MultiScaleLowRank`,
  `StructuredLowRank`), whose SVDs
  genuinely do benefit from threading. See [`with_serial_blas`](@ref).
- **Applies the same size rule to one slice of a task-split problem.** When there are few enough
  slices that they are reconstructed one at a time, the work *inside* a slice threads only if
  that slice is itself large enough to pay for it — a 2-slice 128² problem runs serially inside
  even at `threaded = true`, which is ~10% faster end to end. With many slices the slice loop
  itself is the parallelism and the inside is serial regardless.
- **Runs the whole reconstruction serially when there is nothing to parallelise over.** If the
  problem has no batch dimension (a single 2-D slice, no coil/time/slice loop) and the image is
  small, `threaded = true` is ignored — threading a lone 128²-ish problem is a 2–3x loss, not a
  gain. Large single volumes (roughly 16 MiB per image and up) still thread. See
  [`maybe_disable_unsplit_threading`](@ref).

## What you have to set yourself

### `KMP_BLOCKTIME=0` — MKL only, before Julia starts

Intel's MKL keeps its worker threads spinning at full CPU for 200 ms after finishing each piece
of work. Those spinning threads crowd out the Julia threads doing the FFTs and prox steps.

```bash
export KMP_BLOCKTIME=0
```

On the benchmark above, with MKL and 8 BLAS threads, this is the difference between 16.1 s and
3.8 s.

!!! warning "This cannot be set from inside Julia"
    MKL reads the variable once, when it loads, which happens before any of your code runs.
    Setting `ENV["KMP_BLOCKTIME"]` in a script or a package is too late to have any effect, and
    the runtime function that claims to change it (`kmp_set_blocktime`) succeeds while silently
    doing nothing. It has to come from the shell, a job script, or your `.bashrc`.

    MRT warns you if it sees MKL loaded without it.

### Slurm

```bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
julia --project -t 8 recon.jl
```

Ask for memory explicitly — the default per-CPU allowance is small, and a single-core job will be
killed by a reconstruction that would otherwise fit in a few GB.

Julia detects your allocation correctly, so BLAS sizes itself to `--cpus-per-task`. Note that
`Sys.CPU_THREADS` reports the whole node regardless of what you were given; don't use it to size
anything.

### Pinning threads to cores

```julia
using ThreadPinning
pinthreads(:affinitymask)   # stays within your Slurm allocation
```

This matters when the OS scheduler would otherwise move your threads around and cost you memory
locality:

- **Multi-socket / multi-NUMA machines** — most HPC compute nodes, dual-socket workstations, and
  the larger Threadripper/EPYC desktops. A thread that migrates to the other socket reads its
  data across the interconnect. This is the big one.
- **Shared machines** — a login node or a workstation with other users, where contention makes
  the scheduler reshuffle runnable threads.
- **Benchmarking** — pinning removes a large source of run-to-run variance regardless of topology.

On a single-socket machine that you have to yourself (a typical laptop or desktop) it does
little, and aggressive pinning can fight the OS power management. `pinthreads(:affinitymask)` is
the safe default: it pins within whatever cores you were actually given and is a no-op if that
is the whole machine.

### OpenBLAS or MKL?

Either is fine. They perform the same on typical reconstruction work once threading is set up
correctly. MKL is somewhat better at the SVDs in low-rank methods; its downside is the
`KMP_BLOCKTIME` trap above. Loading `MKL.jl` replaces OpenBLAS rather than running alongside it,
so there is nothing to balance between them.

## If a reconstruction is slow

Try this first:

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
```

If that makes it faster, some library underneath the slice loop is threading when it shouldn't
be. Report it — that is a bug in MRT's threading setup, not something you should have to work
around.

Otherwise, check what is actually configured:

```julia
using LinearAlgebra, FFTW
@show Threads.nthreads() BLAS.get_num_threads() FFTW.get_num_threads()
@show get(ENV, "KMP_BLOCKTIME", "unset")
```

## Non-Cartesian accuracy / speed trade-off

`get_fourier_operator`/`get_encoding_operator` take `m`, `sigma` and `precompute` keywords that
forward straight to `NFFTOp`/NFFT.jl, exposing the gridding operating point instead of leaving
it fixed. Left at `nothing` (the default), **MRT's own default operating point is used**: `m = 4`,
`σ = 1.5`, `precompute = NFFT.POLYNOMIAL` (`DEFAULT_NFFT_M`, `DEFAULT_NFFT_SIGMA`,
`DEFAULT_NFFT_PRECOMPUTE` in `src/encoding/fourier_operators.jl`).

That default was previously NFFT.jl's own (`m = 5`, `σ = 2.0`, `NFFT.POLYNOMIAL`), which is far
more accurate than the reconstruction needs. Measured on a 128×128 radial phantom
(`GeometricMedicalPhantoms`'s Shepp-Logan, 256 samples × 128 spokes), single thread, timings
interleaved round-robin across configs rather than one after another (a single measurement on
this shared login node can swing 30-60%):

| m | σ | precompute | forward (min/median ms) | adjoint (min/median ms) | forward rel. error vs `m=5,σ=2` |
|---|---|---|---|---|---|
| 5 | 2.00 | POLYNOMIAL (former default = NFFT.jl's own) | 8.5 / 9.7 | 7.5 / 8.6 | 0 (reference) |
| 4 | 2.00 | POLYNOMIAL | 7.0 / 8.1 | 5.5 / 6.4 | 3.8e-8 |
| **4** | **1.50** | **POLYNOMIAL (new MRT default)** | **4.0 / 4.5** | **4.6 / 5.3** | **2.5e-7** |
| 3 | 2.00 | POLYNOMIAL | 6.0 / 6.8 | 4.3 / 4.9 | 2.4e-6 |
| 3 | 1.50 | POLYNOMIAL | 2.8 / 3.2 | 3.3 / 3.8 | 1.7e-5 |
| 3 | 1.25 | TENSOR (MRIReco's point) | 2.5 / 2.9 | 2.8 / 3.1 | 7.1e-5 |
| 2 | 1.50 | POLYNOMIAL | 2.3 / 2.6 | 2.4 / 2.8 | 7.4e-4 |
| 2 | 1.25 | TENSOR | 2.0 / 2.3 | 1.9 / 2.2 | 2.1e-3 |

The direct (gridding) reconstruction's NRMSE against the phantom does not move outside
run-to-run noise across this whole table (consistent with the older measurement below, where
NRMSE was ≈ 0.085 at both the old default and MRIReco's point) — non-Cartesian gridding-adjoint
NRMSE is dominated by sampling/DCF artifacts, not by the gridding kernel's own accuracy, so
forward relative error against the reference is the right proxy for "is this operating point
accurate enough."

`m=4, σ=1.5` is picked as the new default because its forward error (2.5e-7) is indistinguishable
from full accuracy while it runs about 2x faster on the forward transform and about 1.6x faster
on the adjoint than the old default; every point below it in the table trades measurably more
accuracy for comparatively little extra speed. This default was chosen to keep passing the
existing NFFT test suite (`test/test_encoding_op.jl`'s `"NFFT operating point (S6)"` item and the
`:nfft`-tagged tests in `deps/AbstractOperators/NFFTOperators/test`) without loosening any
tolerance.

Independently, MRIReco's operating point (`m = 3`, `σ = 1.25`, `NFFT.TENSOR`) is faster still, at
real accuracy cost: measured per coil, 4.28 ms vs 1.00 ms for a forward error of 1.6e-7 vs 5.7e-5
against MRT's old (`m=5,σ=2`) default, while the reconstructed image's NRMSE is 0.085 either way
(measured whole multi-coil DCF adjoint, single thread, `benchmark/comparison/scripts/
run_noncart.jl`, 2026-09-04: MRT at its old default 25.9 ms, MRT at MRIReco's operating point
6.1 ms, MRIReco 27.3 ms — all at NRMSE ≈ 0.085 against the phantom). At MRIReco's operating point
MRT is faster than MRIReco while gridding more accurately; ask for it explicitly if you want the
faster, less accurate end of the curve:

```julia
𝒜 = get_encoding_operator(info; m = 3, sigma = 1.25, precompute = NFFT.TENSOR)
```

or go back to the old high-accuracy default with `m = 5, sigma = 2.0`.

## Notes for developers

- `serial_blas_threshold_bytes()` (16 MiB by default) is the size above which MRT stops forcing
  serial BLAS — and, since it also gates whether a solve threads at all, the size above which a
  small problem is allowed to use more than one thread. Re-fitted on real solves on an exclusive
  node, the crossover depends on the BLAS backend: below 4 MiB the threaded and serial paths are
  within 2% of each other on both, but at 8 MiB threading is a 1.3x *loss* on OpenBLAS and a
  1.4x *win* on MKL. 16 MiB is the value that is safe on both. Move it with
  `MriReconstructionToolbox.set_serial_blas_threshold_bytes!` or the
  `MRT_SERIAL_BLAS_THRESHOLD_BYTES` environment variable if you know your backend and hardware.
- `NestedThreading`'s `exclude` keyword only affects Polyester. Passing `:blas`, `:mkl`, `:fftw`
  or `:nfft` is accepted silently and does nothing, which is why MRT narrows the BLAS budget from
  the inside instead.
- `with_full_threads` raises thread counts to capacity, overriding a lower count you set
  deliberately. Don't wrap one around hand-tuned settings.
- `--gcthreads` does not help the residual `-t 8` cost (GC growing from ~10 ms to ~35-46 ms per
  solve on an allocation-heavy serial solve): measured at `--gcthreads=1`, no change. The lever
  for that residual is allocation, not GC thread count — see the per-iteration prox-buffer work
  tracked as `IMPLEMENTATION_PLAN.md`'s `C7`.

## API

```@docs
MriReconstructionToolbox.with_serial_blas
MriReconstructionToolbox.serial_blas_threshold_bytes
MriReconstructionToolbox.set_serial_blas_threshold_bytes!
MriReconstructionToolbox.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES
MriReconstructionToolbox.uses_blas3
MriReconstructionToolbox.maybe_disable_unsplit_threading
```
