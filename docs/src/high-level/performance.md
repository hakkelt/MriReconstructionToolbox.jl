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
  for low-rank regularizers (`LowRank`, `LocallyLowRank`, `MultiScaleLowRank`), whose SVDs
  genuinely do benefit from threading. See [`with_serial_blas`](@ref).
- **Applies the same size rule to one slice of a decomposed problem.** When there are few enough
  slices that they are reconstructed one at a time, the work *inside* a slice threads only if
  that slice is itself large enough to pay for it — a 2-slice 128² problem runs serially inside
  even at `threaded = true`, which is ~10% faster end to end. With many slices the slice loop
  itself is the parallelism and the inside is serial regardless.
- **Runs the whole reconstruction serially when there is nothing to parallelise over.** If the
  problem has no batch dimension (a single 2-D slice, no coil/time/slice loop) and the image is
  small, `threaded = true` is ignored — threading a lone 128²-ish problem is a 2–3x loss, not a
  gain. Large single volumes (roughly 16 MiB per image and up) still thread. See
  [`maybe_disable_undecomposed_threading`](@ref).

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
it fixed. Left at `nothing` (the default), nothing is passed on and behaviour is unchanged —
MRT has always taken NFFT.jl's own defaults (`m = 5`, `σ = 2.0`, `NFFT.POLYNOMIAL`).

That default is far more accurate than MRIReco's operating point (`m = 3`, `σ = 1.25`,
`NFFT.TENSOR`) for accuracy the reconstruction does not use: measured per coil, 4.28 ms vs
1.00 ms for a forward error of 1.6e-7 vs 5.7e-5, while the reconstructed image's NRMSE is 0.085
either way. At MRIReco's operating point MRT reconstructs in 8.35 ms against MRIReco's 47.1 ms
for the same accuracy — MRT is faster at every point on this curve, but a caller who wants the
faster end of *MRT's own* curve now has a way to ask for it:

```julia
𝒜 = get_encoding_operator(info; m = 3, sigma = 1.25, precompute = NFFT.TENSOR)
```

Changing MRT's own defaults is a separate, measured decision (`IMPLEMENTATION_PLAN.md`, `C9`):
it would move every non-Cartesian result in the test suite and needs its own tolerances.

## Notes for developers

- `SERIAL_BLAS_THRESHOLD_BYTES` (16 MiB) is the size above which MRT stops forcing serial BLAS.
  It was fitted on one machine and is only known to within a factor of a few; it will move on
  different hardware.
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
MriReconstructionToolbox.SERIAL_BLAS_THRESHOLD_BYTES
MriReconstructionToolbox.uses_blas3
```
