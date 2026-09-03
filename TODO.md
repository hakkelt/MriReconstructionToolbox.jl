# TODO — open observations

Context gathered from a code-quality review (reuse / simplification / efficiency / altitude) of the work
between `463c223` and `706042c` (vendored subtrees under `deps/` excluded). This file records **what was
observed and why it was left alone** — it is not a plan, and none of the entries below is scheduled or
sequenced.

---

## Residual observations

- `results = Array{AbstractArray}` in `src/reconstruction/decomposition.jl:72, 179` (type instability in slice container; noise relative to slice solve).
  **Resolved** (`IMPLEMENTATION_PLAN.md` `S5`, 2026-09-03): both sites now run the first slice
  outside the loop to learn the concrete result type, mirroring `execute_two_phase`'s existing
  `prelim` idiom, and allocate `results` concretely from it.
- The `disable_normalop_optimization` divergence in the component path is **intentional and documented** at `build_model.jl:141-144` (plain `ls` is always used as fast normal-operator path for a sum of shared operators requires upstream `HCAT` normal-op fusion), not a defect.
- `two local prox_of copies` in `test/test_reg_low_rank.jl` and repeated `using Wavelets` in `test/test_reg_shared.jl` folded into `test_snippets.jl` in Stage 0.

---

## Performance & Threading Findings (Multi-Toolbox Benchmarks)

Measured on the cluster `test` node (`x1001c4s3b0n1`, dual AMD EPYC 7352, 128x128 8-coil brain datasets). Raw results saved in:
- `comparison/results/benchmark_openblas_1threads.json`
- `comparison/results/benchmark_openblas_8threads.json`
- `comparison/results/benchmark_mkl_1threads.json`
- `comparison/results/benchmark_mkl_8threads.json`

| Method | Framework | 1T OpenBLAS | 1T Intel MKL | 8T OpenBLAS | 8T Intel MKL | NRMSE (GT) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Coil Adjoint** | **MRT** / SigPy / BART | **0.10 ms** / 0.63 / 3.89 | **0.10 ms** / 0.46 / 2.95 | **0.32 ms** / 0.62 / 6.15 | **0.32 ms** / 0.48 / 5.61 | `1.70e-07` |
| **Cartesian MC Adjoint** | **MRT** / SigPy / BART | **1.32 ms** / 2.90 / 10.71 | **1.31 ms** / 2.87 / 9.67 | **1.43 ms** / 2.98 / 19.73 | **1.54 ms** / 3.00 / 19.04 | `6.34e-09` |
| **DCF Adjoint (Gridding)**| **MRT** / MRIReco | **26.25 ms** / 27.71 | **24.37 ms** / 32.24 | **7.10 ms** / 35.06 | **23.96 ms** / 31.46 | `8.52%` |
| **CG-SENSE (10 it)** | **MRT** / SigPy / BART | **34.24 ms** / 69.60 / 322.33 | **37.64 ms** / 68.75 / 320.47 | 430.37 ms / **67.21** / 624.66 | 350.71 ms / **101.84** / 628.65 | `6.34e-09` |
| **Total Variation (30 it)**| **MRT** / BART | **1,086 ms** / 3,835 | **1,108 ms** / 3,800 | 17,090 ms / **2,795** | 27,821 ms / **2,795** | `1.16%` |
| **L1-Wavelet (30 it)** | **MRT** / BART | **218.9 ms** / 446.1 | **217.5 ms** / 444.0 | **498.2 ms** / 1,332 | **490.2 ms** / 1,318 | `0.838%` |
| **TGV (30 it)** | **MRT** / BART | **1,634 ms** / 4,901 | **1,707 ms** / 4,877 | **2,584 ms** / 4,455 | 9,671 ms / **4,313** | `0.967%` |
| **Locally Low-Rank (20 it)**| **MRT** / BART | **214.3 ms** / 281.3 | **217.4 ms** / 296.8 | **194.4 ms** / 476.3 | **169.1 ms** / 483.9 | `11.1%` |
| **Temporal TV (20 it)** | **MRT** / BART | **924.5 ms** / 2,524 | **929.7 ms** / 2,521 | **1,280 ms** / 1,863 | **1,400 ms** / 1,854 | `8.78%` |
| **GRAPPA (RSS)** | **MRT** | **15.15 ms** | **17.91 ms** | 46.99 ms | **22.35 ms** | `2.12%` |

### 1. ADMM CG-operator rebuild, and the 54M-allocation figure

- **Not a slowdown driver — re-measured 2026-09-01.** A 30-outer-iteration TV solve on the
  benchmark problem (128²×8, 2× undersampled) allocates ~267 MiB across **67k–129k** allocations
  total, and the count and bytes are *the same* with `threaded = true` and `threaded = false`.
  The "54.38 million allocations / 3.43 GiB" figure is not reproducible here; it must have come
  from a different (much larger, or component-path) problem or a misread. Whatever it was, it is
  not what makes 8 threads lose to 1 — see finding 2, which is.

- **Original diagnosis (wrong)**: the entry previously blamed "evaluating operator sums with `+`
  allocating intermediate heap buffers on every CG step", with the remedy "add
  `Sum(..., in_place=true)` / pre-allocated evaluation workspaces". That infrastructure already
  exists (`Sum` carries `bufC`/`bufD`; `Compose` carries `mid`; the CG inner loop in `cg.jl` is
  pure `mul!`/`axpy!`/`dot`), so the remedy would have been a no-op.

- **`admm.jl` CG-operator handling — fixed 2026-09-01 in the fork.** The old code:
  ```julia
  if rho_changed
      cg_operator = (iter.A' * iter.A) + sum(rho[i] * (iter.B[i]' * iter.B[i]) for i in …)
  else
      cg_operator = state.cg_operator      # never written since construction
  end
  ```
  rebuilt the operator on every ρ change but never stored it, so once ρ stabilised CG silently
  reverted to the initial ρ and solved the wrong system for the rest of the run. A naive
  write-back throws `MethodError: Cannot convert` — the field is concretely typed and the
  type-unstable `sum(generator)` rebuild has a different `Sum{…}` type.

  Fix: new `struct ADMMNormalOp` (in `admm.jl`) implementing `AᴴA + ∑ᵢ ρᵢ BᵢᴴBᵢ` as a single
  `mul!` that holds the fixed `AᴴA` / `BᴴB[i]` operators and reads `ρ` **live** from
  `iter.penalty_sequence.rho` (every `PenaltySequence` mutates that vector in place, never
  reassigns). Built once in `ADMMState`, never rebuilt — the `if rho_changed` branch is gone.
  One x-shaped scratch buffer for the `BᴴB` accumulation. Validated: full MRT reconstruction +
  regularization + integration + JET suites green; `recon_bench.jl` TV/TGV (adaptive
  `SpectralRadiusApproximationPenalty`, ρ moving each iteration) converge to the expected NRMSE.

- **`AᴴA` was still built twice per ADMM solve — resolved** (`IMPLEMENTATION_PLAN.md` `C3`,
  2026-09-03). `SqrNormL2WithNormalOp` builds `AᴴA` eagerly in its constructor, `parse.jl` handed
  ADMM only `f.A`, and `get_cg_operator` built `iter.A' * iter.A` again. The cached one is now
  threaded through, under a new optional `AHA` key on the `LeastSquaresTerm` assumption (only
  algorithms that actually form the normal operator ask for it) and a matching `AHA` field on
  `ADMMIteration`. **Measured saving is ~4 MiB, not the 57 MiB this file claimed**: the TV
  benchmark solve goes 78.18 → 74.19 MiB and 144.53 → 136.91 ms at one outer iteration, and
  245.05 → 241.13 MiB at 30, where the wall time is indistinguishable. Treat the 57 MiB figure as
  retracted.


### 2. Intra-Slice Threading vs. Batch Decomposition Scaling
- **Observation**: For single 2D slices ($N \le 256$, $\le 1\text{ MB}$ memory footprint), single-threaded execution (`threaded = false`) is **10x–25x faster** than multi-threading across 8 cores.
- **Mechanism**:
  1. A $128\times 128$ slice fits completely in L2 CPU cache ($256\text{ KB}$). Single-threaded execution runs at full cache clock speed with zero memory bus traffic.
  2. Multi-threading a small 2D array partitions $\approx 16\text{ KB}$ per thread, causing inter-core cache line false sharing and POSIX/Polyester barrier synchronization latency ($100+\ \mu\text{s}$) that dwarfs the arithmetic compute time of individual operations (e.g. $15\ \mu\text{s}$ FFT).
- **Architecture**: Multi-threading in MRI reconstruction is fundamentally designed for **outer batch/slice decomposition** (`ProblemDecompositionPlan` / `execute_regularized`), where each physical core processes an independent 2D slice serially out of its local L1/L2 cache without thread synchronization. Intra-slice operator multi-threading should be reserved for large 3D/4D volumes ($N \ge 512$ or 3D cubes).

- **Status: partially addressed.** The mechanism above is correct, but the entry recorded no fix,
  and the supporting machinery is further along than it suggests:
  - The nesting contract already exists.
    `deps/AbstractOperators/src/threading_policy.jl` documents `threaded = false` as a hard veto
    precisely so "a threaded batch/block loop switches its children off with `threaded = false`
    and must be able to rely on that". Outer batch decomposition + `threaded = false` operators is
    therefore the *supported* path, not something still to be designed.
  - Size gating exists, but only for FFTW.
    `_fftw_num_threads(kind, num_threads, threaded, length(x))` consults
    `fftw_threading_threshold(kind)`, so `DFT` already declines to thread small transforms.
    Passing `threaded` straight through in `get_fourier_operator`
    (`src/encoding/fourier_operators.jl`) instead of the old
    `num_threads = threaded ? nthreads() : 1` is what lets that heuristic run — keep that change.
    (The `AGENTS.md` "DFT accepts `num_threads`, not `threaded`" gotcha is stale; `DFT` accepts
    both, and `num_threads` wins when both are given.)
- **This is the dominant cause of the 8T regressions. Profiled 2026-09-01.** The benchmark TV
  solve, `threaded=true` vs `threaded=false`: ~1.4x slower on a `-t 4` login session, 1.24x on
  the clean `-t 8` `test` node. No *single* threaded layer is the culprit — the pin-the-whole-
  -solve fix is right because you have to turn all of it off at once:

  | what was varied (128²×8 TV, `-t 4`, threaded reconstruct vs serial) | ratio vs serial |
  |---|---|
  | everything threaded (FFT + Polyester + BLAS 4)                      | 1.39x |
  | BLAS forced to 1, FFT + Polyester still threaded                    | 1.42x  (BLAS ≈ noise here) |
  | BLAS 1 **and** Polyester disabled, FFT still threaded               | 1.64x  (Polyester was *helping*) |
  | isolated encoding normal-op `mul!` loop, FFT-threaded vs FFT-serial | 0.87x  (FFT threading ≈ neutral) |

  So: Polyester threading of the gradient/prox stencils is a net *win* (`disable_polyester_threads`
  over the whole case made it slower, 2.09 s vs 1.81 s). FFT plan threading is roughly neutral at
  this size. BLAS-1-vs-4 barely moves a 9k-element level-1 CG loop. What remains — the ~40% loss —
  is the **accumulated fork/join + NestedThreading budget enter/exit + `@spawn` overhead of a few
  hundred small threaded operations per solve**, where each op's own parallel speedup does not
  cover its coordination cost. An earlier note here blamed "Polyester workers spinning and
  contending with FFTW"; the `disable_polyester_threads` test refutes that — retracted.

- **Applied 2026-09-01:**
  - `src/utils.jl` — `maybe_disable_undecomposed_threading(config, method, acq_data)`: when the
    reconstruction has no batch dimension to decompose over and the variable is under
    `SERIAL_BLAS_THRESHOLD_BYTES`, force `threaded = false` for the whole reconstruction. A lone
    small 2-D problem has nothing to parallelise across and every threaded library underneath it
    is pure overhead. Large single volumes are untouched.
  - `src/reconstruction/reconstruct.jl` — both `isnothing(decomposition_plan)` branches (plain
    and component) call it before `@conditionally_enable_threading`.
  - `src/reconstruction/solve_core.jl` — the non-`uses_blas3` solve now runs inside
    `with_restricted_threads()` (BLAS **and** FFTW **and** Polyester → 1) rather than just
    `with_serial_blas`, when the work item is under the threshold. This also covers the decomposed
    sequential-executor path, which `maybe_disable_undecomposed_threading` does not reach.

  **Confirmed on the clean `test` node (SLURM, `--exclusive`), OpenBLAS 8T ÷ 1T:** CG-SENSE
  1.00x (was ~11x worse), TV 0.81x (was ~16x), TGV 0.94x, L1-Wav 0.87x, LR 0.88x, LLR 0.91x,
  tTV 0.88x. NRMSE unchanged everywhere. MKL matches. The collapses are gone.

- **The ~10-25% that remains at 8T is not the recon math — it is the cost of the `-t 8`
  process itself running a ~1 s allocation-heavy *serial* job.** The gate makes every FFT /
  gradient / CG step single-threaded; measured breakdown (gated-serial TV recon, `-t 1` vs `-t 8`):
  - **GC**: ~10 ms at `-t 1` → ~35-46 ms at `-t 8`, *independent of `--gcthreads`* (tested
    `--gcthreads=1`, no change). 8 mutator threads = more thread-local arenas/stacks to scan per
    collection, and the ADMM/prox path allocates ~200-760 MiB per solve. ≈ +2-3%.
  - **`pinthreads` + exclusive dual-socket node** widens it: on the login node (`taskset`, no
    pin) `-t 8` gated-serial is only +4% vs `-t 1`; on the SLURM node it is +22%. The 8 pinned
    Julia threads (+ 8 GC threads) share 8 cores while `jl_effective_threads`-derived BLAS
    default (32) also targets them between solves; plus NUMA first-touch.
  - **Scope machinery**: `with_restricted_threads()` + the PolyesterWeave guard wrap every gated
    solve — budget enter/exit + Polyester disable/restore, cheap at `-t 1`, more coordination at
    `-t 8`.

- **Still open:**
  0. Close the residual 8T gap: (a) cut the ADMM/prox per-solve allocation (200+ MiB is high —
     buffer reuse in the prox scratch, `TODO §5`); (b) for a gated solve, skip the
     `with_restricted_threads` / Polyester-guard wrapper entirely and run raw serial (the guard
     only needs to *narrow*, and at `threaded=false` there is nothing to narrow) —
     **resolved** (`IMPLEMENTATION_PLAN.md` `S4`, 2026-09-03): `solve_core.jl`'s gated branch now
     calls `with_restricted_threads_if_needed` (`src/utils.jl`), which checks
     `BLAS.get_num_threads() == 1` — the same signal `with_serial_blas` already uses — and skips
     entering the scope entirely when the caller is already serial; (c) consider
     `--gcthreads` guidance in the perf docs (it doesn't help here but a smaller heap might) —
     **resolved** (`IMPLEMENTATION_PLAN.md` `S7`, 2026-09-03): documented as a tested negative
     result in `docs/src/high-level/performance.md`.
  1. `maybe_disable_undecomposed_threading` gates on total variable bytes, not on FFT-transform
     size or SVD-block size. A large low-rank problem with tiny per-block SVDs would still be
     threaded (harmlessly — the outer FFTs benefit); a medium problem near the 16 MiB line is the
     ambiguous case. The threshold is the same one-node-fitted number as `SERIAL_BLAS_THRESHOLD_BYTES`.
  2. Per-operator size vetoes in `DSPOperators` / `WaveletOperators` / the sensitivity `DiagOp`
     are still absent; only FFTW has one. Less urgent now that the whole small solve goes serial,
     but relevant for the large-volume path where some operators are still below their own
     threshold.
  3. The decomposed **sequential-executor** path (few slices, `length(plan) ≤ nthreads()`) still
     opens `with_full_threads()` around the slice loop via `@conditionally_enable_threading`. The
     inner `with_restricted_threads` in `solve_core` covers the solve, but the per-slice operator
     build / adjoint / opnorm still run threaded. Size-gate `slice_threaded` in `run_slices!` /
     `execute_two_phase` the same way.

### Problem-decomposition threading audit (2026-09-01)

Does decomposition disable threading *within* each slice reconstruction? **Mostly, with one gap.**

| path | `slice_threaded` | correct? |
|---|---|---|
| `MultiThreadingExecutor` (`length(plan) > nthreads()`) | `false`, hard-coded (`decomposition.jl:105,193`) | yes — slices run serial, `@budgeted_threads` parallelises the slice loop |
| `SequentialExecutor` (few slices) | `config.threaded` (true) + `@conditionally_enable_threading` opens all pools | **the gap** — a small few-slice problem threads every library on a small work item |
| no decomposition (no batch dims) | was `config.threaded`; now gated by `maybe_disable_undecomposed_threading` | fixed for small problems |

The `MultiThreadingExecutor` design is sound. The two other paths were the leak; the
no-decomposition one (which is what the TV/TGV/L1-Wavelet benchmarks hit — 128²×8, no
coil/time/slice loop) is now fixed, the sequential-executor one is item 3 above.

### 3. Threaded BLAS on the iterative path (was: "Intel MKL OpenMP Thread Thrashing")

**Resolved.** The entry framed this as an MKL problem; it is not. Measurements below.

#### Benchmark

Synthetic reproducer of the ADMM/CG inner loop: 32 slabs of 128×128×8 `ComplexF32`, each
solved serially inside a Julia-level `@threads` loop, 300 iterations alternating small BLAS-1
calls (`dot` / `axpy!` / `norm`, exactly what `cg.jl` does) with non-BLAS work (FFTs,
broadcasts). 8 Julia threads pinned to 8 cores, FFTW at 1 thread. Best of 3.

| backend  | Julia threads | BLAS threads | `KMP_BLOCKTIME` | wall | cpu/wall |
|----------|---------------|--------------|-----------------|------|----------|
| OpenBLAS | 8 | 8 | n/a          | 15.66 s | 7.62 |
| OpenBLAS | 8 | 1 | n/a          | **2.83 s** | 6.25 |
| MKL      | 8 | 8 | default      | 16.10 s | 7.59 |
| MKL      | 8 | 8 | `0`, set pre-launch | 3.77 s | 5.65 |
| MKL      | 8 | 8 | `0`, set in Julia before `using MKL` | 6.07 s | 3.88 |
| MKL      | 8 | 8 | `kmp_set_blocktime(0)` at runtime | 16.17 s | 7.62 |
| MKL      | 8 | 1 | default      | **2.87 s** | 6.60 |
| MKL      | 1 | 8 | default      | 21.18 s | 7.57 |
| OpenBLAS | 1 | 8 | n/a          | 22.52 s | 7.45 |

#### What the numbers say

1. **Not MKL-specific.** OpenBLAS at 8 BLAS threads (15.66 s) is as bad as MKL (16.10 s), and
   at 1 BLAS thread the two are within noise (2.83 s / 2.87 s). The original table's "MKL is
   slower than OpenBLAS at 8T" is a second-order difference on top of a much larger effect
   that both backends share.
2. **Not oversubscription.** With a *single* Julia thread and no nesting at all, 8 BLAS threads
   still costs 21.18 s against ~2.8 s serial. Threaded BLAS is the wrong tool for level-1 calls
   at *this* size: the fork/join barrier per `dot` / `axpy!` / `norm` dwarfs the arithmetic.
   This is the same mechanism as finding 2, one layer down — **and, like finding 2, it is a
   size effect, not an absolute.** See the size sweep below.
3. **`KMP_BLOCKTIME` is real but secondary.** It is worth 16.1 s → 3.8 s for MKL at 8 BLAS
   threads, but serial BLAS (2.9 s) beats it, and once BLAS is serial the blocktime setting no
   longer matters.
4. **The blocktime knob only works before `libiomp5` loads.** It reads the variable once, at
   load, and never again. Setting it pre-launch: 3.77 s. Setting it from Julia before
   `using MKL`: 6.07 s. Setting it after MKL is loaded, or calling
   `ccall((:kmp_set_blocktime, "libiomp5.so"), Cvoid, (Cint,), 0)` — which resolves and
   returns cleanly — has **no effect at all**: 16.17 s. (`libmkl_rt` does not export the symbol.)

#### Size sweep: where "serial BLAS wins" stops being true

The table above is one problem size. Sweeping it (MKL, 8 Julia threads on 8 cores, `ComplexF32`;
"wide" = enough slabs to saturate the cores, "narrow" = a single slab):

| work item        | batch  | BLAS=1  | BLAS=8  |                    |
|------------------|--------|---------|---------|--------------------|
| 128²×8  (1 MiB)  | wide   |  2.73 s | 14.91 s | serial 5.5x        |
| 256²×8  (4 MiB)  | wide   | 14.80 s | 64.76 s | serial 4.4x        |
| 512²×8  (16 MiB) | wide   | 21.29 s | 26.35 s | serial 1.24x       |
| 128²×8  (1 MiB)  | narrow |  0.51 s |  0.71 s | serial 1.4x        |
| 256²×8  (4 MiB)  | narrow |  6.81 s |  7.02 s | serial 1.03x       |
| 512²×8  (16 MiB) | narrow | 21.21 s | 19.78 s | **threaded 1.07x** |
| 512²×32 (64 MiB) | narrow | 43.94 s | 38.35 s | **threaded 1.15x** |

And over a BLAS-3 (SVD) workload, of the kind the locally-low-rank / multi-scale low-rank prox
steps run — where the inversion is far sharper:

| block     | batch  | BLAS=1  | BLAS=8  |                    |
|-----------|--------|---------|---------|--------------------|
| 256×64    | wide   |  2.95 s |  3.98 s | serial 1.35x       |
| 1024×256  | wide   |  7.45 s | 25.28 s | serial 3.4x        |
| 2048×512  | narrow |  4.87 s |  2.20 s | **threaded 2.2x**  |
| 4096×1024 | narrow | 18.51 s |  5.22 s | **threaded 3.5x**  |

Reading:

- **With batch width, serial BLAS wins at every size tested**, BLAS-1 and BLAS-3 alike. The
  outer loop is already using the machine; BLAS's barriers are pure overhead. The margin
  narrows with size (5.5x → 1.24x) but never inverts.
- **Without batch width, serial BLAS runs one core out of eight** — `cpu/wall = 0.99` on those
  rows — and past roughly 16 MiB per item that idle capacity is worth more than the barriers
  cost. For BLAS-3 it is worth a great deal more.
- So the governing variable is *bytes per work item*, gated by whether the batch already
  saturates the cores. An unconditional serial-BLAS scope would be a 3.5x regression on a large
  single-volume low-rank reconstruction, which is a real workload here.

#### Why the previously applied fix did nothing

- `ext/MriReconstructionToolboxMKLExt.jl` **never loaded**: the `[weakdeps]` entry in
  `Project.toml` carried the UUID `33e6dc65-8f57-5167-9d61-e5704d37f427`, which is not MKL.jl
  (`33e6dc65-8f57-5167-99aa-e5a354878fb2`). Fixed.
- Even once loaded, its `ENV["KMP_BLOCKTIME"] = "0"` could not work: an extension *of MKL* runs
  after MKL, hence after `libiomp5`, is loaded. Per (4), that is too late.
- Its `mkl_set_dynamic(0)` `ccall` had the right ABI (lowercase MKL symbols take the argument by
  pointer) but **zero measured effect** on throughput — 3000³ gemm 0.242 s vs 0.236 s, 16 MiB
  axpy 0.506 s vs 0.554 s, both noise. Removed.
- The root cause — MRT's own thread budget on the iterative path — was not addressed at all.

#### Applied

- `Project.toml` — corrected the MKL weakdep UUID (was `…9d61-e5704d37f427`, not MKL.jl) and the
  compat (`MKL = "0.6, 0.7"` was unsatisfiable against NestedThreading's 0.9.1 → `"0.6 - 0.9"`),
  so the extension finally loads. Verified.
- `src/utils.jl` — `with_serial_blas(f)` / `with_serial_blas(f, x)` (pin BLAS to one thread,
  size-gated on `SERIAL_BLAS_THRESHOLD_BYTES`), and `maybe_disable_undecomposed_threading` (see
  finding 2 — force `threaded = false` for a small problem with no batch dimension).
- `src/reconstruction/solve_core.jl` — the non-`uses_blas3` solve runs inside
  `with_restricted_threads()` (BLAS + FFTW + Polyester → 1) when the work item is small, else
  `with_serial_blas`. Backend-agnostic, no environment setup.
- `src/reconstruction/reconstruct.jl` — both no-decomposition branches call
  `maybe_disable_undecomposed_threading` before opening thread pools.
- `src/regularization/*` — `uses_blas3` trait (`LowRank` / `LocallyLowRank` / `MultiScaleLowRank`
  → `true`) so their SVD-bound solves keep threaded BLAS.
- `ext/MriReconstructionToolboxMKLExt.jl` — reduced to a single `@warn` (`maxlog = 1`) when MKL
  is loaded without `KMP_BLOCKTIME=0`. It cannot fix the variable (libiomp5 reads it once, at
  load, before any Julia runs), so telling the user is all it can do.
- `src/MriReconstructionToolbox.jl` — the `__init__` that tried `get!(ENV, "KMP_BLOCKTIME", "0")`
  was **removed**: assigning `ENV` from `__init__` is still after `libiomp5` loads in the
  `using MKL; using MriReconstructionToolbox` order, so it never won the race. The doc page and
  the extension warning carry this instead.
- `@conditionally_enable_threading` left as-is (the `exclude = (:blas, :mkl)` it wants is not
  expressible — see the NestedThreading item below); the inner scopes cover it.
- `benchmarking/` — new folder: MRT-only perf harness (`recon_bench.jl`, `threading_sweep.jl`,
  `probe.jl`, `run_slurm.sh`) moved out of `/scratch`. `comparison/scripts/run_benchmarks.jl`
  reads `benchmarking/results/mrt_<backend>_<n>threads.json` as the MRT baseline column when
  present. `phantoms.jl` moved to `benchmarking/src/Phantoms.jl` (single source of truth).

Note that `LinearAlgebra.__init__` derives its default BLAS budget from `jl_effective_threads()`
— under SLURM this comes out exactly `--cpus-per-task`; under a bare `taskset` window on the
login node it is `n_allowed ÷ 2`. Never from `-t`. That is why this had to be an explicit scope
rather than something `threaded = false` could imply.

#### Upstream: `NestedThreading.exclude` is a silent no-op for counted pools

Found while trying to express the fix in the existing API. `exclude` is consulted **only** in
`NestedThreading._run_guarded`, which iterates `GUARDED_POOLS`. Counted pools are applied by
`_enter!` → `_apply!`, neither of which takes an `exclude` argument. Of the registered pools,
`:blas`, `:mkl`, `:fftw` and `:nfft` are all *counted*; `:polyester` is the only *guarded* one.
So `exclude` can only ever name `:polyester` — which is the single use in the package
(`@budgeted_batch`) and why this has gone unnoticed. Unknown names are accepted without error.

Measured, `-t 8`, after `BLAS.set_num_threads(2)`:

```
outside                                   = 2
with_full_threads()                       = 8
with_full_threads(exclude=(:blas,:mkl))   = 8   # silently ignored
with_restricted_threads(exclude=(:blas,)) = 1   # silently ignored
with_full_threads(exclude=(:nonsense,))         # accepted, no error
```

Two things worth upstreaming to `NestedThreading`:

1. **Bug**: either honour `exclude` in `_apply!` for counted pools, or validate the names and
   throw on one that cannot be excluded. Silently accepting a knob that does nothing is the
   worst of the three.
2. **Missing API**: an allowlist — `with_thread_budget(f, n; only = (:blas, :mkl))` — so
   "restrict exactly these pools, leave the rest alone" is expressible. A denylist cannot say
   it without enumerating every other pool, which then breaks whenever a new one registers.
   `with_serial_blas` in `src/utils.jl` is exactly `with_restricted_threads(only = (:blas, :mkl))`
   and is general enough to belong there rather than here: any library doing Julia-level
   parallelism over small work items with BLAS-1 inside (Krylov solvers, ODE solvers, proximal
   algorithms) hits the same wall, especially since `LinearAlgebra`'s default budget follows CPU
   affinity rather than `-t`. NestedThreading already documents that trap in
   `_with_blas_threading`'s notes but offers no primitive to act on it selectively.

#### Not done

- The benchmark above is the synthetic reproducer, not the real recon. Re-run
  `comparison/scripts/run_benchmarks.jl` at 8 threads on both backends and refresh the table at
  the top of this section; the 8T columns for TV/TGV/CG-SENSE are the ones expected to move.
- `SERIAL_BLAS_THRESHOLD_BYTES` is one number fitted to one node (dual EPYC 7352, 8 cores
  visible, MKL). The crossover is a memory-bandwidth-per-core property, so it will move on
  other hardware and with core count. It is also measured on the synthetic reproducer, not on
  a real solve. Treat 16 MiB as a starting point, not a constant — and note the sweep only
  brackets it between 4 MiB (serial still wins) and 16 MiB (threaded wins by 7%), so the true
  crossover is somewhere in that octave.
- The gate keys on the *work item*, not on batch width, because `_iterative_reconstruct_core`
  does not know how many slabs are in flight. For the multi-threaded executor that is already
  handled — `@budgeted_threads` gives each task `capacity ÷ ntasks` — so the gate only really
  decides the sequential-executor case, where batch width is 1 by construction. If the two ever
  disagree (a wide batch of large slabs), the budget wins, which is the conservative direction.
- Core partitioning between backends (as in `radial_recon/notebooks/profile.jl`,
  `openblas_pinthreads(cpuids[end÷2+1:end])`) is **not** needed here and was not adopted: MKL.jl
  replaces OpenBLAS through libblastrampoline rather than running alongside it, so there is only
  ever one pool to size.

---

## Cross-Toolkit Comparison Findings (MRT vs BART / SigPy / MRIReco, 2026-09-02/03)

Time-to-common-NRMSE-target race, 128²×8 (dynamic rows 64²×4×8), R≈1.8, 30 dB, single-threaded,
per-toolbox calibrated λ, all conventions aligned (`run_accuracy_race.jl`):

| method (target) | MRT | BART | SigPy | MRIReco |
|---|---|---|---|---|
| TV ≤ 0.005 | 419 ms (12 it) | 418 (`-i 40`) | 665 (8) | **295** (8) |
| L1-wavelet ≤ 0.010 | 181 (20) | 344 (20) | 321 (20) | **91** (20) |
| TGV ≤ 0.005 | **1071** (20) | 1181 (`-i 80`) | — | — |
| global low-rank ≤ 0.090 | 531 (12) | 1828 (`-i 150`) | — | **379** (12) |
| locally low-rank ≤ 0.055 | 1362 (30) | 1886 (`-i 150`) | — | **646** (20) |
| temporal TV ≤ 0.090 | **149** (3) | 1030 (`-i 80`) | — | — |

MRT beats BART and SigPy on every row, and wins CG-SENSE, the adjoint rows, and non-Cartesian
gridding at matched accuracy outright. MRIReco wins the four rows above, each with a named cause:

### 7. FISTA pays a full `normalize_op` instead of just `Lf`

MRT runs all 20 power iterations plus a separate normalization pass for L1-wavelet (~137 ms of a
322 ms solve). MRIReco's `power_iterations` early-exits at `rtol = 1e-3` for ~7 ms, because FISTA
only needs the number `Lf = ‖A‖²`, not a normalized operator. Per-iteration MRT is also 11.1 vs
9.0 ms despite less wavelet work (3 levels vs full depth) — suspect `FastForwardBackwardIteration`
evaluating `f_x`/`g_z` every iteration, an extra `A` application nothing consumes when `tol = 0`.
Net: MRIReco is 2x faster on L1-wavelet, and it is real (unlike the ADMM opnorm case already fixed
2026-09-02 — see Performance & Threading Findings above).

**Measured and mostly retracted** (`IMPLEMENTATION_PLAN.md` `C5.1`, 2026-09-03). On the shipping
baseline (after `S2`/`C2`) `normalize_op` is 72.40 ms of a 293 ms L1-wavelet solve, not 137 of
322. Decomposed: building `AᴴA` inside `powerit` is **0.41 ms** — so "do not build it twice" buys
nothing — and the `1/L * A` `Scale` wrapper costs **+4.3%** per application. The cost is the 20
power iterations themselves, and all 20 do run: the top of the spectrum is near-degenerate, the
`tol = 1e-3` rule would first fire at iteration 22, and at 20 iterations the estimate is still
0.4-0.7% *below* the true norm.

**"FISTA only needs `Lf`, not a normalized operator" was right, and MRT now does exactly that**
(`IMPLEMENTATION_PLAN.md` `C5.3`, 2026-09-03). `𝒜` is no longer rescaled: `L = ‖𝒜‖` is estimated
purely as a step size and passed as `Lf = n·L²`, so the problem solved is `½‖𝒜x − y‖² + R(x)`.
Measured: TV 1365.77 → 1178.08 ms, LLR 224.06 → 199.24 ms, L1-wavelet 285.87 → 262.10 ms, TGV
2239.83 → 2083.52 ms.

Chasing that also found what the old convention was doing. Substituting `x = Lv` in
`½‖(𝒜/L)x − y‖² + R(x)` gives `L²·[½‖𝒜v − y‖² + L·λ‖Ψv‖₁]`, so the weight actually applied was
`λ·L`, **and the returned image was `L` times the data's own units** — exactly `L` as `λ → 0`
(measured `‖x‖/‖x_true‖ = 1.5214` against `L = 1.5214`, TV and L1-wavelet alike). Amplitude-aligned
NRMSE hid it in every benchmark. `L` scales linearly with the sensitivity maps' own scaling (×3 ⇒
`L`×3) and varies with coil count (1.5250 at 8 coils, 1.0872 at 4), so the same `λ` regularized
~40% harder with 8 coils than 4. Both are gone. **Migration: a `λ` tuned before this reproduces
its old behaviour as `λ·L`**, and the regularized benchmark rows move by construction (TV
0.0116 → 0.0076, L1-wavelet 0.0084 → 0.0058, TGV 0.0097 → 0.0062) — that is the convention change,
not an accuracy win, and `C9` re-baselines the tables.

The 72 ms itself stays: cutting the 20 power iterations still makes `L` smaller, and a too-small
`Lf` is the direction that diverges. `C5.2` dropped.

The measurement did find a real defect: `powerit` drew its start vector from the **global RNG**,
and since the iteration does not converge that vector leaks into the result — `estimate_opnorm`
varied 6.5e-4 relative across six calls, and two identical `reconstruct` calls differed by 7.9e-4.
**Fixed**: `powerit`/`estimate_opnorm` take an `rng` defaulting to a fixed-seed generator, so a
reconstruction is now bitwise reproducible.

The `g_z` half of this entry — `prox!` returning the regularizer's value on every iteration when
nothing consumes it — is still open (`IMPLEMENTATION_PLAN.md` `C1`, second half).

### 8. Non-Cartesian gridding — not a speed gap, an accuracy mismatch

MRT takes NFFT.jl's defaults (m=5, σ=2.0, POLYNOMIAL); MRIReco hardcodes m=3, σ=1.25, TENSOR. Per
coil that is 4.28 ms vs 1.00 ms for an NFFT ~360x more accurate (forward error 1.6e-7 vs 5.7e-5) —
accuracy the recon never uses (NRMSE 0.085 either way). At MRIReco's operating point MRT is 8.35 ms
vs MRIReco's 47.1 ms — MRT wins 5.6x at matched accuracy. MRT's API does not currently expose the
NFFT plan options (m, σ, precomputation kind) to let a caller choose the operating point.

**Resolved** (`IMPLEMENTATION_PLAN.md` `S6`, 2026-09-03): `get_fourier_operator`/
`get_encoding_operator` now take `m`, `sigma`, `precompute` keywords forwarded to `NFFTOp`, left
at `nothing` by default so nothing changes unless a caller asks. MRT's own defaults are
unchanged — that is a separate, measured decision (`C9`). See "Non-Cartesian accuracy / speed
trade-off" in `docs/src/high-level/performance.md`.

### 9. `fftshift` emulation (`_alternate_sign!`) dominates the dynamic low-rank / LLR solve

**43%** of a 20-iteration dynamic low-rank solve, against FFTW's own 18% and the SVD prox's 2.5%.
Two causes, both in `FFTWOperators`:
- the inner loop recomputes per-element index parity every call; a hoisted + `@simd` rewrite
  measured 1.6-1.9x faster.
  **Resolved** (`IMPLEMENTATION_PLAN.md` `S2`, 2026-09-03): `_alternate_sign!` in
  `FFTWOperators/src/Shift.jl` now hoists the parity of every dimension but the first into a
  per-column value computed once, and vectorizes the true per-element alternation (dim 1) with
  `@simd`; the threaded branch parallelises the outer (column) loop only. Measured 2.68x
  (128²×8) and 2.76x (64²×4×8) `ComplexF32`, both above the 1.6-1.9x estimate above.
- the k-space shift pair inside `𝒜ᴴ𝒜` is never cancelled. Sign alternation commutes exactly with
  the (diagonal) sampling mask, but `get_normal_op(::Compose)` only folds the outermost factor, and
  `Compose`'s constructor only cancels directly-adjacent `Aᴴ A` — so a pair separated by the mask
  survives uncancelled.
  **Resolved** (`IMPLEMENTATION_PLAN.md` `C2`, 2026-09-03): a triple
  `can_be_combined`/`combine` rule in `FFTWOperators` rewrites `± M ±` to `M` for any square
  diagonal `M` carrying the same `dirs`. On the real MRT chain `get_normal_op(𝒜)` drops from
  9 operators to 7. Measured (1 thread, min of 3, rule reverted vs applied): CG-SENSE
  94.66 → 46.05 ms (**2.06x**), global low-rank 247.48 → 173.99 ms (1.42x), temporal TV
  877.07 → 724.52 ms (1.21x), LLR 250.38 → 224.06 ms (1.12x), TV 1490.84 → 1365.77 ms (1.09x),
  L1-wavelet 293.67 → 285.87 ms (1.03x), TGV 2256.61 → 2239.83 ms (1.01x). It also exposed a
  latent `Compose` bug that silently dropped an operator from `𝒜ᴴ𝒜` — see
  `IMPLEMENTATION_PLAN.md` `C2` and commit `ec75b5c`.

The combined "890 ms → ≈550 ms" estimate for the dynamic low-rank solve is superseded by the
per-row numbers above; the benchmark tables are regenerated once, in `C9`.

### Smaller MRT-internal gaps found during the same profiling pass

- `Variation` adjoint is 244 us against 21 us forward — 12x asymmetry in
  `_variation_adjoint_term`'s scalar loop.
  **Resolved** (`IMPLEMENTATION_PLAN.md` `S3`, 2026-09-03): the adjoint is rewritten in the
  forward's own flat/strided idiom (`_variation_adjoint_dim1!`/`_variation_adjoint_dim!` in
  `Variation.jl`), zero-allocating and matching the old scalar implementation exactly (dot-test,
  dense-transpose test, and an axis-of-length-2 edge case all still pass). Measured 9-10x on a
  128² adjoint (was ~12x asymmetry vs forward; forward itself is unaffected).
- `WaveletOp` declares `is_AcA_diagonal = true, diag_AcA = 1` but not `has_optimized_normalop`, so
  ADMM's `get_cg_operator` runs a real `dwt!`/`idwt!` pair per inner CG iteration for what is
  mathematically the identity.
  **Resolved** (`IMPLEMENTATION_PLAN.md` `S1`, 2026-09-03): `has_optimized_normalop`/
  `get_normal_op` added, both guarded on `L.wavelet isa Wavelets.WT.OrthoFilter` — the identity
  only holds for orthogonal families; a biorthogonal (lifting-scheme) wavelet now correctly
  reports `false` on every diagonal-identity trait instead of the previous unconditional `true`.
- `A'A` is built twice for ADMM — once eagerly by `normalop_ls` inside `SqrNormL2WithNormalOp`,
  again by `admm.jl:266` `get_cg_operator` — each with its own full set of k-space-sized `Compose`
  buffers (57 MiB allocated for a single-iteration solve).
- ADMM computes 6 vector norms and one extra `B'` application unconditionally per outer iteration
  (`admm.jl:418-428`); dead work under `FixedPenalty` with `tol = 0`.
- `Threads.@threads` over `eachindex(iter.g)` twice per outer iteration (`admm.jl:376, 403`) is a
  one-trip loop with a single regularizer.

What is already good (do not "optimize"): one `A`+`A'` costs 3.50 ms against a 2.95 ms floor for
two batched 128x128x8 in-place FFTs (MRIReco's fused `AHA` is 3.34 ms — on par). `NamedDimsOp` is
free (compile-time symbol-tuple compare, unwrapped before solve). 10 normal-op applications per
ADMM outer iteration vs 11 for BART/MRIReco/SigPy.

Full comparison-methodology pitfalls (BART `-i`/`-w 1`/`-e` flags, MRIReco's `-t 1` BLAS-pin bug,
TV/wavelet convention mismatches per toolbox, λ-must-match-inner-CG-exactness trap, and the
subsampling-mask-must-be-passed trap) are not repeated here — see project memory
`toolbox-comparison-pitfalls.md` for the source-level detail.

---

## Out of Scope

### 5. Per-iteration allocation in the new prox implementations
None of these were changed: they are all inner-loop scratch, and the natural remedy (buffers owned by the
term) needs a decision about thread ownership first — terms are materialized per slice, but nothing currently
documents whether a materialized term may be entered from more than one thread.
- `src/regularization/multi_scale_low_rank_reg.jl:43,45`
- `src/regularization/multi_scale_low_rank_reg.jl:53`
- `src/regularization/locally_low_rank_reg.jl:121,138-142`
- `src/regularization/locally_low_rank_reg.jl:76`
- `src/regularization/plug_and_play_reg.jl:40,54-55`

---

## Already applied (applied in `d9964eb` and Stage 0)

- **1. Component path consolidation**: Shared iterative driver (`_iterative_reconstruct_core` in `src/reconstruction/reconstruct.jl:337`), folding duplicated setup.
- **2. Lipschitz constant estimation (`Lf`)**: Sourced directly from built variables (`reconstruct.jl:370`).
- **3. Operator caching in decomposition**: `execute_two_phase` caches encoding operator `𝒜` across phases (`src/reconstruction/decomposition.jl:147`).
- **4. Concrete type in `prelim`**: Typed tuples in decomposition prelim array (`src/reconstruction/decomposition.jl:164-167`).
- **6. Defensive copy on solution extraction (`copy(~x_var)`)**: Profiled in Stage 1 benchmark. Measured memory delta is 32 KB out of 15.5 MB (0.21%), well below the 2% threshold. Retained as won't-fix to guarantee callers and multi-variable solvers never alias internal solver state.
- **7. Sparsifying domain dispatch**: `hard_threshold_reg.jl` refactored with `Val`-dispatch.
- **8. Entry point dispatch**: Entry points dispatch on tuple shape (`reconstruct.jl:78-96`).
- **9. Test-side duplication**: Centralized test snippets in `test/test_snippets.jl` (`RegTestSetup`, `ProxOf`).
- `get_affected_dims`: terms now implement the `::Nothing` method and the base layer forwards
  `::AcquisitionInfo` to it (`regularization.jl`). Ten forwarding methods and `_box_affected_dims` deleted.
  `EdgePreservingRoughness2D/3D`, `SecondOrderTotalVariation2D/3D` and `TotalGeneralizedVariation2D`, which
  had only the `::AcquisitionInfo` method, now work through the no-acquisition entry point as well.
- `identity_operator(x)` — replaced 7 copies of the `Eye` / `NamedDimsOp{names,names}(Eye(parent(x)))` pair.
- `_collapse_direction_axes(op, x, n)` — replaced 3 copies of the unwrap/reshape/rewrap idiom (TV,
  second-order TV, TGV).
- `_check_dim_spec(dim, name; allow_nothing)` — replaced `_check_lowrank_dims` and 3 inlined copies.
- `dims_of(x)` — replaced 6 copies of `x isa NamedDimsArray ? dimnames(x) : (1:ndims(x))`.
- `materialize_all(regs, x)` — shared by `build_model_with_variables` and `Component`'s fold.
- `execute_two_phase` — shared skeleton of `execute_regularized` and `execute_regularized_components`.
- `run_slices!` — one method built on `for_each_item!` instead of one per executor.
- `_resolve_scale` — shared by `_direct_reconstruct` and `_direct_reconstruct_components`.
- `get_component_x0s` — no longer re-validates what `check_x₀_components_size` already checked.
- `stack_image_slices` — dispatches on the first result instead of an `isa` test.
- ADMM defaults — single-key merge framework reduced to a membership test.
- `reduce(+, xs)` → `broadcast(+, xs...)` for the decomposition total.
