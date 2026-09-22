# benchmark/hpc/

MRT-only performance measurement. No external frameworks — that is `benchmark/comparison/`'s job.

This folder owns the numbers that describe MRT's own behaviour: how a reconstruction scales
with thread count, where intra-operator threading starts to pay, what the BLAS crossover is.

## Scripts

| script | what it measures |
|---|---|
| `scripts/threading_sweep.jl` | the BLAS-1 (CG-shaped) and BLAS-3 (SVD-shaped) work-item sweeps behind `SERIAL_BLAS_THRESHOLD_BYTES` and `maybe_disable_unsplit_threading`. |
| `scripts/serial_blas_threshold_sweep.jl` | re-fits `SERIAL_BLAS_THRESHOLD_BYTES` against real reconstruction solves rather than `threading_sweep.jl`'s synthetic CG reproducer; interleaved threaded/serial (A/B/A/B…) runs since whole-solve timing here swings ±30-60% run to run. |
| `scripts/probe.jl` | what the process sees of its own core budget (`Cpus_allowed_list`, `jl_effective_threads`, the `LinearAlgebra` default, SLURM env). `benchmark/comparison/scripts/_setup.jl`'s `check_environment()` runs the same checks automatically and fails the job on a mismatch instead of just printing one; run this by hand for the interactive/diagnostic version. |

MRT-only end-to-end timings are no longer produced here — `recon_bench.jl` was removed;
`benchmark/comparison/scripts/run_all.jl` now always times MRT inline (see "Baseline for
`benchmark/comparison/`" below).

## Running

```sh
julia --project=benchmark/hpc -t 8 benchmark/hpc/scripts/threading_sweep.jl mkl 8 cg
julia --project=benchmark/hpc -t 8 benchmark/hpc/scripts/threading_sweep.jl mkl 8 svd

julia --project=benchmark/hpc -t 8 benchmark/hpc/scripts/serial_blas_threshold_sweep.jl openblas --rounds=3 --reps=3
```

## Real scanner data

`src/RealData.jl` (shared with `benchmark/comparison/`) pulls real fully-sampled Cartesian k-space via
[`MRITestData.jl`](https://github.com/hakkelt/MRITestData.jl) — not yet registered, wired in as a
`[sources]` url. `load_real_case()` downloads (cached on first use) one pinned dataset —
`RealData.PINNED_DATASET`, currently **M4RAW `multicoil_train/2022062402_T203`** (0.3 T brain,
4-channel, 256², 11.7 MB) — assembles its middle slice into `(:kx, :ky, :coil)` k-space, and
returns ESPIRiT sensitivity maps plus an RSS reference — the same shape
`generate_multicoil_brain` produces. `combine_coils = true` folds that to one virtual channel
(SENSE-optimal combine, `smaps ≡ 1`) — a real-anatomy single-coil / compressed-sensing point,
since no catalog source ships native single-channel data (fastMRI's `singlecoil_*` is behind a
registration wall and is not a `MRITestData` source). The comment block in `RealData.jl` is a
full per-data-type catalog survey with the open pick first (M4RAW / MRIDATA / OCMR for
multi-coil Cartesian, USC_SPEECH for spiral, **OCMR `us_*`** as the open GRAPPA-undersampled
alternative to the Synapse-gated CMRxRecon-300).

Opt in with an environment variable; `benchmark/ci/benchmarks.jl` / `benchmark/comparison/scripts/run_real.jl`
then append a **Real Data** block plus a **Real Data 1ch** block, each with CG-SENSE + undersampled
TV / L1-wavelet:

```sh
MRT_BENCH_REAL_DATA=1 julia --project=benchmark/ci -t 8 benchmark/ci/benchmarks.jl
```

| variable | default | meaning |
|---|---|---|
| `MRT_BENCH_REAL_DATA` | `0` | `1` enables the real-data rows |
| `MRT_BENCH_REAL_SOURCE` | `M4RAW` | `M4RAW` (0.3 T brain, 4ch, ~12 MB) · `MRIDATA` (mridata.org knee/brain, 8–15ch, ~1 GB) · `OCMR` (1.5/3 T cardiac cine, ~200 MB) |
| `MRT_BENCH_REAL_ID` | pinned id | a specific catalog id, or `auto` for the smallest matching entry |
| `MRT_BENCH_REAL_FILTER` | — | with `MRT_BENCH_REAL_ID=auto`, keep only entries whose id contains this (e.g. `T2`, `knee`) |

Each provider's own licence and citation terms apply — see
<https://hakkelt.github.io/MRITestData.jl/stable/legal/>.

## Results

`results/*.json` is **gitignored** — it is a local measurement, hardware- and load-dependent,
and regenerated on demand. Only `benchmark/comparison/results/*.json` (MRT *plus* BART / SigPy / MRIReco,
for the documentation's comparison page) is committed.

## Cluster: full re-verification

`scripts/slurm_full_matrix.sh` runs the `compare-*` phases (`run_all.jl`, MRT timed inline) on the
`test` node, one exclusive job per phase so nothing contends:

```sh
sbatch benchmark/hpc/scripts/slurm_full_matrix.sh compare-openblas
sbatch benchmark/hpc/scripts/slurm_full_matrix.sh compare-mkl
```
