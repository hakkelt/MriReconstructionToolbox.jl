# benchmarking/

MRT-only performance measurement. No external frameworks — that is `comparison/`'s job.

This folder owns the numbers that describe MRT's own behaviour: how a reconstruction scales
with thread count, where intra-operator threading starts to pay, what the BLAS crossover is.
`comparison/` consumes the JSON produced here as the MRT baseline it diffs BART / SigPy /
MRIReco against, so the two folders never re-measure the same thing.

## Scripts

| script | what it measures |
|---|---|
| `scripts/recon_bench.jl` | end-to-end `reconstruct` timings for the standard method set (TV, L1-wavelet, TGV, CG-SENSE, low-rank, temporal TV, GRAPPA), 1 vs N threads, OpenBLAS vs MKL. Writes `results/mrt_<backend>_<n>threads.json`. |
| `scripts/threading_sweep.jl` | the BLAS-1 (CG-shaped) and BLAS-3 (SVD-shaped) work-item sweeps behind `SERIAL_BLAS_THRESHOLD_BYTES` and `maybe_disable_undecomposed_threading`. |
| `scripts/probe.jl` | what the process sees of its own core budget (`Cpus_allowed_list`, `jl_effective_threads`, the `LinearAlgebra` default, SLURM env). |

## Running

```sh
julia --project=benchmarking -t 8 benchmarking/scripts/recon_bench.jl --threads=8            # OpenBLAS
julia --project=benchmarking -t 8 benchmarking/scripts/recon_bench.jl --threads=8 --use-mkl  # MKL

julia --project=benchmarking -t 8 benchmarking/scripts/threading_sweep.jl mkl 8 cg
julia --project=benchmarking -t 8 benchmarking/scripts/threading_sweep.jl mkl 8 svd
```

## Real scanner data

`src/RealData.jl` (shared with `comparison/`) pulls real fully-sampled Cartesian k-space via
[`MRITestData.jl`](https://github.com/hakkelt/MRITestData.jl) — not yet registered, wired in as a
`[sources]` url. `load_real_case()` downloads (cached on first use) the smallest matching dataset,
assembles its middle slice into `(:kx, :ky, :coil)` k-space, and returns ESPIRiT sensitivity maps
plus an RSS reference — the same shape `generate_multicoil_brain` produces.

Opt in with an environment variable; `recon_bench.jl` / `benchmark/benchmarks.jl` /
`comparison/scripts/run_benchmarks.jl` then append a **Real Data** block (CG-SENSE + undersampled
TV / L1-wavelet):

```sh
MRT_BENCH_REAL_DATA=1 julia --project=benchmarking -t 8 benchmarking/scripts/recon_bench.jl --threads=8
```

| variable | default | meaning |
|---|---|---|
| `MRT_BENCH_REAL_DATA` | `0` | `1` enables the real-data rows |
| `MRT_BENCH_REAL_SOURCE` | `M4RAW` | `M4RAW` (0.3 T brain, 4ch, ~12 MB) · `MRIDATA` (mridata.org knee/brain, ~1 GB) · `FASTMRI` (needs the data-use form) |
| `MRT_BENCH_REAL_FILTER` | — | keep only entries whose id contains this (e.g. `T2`, `knee`) |

Each provider's own licence and citation terms apply — see
<https://hakkelt.github.io/MRITestData.jl/stable/legal/>.

## Results

`results/*.json` is **gitignored** — it is a local measurement, hardware- and load-dependent,
and regenerated on demand. Only `comparison/results/*.json` (MRT *plus* BART / SigPy / MRIReco,
for the documentation's comparison page) is committed.

## Cluster: full re-verification

`scripts/slurm_full_matrix.sh` runs the 1T/8T × OpenBLAS/MKL matrix on the `test` node, one
exclusive job per phase so nothing contends:

```sh
b=$(sbatch --parsable benchmarking/scripts/slurm_full_matrix.sh baseline)
o=$(sbatch --parsable --dependency=afterany:$b benchmarking/scripts/slurm_full_matrix.sh compare-openblas)
sbatch          --dependency=afterany:$o benchmarking/scripts/slurm_full_matrix.sh compare-mkl
```

`baseline` writes `results/mrt_*.json`; the `compare-*` phases read those and write
`comparison/results/benchmark_*.json`. `scripts/run_slurm.sh` is the baseline-only shortcut.

## Baseline for `comparison/`

`comparison/scripts/run_benchmarks.jl` reads `benchmarking/results/mrt_<backend>_<n>threads.json`
when it exists (`time_mrt`) and takes those times straight as the MRT column instead of
re-timing (the reconstruction still runs once, untimed, for the cross-framework NRMSE checks).
Regenerate the baseline here first, then run the comparison suite.
