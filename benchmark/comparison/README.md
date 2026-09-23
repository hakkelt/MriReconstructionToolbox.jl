# benchmark/comparison/

Cross-toolkit MRI reconstruction benchmark: MRT vs BART / SigPy / MRIReco / MIRT, on matched
problems at matched effort or matched accuracy. `benchmark/hpc/` is MRT-only; this folder never
re-measures what that one already does — `_setup.jl`'s `time_mrt` always times MRT inline.

## Sections

Each `scripts/run_<section>.jl` is a standalone, runnable Julia script (`include`s `_setup.jl` and
`_toolkits.jl` first) that reconstructs one problem across every toolkit and records the rows:

| section | what it covers |
|---|---|
| `base` | 1-coil and multi-coil Cartesian adjoint (no iterative solve) |
| `noncart` | radial DCF adjoint / gridding, MRT default vs MRIReco's NFFT operating point |
| `cgsense` | CG-SENSE cost-per-iteration on fully-sampled data |
| `sparsity` | 2×-undersampled TV / L1-wavelet / TGV, matched effort |
| `dynamic` | 2D+t global/locally low-rank, temporal TV, matched effort |
| `kspace` | GRAPPA (MRT-only — no cross-toolkit row exists, see the script's header) |
| `real` | real scanner data (M4RAW, knee 3D, OCMR cine) — always runs, no opt-in needed |
| `accuracy_race` | time-to-target-NRMSE per toolkit, the fair comparison (see the script's header for why the other sections' fixed-iteration-count numbers are not directly comparable across toolkits) |

`scripts/run_all.jl` runs every section as its own subprocess (they each define top-level `const`s,
so they cannot share a process) and is the normal entry point:

```sh
julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_all.jl --threads=N [--use-mkl]
```

## Filtering a rerun

Three independent filters, all stackable, all forwarded from `run_all.jl` to each section
subprocess:

| flag | narrows to |
|---|---|
| `--sections=sparsity,dynamic` | specific sections instead of all 8 |
| `--cases=low-rank` | a case-insensitive substring match against `category` OR `method` — e.g. `low-rank` matches every low-rank case across `dynamic` and `accuracy_race` regardless of the exact suffix on its method string; `sparsity` alone matches a whole section the same way `--sections=` would |
| `--frameworks=BART` | gates only the *competitor* toolkits (SigPy/BART/MRIReco/MIRT); MRT's own solve always runs — it is the reference every other framework's `nrmse_mrt` is computed against, and it is cheap next to whichever toolkit is under suspicion |

```sh
# re-verify one suspect BART timing without paying for the other three toolkits or 7 other sections
julia --project=benchmark/comparison -t 16 --use-mkl benchmark/comparison/scripts/run_all.jl \
    --threads=16 --use-mkl --sections=dynamic --cases=low-rank --frameworks=BART
```

Every section checks `should_run` / `should_run_framework` (`_setup.jl`) *before* paying for a
solve, so a narrow filter is actually cheap, not just a smaller printout.

## Results storage (`ResultsStore.jl`)

Every recorded run is its own immutable JSON file under `results/runs/`, never overwritten —
concurrent writers (different SLURM nodes, or a login-node smoke test run alongside a cluster job)
cannot collide with each other by construction, no locking needed, on any filesystem. `flush_results!`
(inside every section script) writes one such file after *each case*, not just once at the end, so
a crash partway through a long section keeps whatever already finished. `source` (`"slurm"` on this
cluster's compute nodes, `"other"` everywhere else — see `ResultsStore.source_tag`) is recorded on
every run, so a login-node measurement stays visibly distinct from a cluster one instead of silently
looking the same.

`results/runs/` is gitignored working data. Nothing merges it automatically — reading it back is a
query, done fresh each time:

- **`query_results.jl`** — prints the latest row per (backend, threads, category, method,
  framework), preferring `source = "slurm"`. Filter with `--backend=`, `--threads=`, `--category=`,
  `--method=`, `--source=`.
  ```sh
  julia --project=benchmark/comparison benchmark/comparison/scripts/query_results.jl --source=slurm --category="Sparsity"
  ```
- **`export_snapshot.jl`** — writes `results/benchmark_<backend>_<n>threads.json` (the committed
  snapshot the documentation's comparison page reads), one per (backend, threads) pair found, from
  the latest `source = "slurm"` row per case. Run this after any cluster rerun that should update the
  committed numbers:
  ```sh
  julia --project=benchmark/comparison benchmark/comparison/scripts/export_snapshot.jl
  ```
- **`migrate_to_store.jl`** — one-shot converter from the old JSON layout (per-section fragments,
  merged files) into `results/runs/`. Only needed once per pre-existing result file; nothing to run
  in normal use.

## Calibration

`scripts/calibrate_lambda.jl` fits each toolkit's own λ to a common target NRMSE
(`results/lambda_calibration.json`, committed) so every section compares toolkits at matched
accuracy rather than at a nominally-equal but differently-scaled λ. Rerun it before a section whose
regularization changed.

## Real scanner data

See `benchmark/hpc/README.md`'s "Real scanner data" section — `src/RealData.jl` is shared between
the two folders.
