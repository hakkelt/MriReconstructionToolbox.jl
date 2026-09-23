# benchmark/comparison/

Cross-toolkit MRI reconstruction benchmark: MRT vs BART / SigPy / MRIReco / MIRT, on the cases of
the benchmark case catalog (`benchmark/utils/`, see [`benchmark/README.md`](../README.md)), at
matched effort or matched accuracy. Comparing one checkout of MRT against another is the MRT
harness's job (`benchmark/run.jl`), not this suite's.

## Sections

Each `scripts/run_<section>.jl` is a standalone, runnable Julia script (`include`s `_setup.jl`,
`_toolkits.jl` and `_methods.jl`). It loops over the catalog cases its method family applies to and
calls `run_method_rows!` per (case, method): MRT's row is timed with the same `mrt_reconstructor`
call the MRT harness times, then every competitor that `supports` the pair. No section prepares
data of its own. Every toolkit gets the same prepared case, rearranged to its layout by
`_toolkits.jl`.

| section | cases × methods |
|---|---|
| `base` | adjoint on the Cartesian cases |
| `noncart` | DCF-weighted gridding on the radial cases, plus MRT at MRIReco's NFFT operating point |
| `cgsense` | CG-SENSE on every multichannel case |
| `sparsity` | TV / L1-wavelet / TGV on the static cases (TV only for radial), matched effort |
| `dynamic` | global / locally low rank, temporal TV on both cine cases, matched effort |
| `kspace` | GRAPPA on the regularly undersampled variant of the 2D and multislice cases (MRT only — no cross-toolkit row exists, see the script's header) |
| `accuracy_race` | time-to-target-NRMSE per toolkit, the fair comparison (see the script's header for why the other sections' fixed-iteration-count numbers are not directly comparable across toolkits) |

`--data=synthetic` (default), `real` or `all` picks which catalog cases the sections iterate; the
real-data analogues use the λ of their synthetic case. Unsupported (toolkit, case, method)
combinations are skipped; the table in `supports` (`_toolkits.jl`) lists what each toolkit covers.

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
| `--sections=sparsity,dynamic` | specific sections instead of all seven |
| `--data=all` | which catalog cases: `synthetic` (default), `real` or `all` |
| `--cases=shepp_logan_2d,low-rank` | case-insensitive substrings matched against a catalog case id, a section, or a method label — `shepp_logan_2d` runs the three 2D Shepp-Logan cases, `low-rank` every low-rank row across `dynamic` and `accuracy_race` |
| `--frameworks=BART` | gates only the *competitor* toolkits (SigPy/BART/MRIReco/MIRT); MRT's own solve always runs — it is the reference every other framework's `nrmse_mrt` is computed against, and it is cheap next to whichever toolkit is under suspicion |

```sh
# re-verify one suspect BART timing without paying for the other three toolkits or 7 other sections
julia --project=benchmark/comparison -t 16 --use-mkl benchmark/comparison/scripts/run_all.jl \
    --threads=16 --use-mkl --sections=dynamic --cases=low-rank --frameworks=BART
```

Every section checks `should_run` / `should_run_framework` (`_setup.jl`) *before* paying for a
solve, so a narrow filter is actually cheap, not just a smaller printout.

`MRT_BENCH_SMALL=1` runs every section on the shrunken catalog in a few minutes, for a smoke test
on a login node:

```sh
MRT_BENCH_SMALL=1 julia --project=benchmark/comparison -t 4 benchmark/comparison/scripts/run_all.jl --threads=4 --frameworks=none
```

## Results storage (`benchmark/utils/results_store.jl`)

Every recorded run is its own immutable JSON file under `results/runs/`, never overwritten —
concurrent writers (different SLURM nodes, or a login-node smoke test run alongside a cluster job)
cannot collide with each other by construction, no locking needed, on any filesystem. `flush_results!`
(inside every section script) writes one such file after *each case*, not just once at the end, so
a crash partway through a long section keeps whatever already finished. `source` (`"slurm"` on this
cluster's compute nodes, `"other"` everywhere else — see `ResultsStore.source_tag`) is recorded on
every run, so a login-node measurement stays visibly distinct from a cluster one instead of silently
looking the same. Every row carries its `case_id`, `data_source` and `schema_version` (2); rows
written before the case catalog (schema 1) are skipped when reading, since their problems no longer
exist.

`results/runs/` is gitignored working data. Nothing merges it automatically — reading it back is a
query, done fresh each time:

- **`query_results.jl`** — prints the latest row per (backend, threads, case, category, method,
  framework), preferring `source = "slurm"`. Filter with `--backend=`, `--threads=`, `--case=`,
  `--category=`, `--method=`, `--source=`.
  ```sh
  julia --project=benchmark/comparison benchmark/comparison/scripts/query_results.jl --source=slurm --case=shepp_logan_2d
  ```
- **`export_snapshot.jl`** — writes `results/benchmark_<backend>_<n>threads.json`, the committed
  snapshot of a full cluster run, one per (backend, threads) pair found, from the latest
  `source = "slurm"` row per case. Run this after any cluster rerun that should update the
  committed numbers:
  ```sh
  julia --project=benchmark/comparison benchmark/comparison/scripts/export_snapshot.jl
  ```

## Calibration

`scripts/calibrate_lambda.jl` fits each toolkit's own λ per case. It sweeps a log grid (8 points, 6
for the heavy cases) at 30 outer iterations. MRT's best NRMSE is the target, and every other toolkit
gets the λ whose NRMSE is closest to it. So every section compares toolkits at matched accuracy
rather than at a nominally equal but differently scaled λ.

Results go to `results/lambda/<case id>.json`, together with `race_target`: the worst toolkit's best
NRMSE × 1.10, the target `run_accuracy_race.jl` races to. `load_lambda` falls back from a case to its
synthetic analogue, then to the pre-catalog `results/lambda_calibration.json`, then to the default
in `benchmark/utils/mrt_methods.jl`. Under `MRT_BENCH_SMALL=1` the files go to `results/lambda_small/`
(gitignored) instead. Rerun calibration for a case whose problem or regularization changed. It
takes one SLURM array task per case:

```sh
benchmark/slurm/submit.sh --array=0-6 calibrate.sh
```

## Real scanner data

The real-data analogues of the catalog cases, where they come from and how their references are
built, are described in [`benchmark/README.md`](../README.md#real-data).
