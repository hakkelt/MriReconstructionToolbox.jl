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
| `sparsity` | isotropic TV, anisotropic TV, L1-wavelet, TGV on the static cases (the two TVs only for radial), matched effort |
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

A row that runs a fixed-penalty ADMM also sweeps ρ, over the decades `RHO_DECADES` (default
`-2,-1,0,1,2`) around the toolkit's default: `admm_rho(c)` for MRT, relative to `‖𝒜‖²`, and
`CMP_RHO` for the others, absolute in their own operator scaling. Each toolkit keeps the ρ at which
it reaches its best NRMSE (the `rho` table, read back by `load_rho`), and its λ is picked on that
ρ's curve; a best ρ at the grid's edge is logged. Without a calibrated ρ, rows fall back to those
defaults.

`--frameworks=mrt,bart,...` recalibrates only the named toolkits and `--methods=tv,...` only the
named methods. The curves of the other toolkits are read back from the case's file, so the target,
the picks and `race_target` are always recomputed over every toolkit calibrated so far, and the
file is merged under a lock, so several processes can calibrate one case at once.

A toolkit's optimum can lie outside the shared grid (BART's and SigPy's radial TV λ lies above
it), so each axis grows past an edge holding the best point, up to `MAX_GRID_EXTENSIONS` (4) steps.
`--resume` reuses the points already stored for the toolkits being calibrated and measures only
the missing ones, which makes widening a finished calibration cheap. Every point is written to the
case's file as soon as it is measured, so a job cut off by its time limit loses nothing, and under
`--resume` a toolkit's new points are added to its stored ones. A slow toolkit can therefore be
split over several processes, one ρ decade each (`RHO_DECADES=-1` and so on), followed by one
`--resume` run over the full grid that measures only the extensions an edge optimum still needs.
SigPy's 3D TV (about 20 minutes a point) and radial cine rows (about 7) were calibrated that way.

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
