# benchmark/

Performance and accuracy measurement of MriReconstructionToolbox (MRT). Everything here reads the
same **case catalog** (`utils/`), so two measurements of the same case, by two checkouts of MRT or by
MRT and another toolkit, reconstruct byte-identical inputs.

| path | what it is |
|---|---|
| `utils/` | the case catalog (`BenchUtils`): phantoms, sampling, noise, real-data loaders, MRT method specs, timing, result store |
| `run.jl` | the MRT harness: every catalog case × every applicable method, one checkout of MRT at a time |
| `compare.jl` | compares two checkouts (refs) measured by `run.jl` |
| `comparison/` | the cross-toolkit suite: MRT against BART, SigPy, MRIReco and MIRT ([its README](comparison/README.md)) |
| `slurm/` | SLURM scripts for both; machine paths live in the untracked `slurm/site.env` |
| `results/` | harness results and SLURM logs (gitignored) |

`benchmark/` is a member of the root workspace (`Project.toml`), so it runs against this checkout of
MRT. `benchmark/comparison/` is a separate environment, since it carries Python and BART bridges.

## The case catalog

`utils/bench_utils.jl` defines `module BenchUtils`. `get_case(id)` returns a `BenchCase`: the
zero-filled k-space, mask or trajectory, sensitivity maps and reference image of one problem,
memoised per process. Synthetic data come only from GeometricMedicalPhantoms, FFTW and explicitly
seeded RNGs. MRT is used only to simulate non-Cartesian data (NFFT at `m = 8, σ = 2`), and to run
ESPIRiT on real data. MRT's own simulation code can therefore change between two measured refs
without changing either one's input.

| id | problem | sampling |
|---|---|---|
| `shepp_logan_2d_1ch_cartesian` | 128², 1 coil | variable-density random ky, 16 ACS lines, R ≈ 2.5 |
| `shepp_logan_2d_8ch_cartesian` | 128², 8 coils | variable-density ky + 16 ACS, R = 4 |
| `shepp_logan_2d_8ch_radial` | 128², 8 coils | golden-angle radial, 256 samples × 64 spokes, ramp DCF |
| `shepp_logan_multislice_8ch_cartesian` | 12 slices of the 128³ volume, maps from a 3D field | one ky mask for every slice, R = 4 |
| `shepp_logan_3d_8ch_cartesian` | 128³, 2 rings × 4 coils of true 3D maps | variable-density ky–kz, 24² calibration, R = 6 |
| `torso_cine_8ch_cartesian` | torso cine, 128², one cardiac cycle | per-frame random ky, 8 centre + 24 random lines |
| `torso_cine_8ch_radial` | the same series | golden-angle radial 256 × 34 per frame (one trajectory shared by every frame) |

All are `ComplexF32`, noise at `MRT_BENCH_SNR_DB` (30 dB) on unit-RMS k-space, with a fixed seed
per id. The 3D volume and the cines are **heavy**: one warm-up and one timed run, where every other
case takes the minimum of three runs.

`applicable_methods(case)` lists what is run on a case: the adjoint (or the DCF-weighted gridding
for radial data) and CG-SENSE for every case, then TV, L1-wavelet and TGV for the static
Cartesian ones (TV and wavelet for the volume, TV for radial), and global low rank, locally low
rank and temporal TV for the cines. Iteration counts, λ and the ADMM ρ are fixed
(`utils/mrt_methods.jl`, `CMP_OUTER` / `CMP_CG_ITERS` to override).

The GRAPPA rows of the comparison suite use `get_case(id; pattern = :regular)`, a regularly
undersampled variant of the 2D and multislice cases, since GRAPPA needs a fixed stride.

`MRT_BENCH_SMALL=1` shrinks every case: 32², 3 slices, 32³, 8 frames and 4 coils. It is meant for
smoke runs on a login node and for the `Benchmark case catalog smoke test` TestItem.

`utils/check_cases.jl` checks the catalog:
- the array layouts agree;
- the acceleration is within 5% of its design;
- no k-space sample outside the mask is nonzero;
- a noiseless, fully sampled Cartesian adjoint reproduces the phantom (NRMSE < 1e-5);
- a noiseless, Nyquist-sampled radial gridding with a Pipe-Menon DCF reproduces the phantom band-limited to the |k| ≤ ½ disc a radial trajectory covers (NRMSE < 0.15; a transposed or shifted convention scores above 1). The band limit is needed whatever the DCF: the corners outside the disc hold 14% of Shepp-Logan's energy and are never sampled.

It also prints a hash of every k-space, for comparing processes:

```sh
MRT_BENCH_SMALL=1 julia --project=benchmark benchmark/utils/check_cases.jl [--cases=radial] [--real]
```

### Real data

Every synthetic case has a real analogue (`real_*` ids, `REAL_CASES`), assembled by
`utils/real_data.jl` from [MRITestData.jl](https://github.com/hakkelt/MRITestData.jl). For each one
there is an open dataset and, where it is better suited, a gated fastMRI one. The open dataset is
preferred when it is as good; a real case falls back to the other when its first choice is
unavailable, and records which one it used (`data_source`).

| real id | first choice | fallback |
|---|---|---|
| `real_2d_1ch_cartesian` | fastMRI `singlecoil_val/file1000107` (native single channel, gated) | M4RAW, SENSE-combined to one channel |
| `real_2d_multichannel_cartesian` | M4RAW `multicoil_train/2022062402_T203`, middle slice | — |
| `real_2d_multichannel_radial` | fastMRI breast `fastMRI_breast_006_2`, central partition, 128 golden-angle spokes (gated) | USC speech spiral, one frame |
| `real_multislice_multichannel_cartesian` | M4RAW, 12 slices | fastMRI brain `file_brain_AXFLAIR_203_6000923` |
| `real_3d_multichannel_cartesian` | MRIDATA knee `52c2fd53-…`, a true 3D encode cropped to 128³ | — |
| `real_cine_multichannel_cartesian` | OCMR `fs_0001_1_5T` | — |
| `real_cine_multichannel_radial` | USC speech spiral real-time (3 of 13 arms per frame) | — |

Cartesian real data are fully sampled scans undersampled retrospectively with the analogue's
pattern. The reference is the SENSE combination of the full k-space, so `nrmse_gt` measures accuracy
here too. A real case uses the λ calibrated for its synthetic analogue. That λ transfers because
both k-spaces are unit-RMS normalised.

A gated source counts as available when its files are cached, or when signed fastMRI URLs are
registered in the environment's gitignored `LocalPreferences.toml`. `MRT_BENCH_REAL_PREFER=open` or
`gated` overrides the order. Prepared cases (ESPIRiT maps included, 3–4 minutes for the 128³ knee)
are cached in `MRT_BENCH_WORK_DIR`.

## The MRT harness: `run.jl`

```sh
julia --project=benchmark -t 8 benchmark/run.jl --threads=8 [--use-mkl] [--cases=shepp_logan_2d] [--methods=tv,cgsense] [--real]
```

Each (case, method) pair is timed with `BenchUtils.time_run`: one warm-up, then the minimum and
median of three runs, or of one for a heavy case. NRMSE is taken against the case's reference. Each
case writes one immutable JSON file to `results/runs/`. It records:
- threads, backend, host, pinned CPUs and node class (CPU model + SLURM partition);
- the git ref;
- tree hashes of MRT's `src/`, `ext/`, `Project.toml` and `deps/`, plus the NestedThreading checkout;
- the environment variables that change a measurement (`KMP_BLOCKTIME`, `OPENBLAS_THREAD_TIMEOUT`, …).

**Measure once.** A configuration already stored for the same *clean* code on the same node class
is skipped ("stored, skipping"). A baseline is therefore measured once, and later iterations of a
branch only pay for the branch. The code key hashes the package code, not the commit, so a commit
that only touches `benchmark/` keeps the stored results. `--remeasure` forces a rerun. A dirty
checkout is never stored as reusable.

**Another checkout.** `--mrt=PATH` measures the MRT checkout at `PATH`, a worktree of master for
example. The same harness re-runs itself in `PATH/benchmark`'s environment. `--dev=Pkg=PATH` points
that environment's manifest entry of `Pkg` at another checkout
(`--dev=NestedThreading=/path/to/checkout`). `--ref-name=NAME` labels the results (default: the
branch name).

```sh
git worktree add ../mrt-master master
julia --project=benchmark -t 8 benchmark/run.jl --threads=8 --mrt=../mrt-master --ref-name=master
julia --project=benchmark -t 8 benchmark/run.jl --threads=8 --ref-name=perf
julia --project=benchmark benchmark/compare.jl master perf [--threads=8] [--backend=openblas]
```

`compare.jl A B` matches the latest result of each side by (case, method, threads, backend,
environment variant). It prints the B/A ratio of the minimum time, each side's spread (median /
minimum) and ΔNRMSE. |ΔNRMSE| > 1e-3 is flagged as a change in *what* is computed, and a ratio outside
the larger spread plus 5% as a real speed change. Results from different node classes are warned
about, not compared silently.

## SLURM

Copy `slurm/site.env.example` to `slurm/site.env` and fill in the paths of this machine: the Julia
depot, the BART builds, the SigPy interpreter, the data directory and the partition names. No tracked
file names a machine path. Every script sources `slurm/common.sh`, which sets:
- the depot Julia binary (the login-node `julia` wrapper fails inside a SLURM cgroup);
- `TMPDIR=/dev/shm`, for BART's `.cfl` files;
- `KMP_BLOCKTIME=0`.

Submit through `slurm/submit.sh`, from the repository root. It uses the **test** partition (one hour)
unless given `--production`, which occupies a node for hours and needs agreement first.

| script | what it runs |
|---|---|
| `matrix.sh` | one exclusive node; each configuration pinned to its own NUMA domain with `numactl`, several at a time. `--suite=harness` runs `run.jl`, `--suite=comparison` runs `comparison/scripts/run_all.jl`. Matrix dimensions: `--matrix-threads=`, `--matrix-backends=`, `--matrix-refs=name:path,...` (harness), `--matrix-env=A=1+B=2,...` (recorded per row). Every other flag is passed through. |
| `calibrate.sh` | λ calibration (`comparison/scripts/calibrate_lambda.jl`), one case per array task |
| `report_efficiency.sh <jobid>` | CPU efficiency of a finished matrix job |

```sh
benchmark/slurm/submit.sh matrix.sh --suite=harness --matrix-refs=master:../mrt-master,perf:. --matrix-threads=1,8
benchmark/slurm/submit.sh --array=0-6 calibrate.sh
```

Logs go to `results/slurm/`.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `MRT_BENCH_SMALL` | `0` | shrink every case (smoke runs) |
| `MRT_BENCH_REAL_DATA` | `0` | include the real-data analogues in `run.jl` (same as `--real`) |
| `MRT_BENCH_SNR_DB` | `30` | noise level of the synthetic cases |
| `MRT_BENCH_CINE_FRAMES` | `30` | frames of the cine cases |
| `MRT_BENCH_DATA_DIR` | MRITestData's cache | where real data are downloaded |
| `MRT_BENCH_WORK_DIR` | `<data dir>/mrt_benchmark` | prepared real-data cases |
| `MRT_BENCH_REAL_PREFER` | — | `open` or `gated`: override the dataset order |
| `CMP_OUTER` / `CMP_CG_ITERS` | `20` / `10` | outer iterations / inner CG iterations and CG-SENSE iterations |

Each provider's own licence and citation terms apply to the real data; see
<https://hakkelt.github.io/MRITestData.jl/stable/legal/>.
