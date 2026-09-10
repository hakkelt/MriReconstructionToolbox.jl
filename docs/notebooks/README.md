# MriReconstructionToolbox notebooks

Twelve Jupyter notebooks (Julia kernel) that walk through the package feature by feature, on
synthetic phantoms and on real scanner data. Only the `docs/notebooks/src/*.jl` scripts are
tracked in git — see [Editing workflow](#editing-workflow-jupytext) for why, and how the
`.ipynb` files are regenerated from them.

| Notebook | Topic |
|---|---|
| `01_getting_started` | A complete reconstruction in twenty lines: phantom → k-space → direct → compressed sensing |
| `02_acquisition_info` | `AcquisitionInfo`: named dimensions, sensitivity maps, sampling patterns, FFT-shift conventions, validation, Cartesian vs. non-Cartesian |
| `03_simulation` | Phantoms, coil sensitivities, every sampling-pattern generator, `simulate_acquisition`, noise, dynamic series |
| `04_reconstruction_methods` | Coil combination, partial Fourier (Homodyne, phase-constrained, POCS), GRAPPA, SPIRiT |
| `05_regularization` | Every spatial regularizer: ℓ₂/ℓ₁, wavelets, contourlets, TV, second-order TV, TGV, Huber, hard thresholding, plug-and-play, joint sparsity, reference priors, constraints |
| `06_algorithms_and_configuration` | CG/CGNR, ISTA/FISTA, ADMM, Douglas–Rachford; `maxit`/`tol`, verbosity, `ReconstructionConfig`, scaling, warm starts, operator-norm options, task splitting |
| `07_dynamic_and_decomposition` | Temporal and low-rank regularizers, image decomposition (L+S), infimal-convolution TV |
| `08_non_cartesian` | Radial and spiral trajectories, NFFT encoding, Pipe–Menon and Voronoi density compensation, gradient-delay correction, gridding accuracy vs. speed |
| `09_real_data_cartesian` | Real 0.3 T brain data (M4Raw) end to end: assembly, prewhitening, coil compression, ESPIRiT, retrospective undersampling, CS and parallel imaging, pseudo-replica noise analysis |
| `10_real_data_dynamic` | Real 1.5 T cardiac cine (OCMR): temporal, low-rank and L+S reconstruction, temporal profiles |
| `11_advanced_reconstruction` | Data-fidelity choices, `TemporalBasis` and `KSpaceToImage` signal models, calibrationless structured low-rank (SAKE / LORAKS-C) |
| `12_low_level_interface` | Operators by hand, `build_model`, `StructuredOptimization` problems, proximal operators, writing a regularizer of your own |

Notebooks 1–8, 11 and 12 need nothing but the environment in this directory. Notebooks 9 and 10
download real datasets (~12 MB and ~200 MB) on first run and cache them.

## Setup

From the repository root:

```julia
using Pkg
Pkg.activate("docs/notebooks")
Pkg.instantiate()                  # resolves the dev-pathed forks under deps/
Pkg.build("IJulia")                # installs the Julia kernel for Jupyter
```

Then start Jupyter — `IJulia.notebook()` from Julia installs a private Jupyter if you have none:

```julia
using IJulia
notebook(dir = "docs/notebooks")
```

or, with an existing Jupyter installation:

```sh
cd docs/notebooks && jupyter lab
```

Select the Julia kernel matching the version you instantiated with (Julia 1.12 or newer; the
notebooks' stored kernel name is `julia-1.12`).

### Threads

Several notebooks (7 in particular) measure the effect of multi-threading. Start the kernel with
more than one thread to see it:

```sh
JULIA_NUM_THREADS=8 jupyter lab
```

Set `KMP_BLOCKTIME=0` in the environment as well if you use MKL — it cannot be set from inside
Julia.

## Runtime

Notebooks 1–8, 11 and 12 are written to run end to end on a laptop: the phantoms are 128² or
smaller, the dynamic series is 64² × 16 frames, and the iteration counts are chosen for a few
seconds per reconstruction. Measured with `export.jl` on 2026-09-09 (Julia 1.12.7,
`JULIA_NUM_THREADS=4`, shared login node), before the renumbering below moved and split some of
this content — **stale, and marked `PENDING` where the split makes an old number no longer
apply; refilled at the next `export.jl` run on a dedicated node (Phase 4)**:

| Notebook | Wall time | Notes |
|---|---|---|
| `01_getting_started` | 3m 23s | |
| `02_acquisition_info` | 0m 40s | |
| `03_simulation` | 1m 55s | |
| `04_reconstruction_methods` | PENDING | was `05_reconstruction_methods`, minus the data-fidelity/signal-model sections now in `11` |
| `05_regularization` | 5m 44s | was `04_regularization`; every regularizer, several reconstructions each |
| `06_algorithms_and_configuration` | 4m 53s | |
| `07_dynamic_and_decomposition` | 5m 21s | |
| `08_non_cartesian` | 1m 57s | plus NFFT precompilation on the first call |
| `09_real_data_cartesian` | 4m 20s | was `10_real_data_cartesian`; after the ~12 MB download |
| `10_real_data_dynamic` | 38m 47s | was `11_real_data_dynamic`; after the ~200 MB download; the λ sweep is ~25 min of it |
| `11_advanced_reconstruction` | PENDING | new — data fidelity, signal models and structured low-rank k-space, split out of the old `05_reconstruction_methods` |
| `12_low_level_interface` | 2m 18s | was `09_low_level_interface` |

Each number is one measurement and includes roughly a minute of first-call compilation, so treat
them as an order of magnitude rather than a benchmark. `10_real_data_dynamic` is deliberately the
expensive one: it sweeps λ for seven methods across two sampling patterns so that no method is
shown at a setting somebody guessed. Drop entries from its `sweeps` tuple if you want it faster.

The regenerated `.ipynb` files carry no stored outputs; every code cell has been executed against
this environment, so "Run All" should complete without errors.

## Shared preamble

Every notebook's first code cell starts with:

```julia
include("NotebookUtils.jl")
using .NotebookUtils
```

`docs/notebooks/NotebookUtils.jl` is the single place that:

- fixes MIRTjim's `jim` vertical-flip default (`jim(:yflip, false)`) so the phantoms render right
  side up in every notebook — do not pass `yflip` at individual call sites, and do not flip the
  underlying arrays;
- notes that plot titles and axis labels use plain ASCII (`title = "Ax"`, not `title = "𝒜x"`),
  while script letters stay in markdown prose and as Julia variable names;
- ships `nrmse(x̂, x)`, `side_by_side(images...; titles, clim)` (shared color scale across panels)
  and `difference_image(x̂, x; scale)` (its own color scale) so notebooks stop redefining these ad
  hoc.

Do not re-fix the flip or the font warning locally — extend `NotebookUtils.jl` instead if a new
case doesn't fit the existing helpers.

## Editing workflow (jupytext)

**Only `docs/notebooks/src/*.jl` is tracked in git.** `docs/notebooks/*.ipynb` is generated from
it via [jupytext](https://jupytext.readthedocs.io) (percent format, one script per notebook, same
base name) and gitignored — raw `.ipynb` JSON diffs are unreviewable and unmergeable across
parallel worktrees, and a notebook that was executed once and re-committed with stored outputs is
exactly the failure mode this avoids. The `.jl` script is the only source of truth; there is
nothing to keep in sync because there is no second copy to drift.

Install jupytext once (already available in this environment via `pip install --user jupytext`):

```sh
python3 -m pip install --user jupytext
```

Workflow:

1. Edit `docs/notebooks/src/NN_name.jl` (a normal Julia file with `# %%` / `# %% [markdown]` cell
   markers — readable and runnable top-to-bottom outside Jupyter too).
2. Generate (or regenerate) the `.ipynb` from it, to open in Jupyter/JupyterLab or to sanity-check
   a render:
   ```sh
   python3 -m jupytext --to ipynb docs/notebooks/src/NN_name.jl
   ```
   or regenerate every notebook at once:
   ```sh
   python3 -m jupytext --to ipynb docs/notebooks/src/*.jl
   ```
   This does NOT execute the notebook — cell outputs are never written by `--to ipynb`.
   `export.jl` (below) does this step for you before it runs nbconvert.
3. If you ran the notebook interactively in Jupyter, its `.ipynb` now carries outputs — that file
   is gitignored, so there is nothing to strip or commit; just leave it, or delete it, before your
   next `--to ipynb`/`export.jl` run regenerates it clean.
4. Commit only the `.jl` script.

A new notebook needs no pairing step: write `docs/notebooks/src/NN_name.jl` directly (copy the
jupytext header comment block from an existing script) and generate its `.ipynb` as in step 2.

## Exporting to HTML

`docs/notebooks/export.jl` first regenerates every `.ipynb` from `docs/notebooks/src/*.jl` (via
`python3 -m jupytext --to ipynb`), then executes one or all of them and renders to HTML (via
`python3 -m nbconvert --to html --execute` against the `julia-1.12` kernel), writing to the
gitignored `docs/notebooks/build/` and leaving the regenerated `.ipynb` files output-free:

```sh
# one notebook, by number or by name fragment
julia --project=docs/notebooks docs/notebooks/export.jl 05
julia --project=docs/notebooks docs/notebooks/export.jl regularization

# all twelve, with a custom per-notebook timeout (seconds)
julia --project=docs/notebooks docs/notebooks/export.jl all --timeout=900
```

It prints a pass/fail summary and exits non-zero if any notebook failed; requires
`python3 -m nbconvert` (`pip install --user nbconvert`) and `python3 -m jupytext` on `PATH`, and
the `julia-1.12` Jupyter kernel (`Pkg.build("IJulia")`, see Setup above).

The rendered HTML is self-contained (figures are embedded), so a file from `build/` is what to
send someone who should see the notebook with its output without running Julia.

If notebook 9 or 10 fails at its first `AcquisitionInfo(raw)` call with *"is3D must be provided
when non-NamedDimsArray k-space is used"*, this environment's `Manifest.toml` predates the
`MriReconstructionToolboxMRIBaseExt` package extension and is silently not loading it. Run
`julia --project=docs/notebooks -e 'using Pkg; Pkg.resolve()'` and re-run.

## Data licensing

The real-data notebooks use [MRITestData.jl](https://github.com/hakkelt/MRITestData.jl) to fetch
public datasets. The package is MIT-licensed; **the datasets are not** — each provider has its own
terms and citation requirements:

- **M4Raw** (notebook 9) — CC-BY. Cite Lyu et al., *M4Raw: A multi-contrast, multi-repetition,
  multi-channel MRI k-space dataset for low-field MRI research*, Scientific Data 10, 264 (2023).
- **OCMR** (notebook 10) — OCMR data-use terms. Cite Chen et al., *OCMR (v1.0) — Open-Access
  Multi-Coil k-Space Dataset for Cardiovascular Magnetic Resonance Imaging*, arXiv:2008.03410
  (2020).

On first use `MRITestData` asks where downloads should go; the notebooks default to the package's
scratch cache.
