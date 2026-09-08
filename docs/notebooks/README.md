# MriReconstructionToolbox notebooks

Eleven Jupyter notebooks (Julia kernel) that walk through the package feature by feature, on
synthetic phantoms and on real scanner data.

| Notebook | Topic |
|---|---|
| `01_getting_started.ipynb` | A complete reconstruction in twenty lines: phantom → k-space → direct → compressed sensing |
| `02_acquisition_info.ipynb` | `AcquisitionInfo`: named dimensions, sensitivity maps, sampling patterns, FFT-shift conventions, validation, Cartesian vs. non-Cartesian |
| `03_simulation.ipynb` | Phantoms, coil sensitivities, every sampling-pattern generator, `simulate_acquisition`, noise, dynamic series |
| `04_regularization.ipynb` | Every spatial regularizer: ℓ₂/ℓ₁, wavelets, contourlets, TV, second-order TV, TGV, Huber, hard thresholding, plug-and-play, joint sparsity, reference priors, constraints |
| `05_reconstruction_methods.ipynb` | Coil combination, partial Fourier (Homodyne, phase-constrained, POCS), GRAPPA, SPIRiT, data-fidelity choices, `TemporalBasis` and `KSpaceToImage` signal models |
| `06_algorithms_and_configuration.ipynb` | CG/CGNR, ISTA/FISTA, ADMM, Douglas–Rachford; `maxit`/`tol`, verbosity, `ReconstructionConfig`, scaling, warm starts, operator-norm options |
| `07_dynamic_and_decomposition.ipynb` | Temporal and low-rank regularizers, image decomposition (L+S), problem decomposition over batch dimensions, threading |
| `08_non_cartesian.ipynb` | Radial and spiral trajectories, NFFT encoding, Pipe–Menon and Voronoi density compensation, gradient-delay correction, gridding accuracy vs. speed |
| `09_low_level_interface.ipynb` | Operators by hand, `build_model`, `StructuredOptimization` problems, proximal operators, writing a regularizer of your own |
| `10_real_data_cartesian.ipynb` | Real 0.3 T brain data (M4Raw) end to end: assembly, prewhitening, coil compression, ESPIRiT, retrospective undersampling, CS and parallel imaging, pseudo-replica noise analysis |
| `11_real_data_dynamic.ipynb` | Real 1.5 T cardiac cine (OCMR): temporal, low-rank and L+S reconstruction, temporal profiles |

Notebooks 1–9 need nothing but the environment in this directory. Notebooks 10 and 11 download
real datasets (~12 MB and ~200 MB) on first run and cache them.

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

The notebooks are written to run end to end on a laptop: the phantoms are 128² or smaller, the
dynamic series is 64² × 16 frames, and the iteration counts are chosen for a few seconds per
reconstruction. A full run of one notebook takes roughly:

| Notebook | Approx. wall time |
|---|---|
| 01, 02, 03 | under a minute each |
| 04, 05, 06, 09 | 1–3 minutes |
| 07 | ~4 minutes |
| 08 | ~1 minute (plus NFFT precompilation on the first call) |
| 10 | ~2 minutes after the download |
| 11 | ~5 minutes after the download |

The notebooks are shipped without stored outputs; every code cell has been executed against this
environment, so "Run All" should complete without errors.

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
- provides `asciilabel(s)` to silence the repeated `GKS: glyph missing from current font: ...`
  warnings caused by script-style operator names (`𝒜`, `𝒫`, `𝒮`, `𝒲`, `𝒞`, ...) in plot
  titles/labels — wrap any such string in it, e.g. `jim(x; title = asciilabel("𝒜x"))`;
- ships `nrmse(x̂, x)`, `side_by_side(images...; titles, clim)` (shared color scale across panels)
  and `difference_image(x̂, x; scale)` (its own color scale) so notebooks stop redefining these ad
  hoc.

Do not re-fix the flip or the font warning locally — extend `NotebookUtils.jl` instead if a new
case doesn't fit the existing helpers.

## Editing workflow (jupytext)

The `.ipynb` files are PAIRED with plain-text Julia scripts under `docs/notebooks/src/*.jl`
(percent format, one script per notebook, same base name) via
[jupytext](https://jupytext.readthedocs.io). The scripts are the files to edit — raw `.ipynb` JSON
diffs are unreviewable and unmergeable across parallel worktrees; the paired `.jl` script is not.

Install jupytext once (already available in this environment via `pip install --user jupytext`):

```sh
python3 -m pip install --user jupytext
```

Workflow:

1. Edit `docs/notebooks/src/NN_name.jl` (a normal Julia file with `# %%` / `# %% [markdown]` cell
   markers — readable and runnable top-to-bottom outside Jupyter too).
2. Regenerate the paired `.ipynb` from it:
   ```sh
   python3 -m jupytext --sync docs/notebooks/src/NN_name.jl
   ```
   (`--sync` reads whichever side is newer; run it after editing either file, though the `.jl`
   script is the intended source of truth.) This does NOT execute the notebook — cell outputs are
   never written by `--sync`.
3. To actually run the notebook (e.g. to sanity-check it, or before an HTML export), open it in
   Jupyter/JupyterLab with the `julia-1.12` kernel, or use `export.jl` (below), then strip outputs
   again before committing — `export.jl` never writes outputs back into the source `.ipynb`, but a
   manual "Run All" in Jupyter will, so re-run `jupytext --sync` (or `Kernel > Restart & Clear
   Output`) before committing if you executed interactively.
4. Commit BOTH the `.ipynb` and the `.jl` script; they must stay in sync (CI/reviewers should treat
   a mismatch as a bug).

A new notebook is paired the same way: create the `.ipynb`, then
`python3 -m jupytext --set-formats ipynb,src//jl:percent NN_name.ipynb`.

## Exporting to HTML

`docs/notebooks/export.jl` executes one or all notebooks and renders them to HTML (via
`python3 -m nbconvert --to html --execute` against the `julia-1.12` kernel), writing to the
gitignored `docs/notebooks/build/` and leaving the source `.ipynb` files output-free:

```sh
# one notebook, by number or by name fragment
julia --project=docs/notebooks docs/notebooks/export.jl 05
julia --project=docs/notebooks docs/notebooks/export.jl regularization

# all eleven, with a custom per-notebook timeout (seconds)
julia --project=docs/notebooks docs/notebooks/export.jl all --timeout=900
```

It prints a pass/fail summary and exits non-zero if any notebook failed; requires
`python3 -m nbconvert` (`pip install --user nbconvert`) on `PATH` and the `julia-1.12` Jupyter
kernel (`Pkg.build("IJulia")`, see Setup above).

## Data licensing

The real-data notebooks use [MRITestData.jl](https://github.com/hakkelt/MRITestData.jl) to fetch
public datasets. The package is MIT-licensed; **the datasets are not** — each provider has its own
terms and citation requirements:

- **M4Raw** (notebook 10) — CC-BY. Cite Lyu et al., *M4Raw: A multi-contrast, multi-repetition,
  multi-channel MRI k-space dataset for low-field MRI research*, Scientific Data 10, 264 (2023).
- **OCMR** (notebook 11) — OCMR data-use terms. Cite Chen et al., *OCMR (v1.0) — Open-Access
  Multi-Coil k-Space Dataset for Cardiovascular Magnetic Resonance Imaging*, arXiv:2008.03410
  (2020).

On first use `MRITestData` asks where downloads should go; the notebooks default to the package's
scratch cache.
