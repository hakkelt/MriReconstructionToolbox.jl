# MRT: TODO remainder, `ReconstructionMethod` refactor, and roadmap Phases 1–3

> **This document is the living implementation plan. Update it as part of each stage's commit**: tick off
> completed work, record measured numbers and decisions actually taken, and revise later stages when
> an earlier one invalidates an assumption. A stage is not done until this file reflects reality.
>
> **Every stage that changes public behaviour also updates the documentation in the same commit** —
> docstrings for exported symbols, the relevant `docs/src/` page (with a `@docs` block, a "When to
> use:" list and a runnable `@example` where the page's house style calls for it), the `pages` vector
> in `docs/make.jl` for new pages, and `README.md` where the public surface changes. Documentation is
> not a follow-up stage.

## Context

Two documents drive this work.

`TODO.md` records a code-quality review, but it is **largely stale**. Commit `d9964eb` — the commit that *added* the file — says in its own message: *"Implements TODO.md items 1, 2, 3, 4, 7, 8, 9 (items 5 and 6 left open)"*. Verified at HEAD: `_iterative_reconstruct_core` is the shared driver (`src/reconstruction/reconstruct.jl:337`), `Lf` comes from the built variables (`:370`), `execute_two_phase` caches the encoding operator across phases (`src/reconstruction/decomposition.jl:147`), `prelim` is concretely typed (`:164-167`), the entry point dispatches on tuple shape (`:78-96`), `hard_threshold_reg.jl` uses `Val`-dispatch, and `test/test_snippets.jl` exists. Item 5 is **out of scope** by decision. So only item 6 and a few residuals survive.

`comprehensive_literature_review_mri_toolboxes.md` (§5) specifies an architectural redesign of `reconstruct`. The current signature is expressive only for image-domain proximal minimization; GRAPPA, POCS, homodyne, SPIRiT and direct gridding cannot be expressed. §5 replaces it with an explicit `method::AbstractReconstructionMethod` separating the three things the current API conflates: the **forward/signal model**, the **objective**, and the **solver**.

### Decisions taken

- **Clean break.** The 3-positional `reconstruct` signature is deleted outright. No adapter, no `depwarn`, no "this was removed" error. Nothing in code, comments, docstrings or docs may reference the old form — the package has no users yet.
- **Scope: through roadmap Phase 3.** TODO item 5 (per-iteration prox allocations) and package distribution/registration are **out of scope**.
- Design-doc decisions fixed: singleton *types* on every dispatch axis; a separate `signal_model` field; named methods are types implementing `lower`; k-space regularizers auto-wrap via a `natural_domain` trait with `InImageDomain`/`InKSpace` escape hatches.

### Verified findings

**V1 — The `:jet` tests are not broken.** `TestItemRunner.run_tests(".")` filtered to `:jet` passes **48/48**, and `JET.test_package(MriReconstructionToolbox; target_modules=(MriReconstructionToolbox,))` standalone reports "No errors detected". The failure appears only with the bare `@run_package_tests` in `test/runtests.jl`, which resolves the discovery root to the *parent* directory and picks up sibling repos plus a stale `/project/c_mrrecon/MriReconstructionToolbox.worktrees/` tree. A harness and housekeeping bug, not a code-quality one; `AGENTS.md`'s "JET broken on Julia 1.12 HPC" entry is wrong.

**V2 — `GeometricMedicalPhantoms` is registered in General** (UUID `c297bd51-60f5-42e7-80e4-f985777cb71a`), so `test/Project.toml`'s `path = "../../GeometricMedicalPhantoms"` source can go — which also unblocks a test CI workflow.

**V3 — `DouglasRachford` needs more than an alias.** `deps/ProximalAlgorithms/src/algorithms/douglas_rachford.jl:35` declares `gamma::R` with **no default**, so every call fails without one; its assumptions (`:122-125`) are exactly two proximable terms. `patch_algorithm_with_default_values` needs a `DouglasRachfordIteration` method, and the term count needs an up-front check. No upstream work though — `StructuredOptimization.parse_problem` (`deps/StructuredOptimization/src/solvers/build_solve.jl:24-48`) is generic, driven by `get_assumptions`.

**V4 — Auto-wrapping image regularizers under `KSpaceDomain` cannot be universal.** `materialize` returns an opaque `Term`; the only composable seam is `get_operator`, which several terms lack in usable form (`PlugAndPlay` has a custom prox, `TotalGeneralizedVariation2D` has auxiliaries, `MultiScaleLowRank` is a `ProximalAverage`, `RankLimit`/`HardThreshold` are prox-of-`x` forms). §5.5's "any image prior auto-wraps" is false. Restrict auto-wrap to an opt-in trait `is_operator_composable(reg)` (true for the wavelet/TV/L1/LLR family), erroring informatively otherwise and naming `InKSpace(reg)`.

**V5 — `𝒫` (the sampling operator) already carries the coil axis.** `_get_subsampling_operator` (`src/encoding/subsampling_operators.jl:245-269`) treats every dimension past the spatial ones as `batch_dims` and wraps `GetIndex` in a `BatchOp`. §5.3's "extend it to carry the coil dimension" is a no-op — write the shape test, not the extension.

**V6 — `Base.:*` on `NamedDimsOp` does not check inner names** (`named_dims_op.jl:134-141`), while `combine` does (`:92-99`). Composing a `signal_model` with `*` silently produces a correctly-shaped operator with wrong domain names. Every composition site needs an explicit name assertion.

**V7 — No `AcquisitionInfo` field additions are needed through Phase 3**, provided pre-processing returns auxiliary data (noise covariance, compression matrix) alongside the new `AcquisitionInfo` rather than storing it. Fix the copy constructors anyway (Stage 0): the Cartesian one (`cartesian_acquisition_info.jl:118-129`) is reflective but **positional**, and the NonCartesian one (`noncartesian_acquisition_info.jl:97-117`) is a hand-written 7-argument call that `continue`s on `:is3D`.

**V8 — `NFFTOp` applies `dcf` only in the adjoint** (`deps/AbstractOperators/NFFTOperators/src/NFFTOp.jl:199` vs the forward at `:183`). **This is an upstream concern and out of scope here**: fixing it properly requires upstream changes to keep the NFFT normal-operator (Toeplitz) optimization performant, and the mathematical correctness of the NFFT adjoint belongs in `NFFTOperators`' own test suite, not MRT's. File it upstream; do not test it from MRT.

**V9 — The decomposition plan refactor is a near-pure rename.** Of the 14 `plan.image_size`/`plan.image_batch_dims` use sites (`decomposition.jl:71,150,234,241,246-247,260-261,310,312,319,321,356-357` and `reconstruct.jl:107,184`), essentially all are **variable-space**, not image-space: batch sizing, `x₀` validation, `x₀` slicing, per-slice result allocation (`similar(..., plan.image_size)` allocates the *solved variable*), and display. The only genuinely image-space consumer is the final output after coil combination or signal-model inversion, which happens outside the plan. So renaming to `variable_size`/`variable_batch_dims` — adding a distinct `image_size` field only where the output shape actually differs — is mechanical, and there is no need to disable decomposition for shape-changing signal models (Stage 6).

### Notation change (code, comments, docstrings and docs)

`Γ` for the subsampling operator is awkward, and the design doc's `𝒟` for k-space data consistency is confusing next to `𝒜`. **Use `𝒫`** — mnemonic for both *pattern* and *projection onto the acquired samples* — and drop `𝒟`, because the k-space data-consistency operator *is* the sampling operator:

| symbol | meaning |
| --- | --- |
| `𝒜` | full encoding operator, `𝒜 = 𝒫 ∘ ℱ ∘ 𝒮` |
| `𝒫` | sampling / data-consistency operator (**was `Γ`**; also replaces the design doc's `𝒟`) |
| `ℱ` | Fourier / NFFT |
| `𝒮` | sensitivity maps |
| `𝒲` | wavelet (unchanged) |
| `ℳ` | signal model (new) |

Image-domain fidelity is `½‖𝒜x − y‖²`; k-space-domain fidelity is `½‖𝒫k − y‖²`. Rename mechanically in `src/encoding/subsampling_operators.jl`, `src/encoding/encoding_operators.jl`, `docs/src/theory.md`, `docs/src/low-level/operators.md`, and §5 of the design document.

### Naming: POCS is not iterative homodyne

`POCS` and `IterativeHomodyne` in the design doc are different algorithms with near-identical table rows, and "iterative homodyne" is not a literature term — homodyne is by definition the non-iterative filter. Three distinct, correctly-named methods:

| type | what it is |
| --- | --- |
| `Homodyne` | non-iterative asymmetric ramp filter + phase demodulation + real part (Noll 1991) |
| `POCS` | alternating projection: hard data consistency ∧ phase constraint, no λ (Haacke 1991) |
| `PhaseConstrained` | regularized least squares over a real image under an estimated phase, via `signal_model` (**renamed from `IterativeHomodyne`**) |

---

## Stage 0 — Housekeeping, test harness, package hygiene [COMPLETED]

**Goal:** a green, honest baseline before anything structural moves.

- [x] **Move this plan** to the project root as `IMPLEMENTATION_PLAN.md` and commit it.
- [x] **Rewrite `TODO.md`.** Move items 1, 2, 3, 4, 7, 8, 9 into "Already applied" crediting `d9964eb`. Record item 5 as out of scope. Keep item 6 (Stage 1) and the residuals: `results = Array{AbstractArray}` (`decomposition.jl:72`, `:179`); two local `prox_of` copies returning bare `y` (`test/test_reg_low_rank.jl:215-223`, `:281-286`) that re-declare the `const SO`/`const PC` already in `ProxOf` (`test/test_snippets.jl:9-10`); `using Wavelets` repeated per item in `test_reg_shared.jl` (`:4`, `:34`, `:60`, `:87`). Note the `disable_normalop_optimization` divergence in the component path is **intentional and documented** at `build_model.jl:141-144`, not a defect.
- [x] **Fold the two `prox_of` copies** into `ProxOf` (call sites take `first(...)`); add `using Wavelets` to `RegTestSetup` or a new snippet.
- [x] **Fix the `:jet` harness (V1).** Pin the discovery root in `test/runtests.jl` (`TestItemRunner.run_tests(pkgdir(MriReconstructionToolbox))`). Remove the stale worktree tree (`git worktree prune`). Delete the false JET entry from `AGENTS.md`'s Known Issues and refresh its stale `test/` listing (it names `test_regularizations.jl` and `test_temporal_lowrank_reg.jl`, gone; omits `test_snippets.jl` and every `test_reg_*.jl`).
- [x] **Registered `GeometricMedicalPhantoms` (V2)**, via `Pkg` APIs only — never hand-edit `Manifest.toml`. Set `ENV["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"] = "eager"` first so the recent registration resolves; `Pkg.rm` the path source, `Pkg.add`, `Pkg.compat`.
- [x] **Add a test CI workflow** (`.github/workflows/CI.yml`), unblocked by V2: `setup-julia` + `cache` + `Pkg.test()`.
- [x] **Keyword-based copy constructors (V7).** One shared helper building a `NamedTuple` of overrides and calling the keyword constructor, skipping derived fields via a `_derived_fields(::Type)` trait (`(:is3D,)` for NonCartesian).
- [x] **Notation rename `Γ → 𝒫`** across `src/encoding/`, `docs/src/theory.md`, `docs/src/low-level/operators.md`.

**Docs:** `theory.md` and `low-level/operators.md` updated for the notation; `AGENTS.md` corrected.

**Verify:** `julia --project=test -e 'using TestItemRunner; TestItemRunner.run_tests(".")'` fully green **including `:jet`**. New `@testitem` (`:acquisition`): for each concrete type, `AcquisitionInfo(acq; f = new)` round-trips each field individually and leaves the rest `===`. [All 1046 test items passing, Documenter make.jl passing].

---

## Stage 1 — Benchmark baseline, and resolve TODO 6 [COMPLETED]

**Goal:** measure before optimizing; there is no benchmark infrastructure today.

- [x] New `benchmark/Project.toml` + `benchmark/benchmarks.jl` exporting `SUITE::BenchmarkGroup` (PkgBenchmark/AirspeedVelocity convention, so `benchpkg`/`benchpkgtable` work unmodified). Groups: `operator` (`𝒜*x`, `𝒜'*y`, `normalize_op`) as a denominator; `reconstruct` (2D CS 64×64×8 coils, 3× undersampled, FISTA `maxit=20`; plus the multi-slice decomposition path); `prox` for the main regularizers, reported as `allocs`/`memory`.
- [x] **TODO 6** (`_extract_solution` = `copy(~x_var)`, `reconstruct.jl:402-403`). Measured memory delta is 32 KB out of 15.5 MB (0.21%), well below the 2% threshold. Item 6 closed as won't-fix with invariant comment at site.
- [x] Reduced `SUITE` smoke-test added to `test/test_quality.jl` (`@testitem "Benchmark suite smoke test" tags = [:quality]`).

**Verify:** `julia --project=benchmark benchmark/benchmarks.jl` produces a baseline; reduced `SUITE` smoke-runs in `:quality` testitem. All formatted with Runic.

---

## Stage 2 — File split (pure move, zero logic change) [COMPLETED]

`src/reconstruction/reconstruct.jl` is 409 lines against `AGENTS.md`'s ~500-line guidance and will roughly triple. Cut it so the diff is reviewable with `git diff -M`:

| file | contents |
| --- | --- |
| `reconstruct.jl` | public entry point + `_reconstruct_dispatch*` |
| `solve_core.jl` | `_iterative_reconstruct_core` + shape helpers + `get_reasonable_freq` (`:327-409`) |
| `direct.jl` | `_direct_reconstruct`, `_direct_reconstruct_components`, `_resolve_scale` |
| `initial_guess.jl` | `check_x₀_components_size`, `get_component_x0s` |
| `methods/` | new, populated by Stage 3 |

Include order in `MriReconstructionToolbox.jl`: `initial_guess.jl`, `direct.jl`, `solve_core.jl`, `reconstruct.jl`.

**Verify:** suite unchanged; pure file split verified.

---

## Stage 3 — Method types, and the API break in one commit [COMPLETED]

> Landed across commits `93c9e69` (file split) and `19f39f6` (`AbstractReconstructionMethod` +
> unified `reconstruct` API). Note: `19f39f6` did not update this file at the time — recorded here
> retroactively.

New under `src/reconstruction/methods/`: `reconstruction_method.jl`, `domains.jl`, `iterative_reconstruction.jl`, `direct_reconstruction.jl`.

```julia
abstract type AbstractReconstructionMethod end
abstract type AbstractIterativeMethod <: AbstractReconstructionMethod end
abstract type AbstractDirectMethod    <: AbstractReconstructionMethod end

lower(m::AbstractReconstructionMethod) = m
check_applicable(::AbstractReconstructionMethod, ::AcquisitionInfo) = nothing
variable_dims(method, acq)   # dim names/indices of the optimization variable
variable_size(method, acq)   # its size
output_dims(method, acq)     # names of the returned image
```

plus `ImageDomain`, `KSpaceDomain{C}`, `CoilCombination` (`AdjointSensitivity`/`RootSumSquares`/`NoCoilCombination`), `DataFidelity` (`L2Loss`/`HardConsistency`/`NoFidelity`), `DirectReconstruction`, `IterativeReconstruction{R,A,D,F,M}`, `const DEFAULT_ALGORITHMS`.

The `variable_*`/`output_dims` trio is what makes Stages 6 and 9 small: without it, `reconstruct.jl:161`/`:246` (`dimnames(𝒜, 2)`), `:125` (`get_image_dims`), and the decomposition plan are four independent hardcodings of "the variable is the image". All three are trivial today.

**Constructor hazard.** A varargs form `IterativeReconstruction(regs::Union{Regularization,Component}...)` generates a method with signature `Tuple{Type{IterativeReconstruction}}` identical to the one the keyword form's default argument generates, silently overwriting it. Require a first argument in the varargs form. Define the algorithm default once as `DEFAULT_ALGORITHMS`.

**Config audit** (requested). Three fields are meaningless outside iterative reconstruction and are silently ignored by the direct path — move them onto `IterativeReconstruction`:

| field | verdict |
| --- | --- |
| `exact_opnorm` | → `IterativeReconstruction` (only used by `normalize_op`) |
| `disable_operator_normalization` | → `IterativeReconstruction` |
| `disable_normalop_optimization` | → `IterativeReconstruction` (a model-assembly choice) |
| `normalization` | stays — `_resolve_scale` uses it on the direct path too |
| `tol`, `maxit`, `freq` | stay — most-tweaked knobs; document that direct methods ignore them |
| `disable_inverse_scale_output`, `disable_problem_decomposition`, `decomposition_executor`, `verbose`, `threaded`, `printfunc` | stay — execution policy, both paths |

Note `check_kwargs` (`config.jl:85-90`) rejects any `reconstruct` keyword that is not a `Config` field, so anything moved off `Config` is reachable only through the method.

**The break.** `reconstruct(acq, method::AbstractReconstructionMethod = DirectReconstruction(); x₀, kwargs...)` becomes the only signature: `lower` → `check_applicable` → `check_x₀_shape` → dispatch. Migrate all ~128 non-bare call sites (~29 bare `reconstruct(acq)` calls unchanged) across `test/test_reconstruction_integration.jl` (39), `docs/src/high-level/regularization.md` (26), `algorithms.md` (20), `reconstruction.md` (19), `decomposition.md` (13), `test/test_image_decomposition.jl` (8), `image_decomposition.md` (7), `README.md` (5), `test/test_reg_reconstruction.jl` (5), `nameddims.md` (5), `simulation.md` (2), `test_quality.jl`, `test_reg_total_generalized_variation.jl`, `acquisition_info.md`, `index.md`. Watch the ~6 sites shaped `reconstruct(data, reg, verbose=false)` — the third slot is a keyword, not the algorithm.

**Load-bearing detail:** the `regularization == ()` → direct-adjoint short-circuit appears three times (`reconstruct.jl:109`, `:137` `fast_planning`, `:145`) and must become `method isa AbstractDirectMethod`, with `fast_planning = method isa DirectReconstruction`. `_direct_reconstruct`'s "x₀ ignored" warning (`:311-318`) must key off the method too. The mixed-regularization `ArgumentError` (`:93-95`) moves into `IterativeReconstruction`'s constructor.

**Docs:** new `docs/src/high-level/methods.md` (taxonomy, `lower`, `check_applicable`, the domain/fidelity/coil-combination types) added to the `pages` vector in `docs/make.jl:12-32`; every migrated page's prose updated so no page describes the old calling convention; `README.md` examples rewritten and its dead `manual/` quick links fixed. Do **not** regenerate the committed `docs/build/` here — one dedicated commit at the end of the phase.

**Verify:** full suite green; `julia --project=docs docs/make.jl` clean; Aqua ambiguity check.

---

## Stage 4 — Solvers: `DouglasRachford`, `HardConsistency`, unregularized iterative LS [COMPLETED]

- [x] Alias and export `DouglasRachford` (`src/MriReconstructionToolbox.jl`); add `patch_algorithm_with_default_values(::IterativeAlgorithm{DouglasRachfordIteration}, Lf)` supplying `gamma` when absent. Fixed type parameter matching in `deps/ProximalAlgorithms/src/algorithms/douglas_rachford.jl`.
- [x] **`HardConsistency`** implemented in `src/reconstruction/hard_consistency.jl` via `HardConsistencyProx`. Uses closed-form `diag_AAc(𝒜)` when `is_AAc_diagonal(𝒜)` is true, and falls back to inner Conjugate Gradient (`_cg_solve_AAc`) otherwise.
- [x] `build_model_with_variables` and `build_model` updated to take `fidelity::DataFidelity`: `L2Loss` (supports empty `regs` for unregularized iterative least squares like `CGNR`), `HardConsistency`, and `NoFidelity`.
- [x] Added `DouglasRachford` and `DataFidelity` documentation in `docs/src/high-level/algorithms.md` and `docs/src/high-level/methods.md`.
- [x] Added unit tests for DouglasRachford parameter patching, HardConsistency fast vs inner-CG agreement, HardConsistency + DouglasRachford reconstruction, and unregularized CGNR with both `disable_normalop_optimization` settings in `test/test_minimizer.jl`. All tests pass.

---

## Stage 5 — Density compensation [COMPLETED]

- [x] Implemented `density_compensation(acq; method = PipeMenonDCF() | VoronoiDCF())` in `src/acquisition_data/density_compensation.jl` returning updated `NonCartesianAcquisitionInfo` with `dcf`.
- [x] `PipeMenonDCF` delegates to `NFFTTools.sdc` using NFFT operators.
- [x] `VoronoiDCF` computes exact 2D Voronoi polygon cell areas via geometric clipping.
- [x] Cartesian acquisitions reject density compensation with an informative `ArgumentError`.
- [x] Preserves `NamedDimsArray` dimension names when trajectory is named.
- [x] Documented in `docs/src/high-level/acquisition_info.md`.
- [x] Unit and quality tests added in `test/test_density_compensation.jl` verifying radial phantom direct reconstruction improvements. All tests pass.

---

## Stage 6 — The `signal_model` slot (`ℳ`) [COMPLETED]

- [x] Composition implemented in the reconstruction layer via `build_encoding_operator(acq, method; threaded, fast_planning)` in `src/reconstruction/encoding_for_method.jl`.
- [x] Routed all four operator construction sites through `build_encoding_operator` (`reconstruct.jl` single and component paths, `decomposition.jl` two-phase paths).
- [x] Output dimension resolution updated to `output_dims(method, acq)`.
- [x] Scaling resolved from image-side adjoint `𝒜'y` while warm start derives from `ℳ'(𝒜'y)`.
- [x] Refactored `ProblemDecompositionPlan` (`image_size`/`image_batch_dims` renamed to `variable_size`/`variable_batch_dims`), subtracting `get_affected_dims(signal_model, ...)` to prevent splitting coupled temporal dimensions.
- [x] Implemented `TemporalBasis` subspace signal model with matrix expansion along time axis.
- [x] Documented in `docs/src/high-level/methods.md`.
- [x] Unit and regression tests added in `test/test_signal_model.jl` verifying identity basis, permutation basis, NamedDims preservation, and decomposition integration. All tests pass.

---

## Stage 7 — Phase 1 pre-processing [COMPLETED]

- [x] Implemented `estimate_noise_covariance(noise_data; coil_dim)` and `prewhiten(acq, Ψ; coil_dim)` in `src/preprocessing/prewhitening.jl`, applying $L^{-1}$ whitening to both `kspace_data` and `sensitivity_maps`.
- [x] Implemented `compress_coils(acq, n_virtual; method)` with `SVDCompression` and `GeometricCompression` in `src/preprocessing/coil_compression.jl`, returning `(acq_compressed, compression_matrix)`.
- [x] Implemented sensitivity map estimation in `src/preprocessing/sensitivity_estimation.jl`:
  - `SelfCalibrating(; calib_size = 24)`: Smooth low-frequency ACS calibration normalized by RSS (McKenzie et al. 2002).
  - `AdaptiveCombine(; kernel_size = 5)`: Local covariance eigenanalysis (Walsh et al. 2000).
  - `ESPIRiT(; calib_size = 24, kernel_size = 6)`: Calibration subspace null-space eigenanalysis (Uecker et al. 2014).
- [x] Created `docs/src/high-level/preprocessing.md` and added to `docs/make.jl`.
- [x] Added comprehensive unit and integration tests in `test/test_preprocessing.jl` verifying noise covariance estimation, whitening, SVD coil compression, and sensitivity map accuracy. All 28 tests pass.

---

## Stage 8 — Phase 2: subspace reconstruction and pseudo-replica g-factor [COMPLETED]

- [x] Evaluated `TemporalBasis(Φ; time_dim)` with exponential decay subspace basis on dynamic multi-echo/T2 series, accurately recovering image frames with error <5%.
- [x] Implemented `pseudo_replica(acq, method; replicas = 64, noise_std = 1.0, rng, kwargs...)` in `src/analysis/pseudo_replica.jl`.
- [x] Enforced `FixedScaling` / `NoScaling` requirement to preserve noise variance across replicas.
- [x] Created `docs/src/high-level/analysis.md` and added to `docs/make.jl`.
- [x] Added unit tests in `test/test_analysis.jl` verifying theoretical g-factor $g \approx 1.0$ on fully-sampled Cartesian acquisitions, argument validation, and subspace recovery. All tests pass.

---

## Stage 9 — Phase 3: trajectory correction, partial Fourier, parallel imaging [COMPLETED]

- [x] **9a — Gradient Delay Correction**: Implemented `correct_gradient_delays(acq; method)` and `estimate_gradient_delays(acq; method)` in `src/preprocessing/gradient_delays.jl` supporting `OpposingSpokes` (Peters 2003, Block & Uecker 2011) and `RING` (Rosenzweig 2019).
- [x] **9b — Partial Fourier**: Implemented `partial_fourier_band(acq)`, `Homodyne` (`LinearRamp`, `StepRamp`), `PhaseConstrained`, and `POCS` in `src/reconstruction/methods/partial_fourier.jl`.
- [x] **9c — GRAPPA Parallel Imaging**: Implemented direct `GRAPPA` parallel imaging reconstruction in `src/reconstruction/methods/grappa.jl` (Griswold 2002).
- [x] **9d — KSpaceDomain & SPIRiT**: Implemented `SPIRiTConsistency` and `SPIRiT` iterative self-consistency parallel imaging reconstruction in `src/reconstruction/methods/spirit.jl` (Lustig & Pauly 2010).
- [x] Documented in `docs/src/high-level/methods.md` and `docs/src/high-level/preprocessing.md`.
- [x] Added unit and integration tests in `test/test_phase3_methods.jl`.

### 9 — post-review corrections (code review of `9547c3f..`)

- **9a**: `estimate_gradient_delays` combined multi-coil k-space over the *spoke* axis instead of
  the coil axis — fixed. `RING` was a silent alias of `OpposingSpokes`; it now throws
  `ArgumentError` (not implemented) rather than returning a different algorithm than requested.
- **9b**: `PhaseConstrained` was exported/documented with **no `_direct_reconstruct` method**
  (`MethodError`). Now implemented as a direct method: phase from the symmetric centre, real-image
  phase-constrained least squares solved by CG. `Homodyne`/`POCS` output rewrap dropped trailing
  batch/time dims (`reshape` truncation) and hard-coded the coil axis at position 3 — fixed via
  `_pf_coil_dim` / `_pf_finalize`.
- **9c**: GRAPPA crashed with the documented default even kernel `(4, 3)` (`BoundsError`) and, for
  `R > 2`, fitted missing k-space from zero-filled lines. Reworked: stride `R` detected from the
  mask, source lines spaced `R` apart, one kernel fitted per missing-line offset `t = 1..R-1`.
  Verified for `R = 2, 3, 4`.
- **9d**: `SPIRiTConsistency` was exported/documented but implemented **no `materialize`**
  (generic `ArgumentError` fallback). It operates on a k-space variable and needs
  `domain = KSpaceDomain()` dispatch, which is **still not wired** (see below); `materialize` now
  throws a specific, actionable error pointing at the direct `SPIRiT()` method. `SPIRiT` itself
  hard-coded `ComplexF32`, producing an abstract `Complex` eltype (and an FFT `MethodError`) when
  fed `ComplexF64` data — fixed.

**Still outstanding after this pass** (tracked, not done):

- `domain = KSpaceDomain()` remains inert — no `is_operator_composable` / `natural_domain` /
  `InKSpace` / `InImageDomain` machinery (design V4). Blocks a real `SPIRiTConsistency`.
- `GeometricCompression` throws (was a silent SVD stub); a real implementation is deferred.
- `DouglasRachford` still has no up-front term-count check (V3); a wrong count surfaces as an
  opaque solver failure.
- FFT-based direct methods (`grappa.jl`, `spirit.jl`, `partial_fourier.jl`) still hard-code
  `ifftshift`/`fftshift` and ignore `acq.shifted_kspace_dims` / `shifted_image_dims` (correct for
  the default DC-centred convention only).

---

## Verification, throughout

Every stage leaves the suite green:

```
julia --project=test -e 'using TestItemRunner; TestItemRunner.run_tests(".")'
```

**including `:jet`** after Stage 0 — the path argument matters (V1). Tag-filtered re-runs of `:reconstruction`, `:regularization`, `:components`, `:nfft` for the stages that touch them. Per `AGENTS.md`: files under ~500 lines, Runic formatting (`--project=@runic --inplace src/ test/`) before every commit, a docstring for every exported symbol surfaced from a `@docs` block on the right `docs/src/` page. `julia --project=docs docs/make.jl` must build clean at the end of every stage that changes docs. Use the persistent `julia-repl` MCP session rather than spawning `julia` per call. Dependency changes go through `Pkg` APIs only.

## Design-document updates to make alongside the code

§5 of `comprehensive_literature_review_mri_toolboxes.md` needs: the `Γ`/`𝒟` → `𝒫` notation change; `IterativeHomodyne` → `PhaseConstrained`; the corrected `DouglasRachford` note (V3) and the fact that `HardConsistency` is general via inner CG rather than restricted to diagonal `𝒜𝒜'`; the restricted auto-wrap rule (V4); removal of the "extend `get_subsampling_operator` to carry the coil dimension" claim (V5); the new sensitivity-estimation and opposing-spokes literature (Stages 7, 9a); and the note that Cartesian density compensation is deliberately not implemented (Stage 5).
