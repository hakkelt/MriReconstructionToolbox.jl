# AGENTS.md — MriReconstructionToolbox

MriReconstructionToolbox (MRT) is a Julia package for MRI image reconstruction. It provides a
modular pipeline: acquisition data → encoding operators → regularization → reconstruction via
proximal algorithms.

## Mission

Keep changes minimal and localized; avoid unrelated refactors. Never weaken a test to force it
green — if a failure reflects a real bug, fix the source. When you touch public API, update its
docstring, the relevant `docs/src/**` page, and its tests in the same change.

## Architecture

```
AcquisitionInfo → Encoding operators → Regularization → Reconstruction
   Cartesian/       FFT/NFFT +          image/transform    ISTA/FISTA/
   NonCartesian     sensitivity maps    domain terms       ADMM/CG/CGNR
```

| Module | Directory | Role |
|---|---|---|
| Acquisition data | `src/acquisition_data/` | `AcquisitionInfo` types, dimension utilities, copy constructors |
| Encoding | `src/encoding/` | Fourier (FFT/NFFT), sensitivity map, subsampling operators; `NamedDimsOp` wrapper |
| Regularization | `src/regularization/` | one file per regularizer + `regularization.jl` (abstract type, contract, fallbacks) |
| Reconstruction | `src/reconstruction/` | `config.jl`, `build_model.jl`, `decomposition.jl`, `components.jl`, `reconstruct.jl` |
| Simulation | `src/simulation/` | phantom sampling patterns, coil sensitivities, full acquisition simulation |

`src/MriReconstructionToolbox.jl` is the authoritative list of source files (`include` order) and
exports — read it rather than trusting a tree here.

### Decomposition over batch dimensions

The reconstruction is decomposed over batch (non-image, non-time) dimensions: each slab is solved
independently, with and without regularization. `reconstruct.jl` merges the component and
single-variable paths, caches the encoding operator `𝒜`, and dispatches on the regularizer's
domain. When editing this path, preserve that a `NamedDimsOp` is unwrapped and rewrapped (not
reshaped in place), and that dimension symbols are resolved to integer indices *before* the image
is unnamed into a `Variable`.

### Key dependencies (custom forks, dev-pathed under `deps/`)

`AbstractOperators` (+ `FFTWOperators`, `NFFTOperators`, `WaveletOperators`, `DSPOperators`),
`StructuredOptimization`, `ProximalOperators`, `ProximalAlgorithms`, `OperatorCore`,
`NestedThreading`. These are local checkouts under `deps/` — never `Pkg.add` an upstream version;
`Pkg.instantiate` the existing Manifest.

### API gotchas

- `materialize` / `materialize_with_auxiliaries` / `materialize_all` are not exported — call as
  `MriReconstructionToolbox.materialize(reg, x::Variable; threaded)`.
- `Variable(T, dims...)` — splat, do not pass a tuple.
- `Base.reshape` on an `AbstractOperator` returns `Reshape(...)`.
- `create_sampling_pattern` returns `(:, mask)` when `subsample_freq_encoding=false` (default).
- `@reexport using AbstractOperators` and `... ProximalOperators` both export `Sum`; resolved via
  explicit `using AbstractOperators: Sum`.

## Adding a regularizer

New file `src/regularization/<name>_reg.jl`, `include`d in `MriReconstructionToolbox.jl`, type(s)
exported there. A regularizer is `struct Foo{T} <: Regularization` plus:

- `get_operator(::Foo, x::AbstractArray; threaded)` — the linear operator; wrap in `NamedDimsOp`
  when `x isa NamedDimsArray`, mapping input to output dimension names.
- `materialize(reg::Foo, x::Variable; threaded)` — build the `StructuredOptimization.Term`
  (operator ∘ norm function). Default throws.
- `get_affected_dims(::Foo, dimspec, image_dims)` — which image dims the term acts on.
- `scale_regularization(reg::Foo, factor::Real)` — only if the term is homogeneous (scale `λ`).
- `bind_dimensions(reg::Foo, image_dims)` — only if parameterized by a dim (`time_dim`, `dim`,
  possibly `nothing`/`Symbol`): resolve it to a concrete index here. Generic fallback is identity.
- `materialize_with_auxiliaries` — only if the term introduces extra optimization variables
  (see `TotalGeneralizedVariation2D`).

Add a `test/test_reg_<name>.jl` (`@testitem`, `tags = [:regularization]`) and a section in
`docs/src/high-level/regularization.md`.

## Testing

- **TestItems.jl** / **TestItemRunner.jl**. Each `@testitem` does `using MriReconstructionToolbox`
  and any extra packages. Multiple `@testitem` blocks per file are fine (regularizer files often
  have several); keep begin/end nesting shallow.
- Tags in use: `:encoding`, `:regularization`, `:reconstruction`, `:acquisition`, `:simulation`,
  `:minimizer`, `:components`, `:integration`, `:nfft`, `:quality` (+ `:aqua`, `:jet`),
  `:operators`. Combine as needed.
- Full suite: `julia --project=test test/runtests.jl`
- Filtered:
  ```sh
  julia --project=test -e 'using TestItemRunner; run_tests("."; filter = ti -> :regularization in ti.tags)'
  ```
- Quality: Aqua (`piracies=false`, `persistent_tasks=false`, `stale_deps=false`) and JET, in
  `test/test_quality.jl`.

## Formatting

Format with **Runic.jl** before committing (there is no `.runic.toml`; defaults apply):

```sh
julia --project=@runic -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace src/ test/
```

## Commit messages

- First line: `<type>(<scope>): <summary>` in the imperative mood, ~72 chars, no trailing period
  (`type` = `feat`/`fix`/`refactor`/`test`/`docs`/`chore`; `scope` optional).
- Blank line, then a body wrapped at ~72 chars explaining *what* changed and *why* — bullets for
  multiple distinct changes, naming the files touched.
- Trailers: attribute the model that wrote the change as co-author, and link the session.

Claude:

```
<type>(<scope>): <summary>

<body>

Co-Authored-By: Claude <Model> <noreply@anthropic.com>
Claude-Session: <session URL>
```

`<Model>` is the exact model, e.g. `Opus 5`, `Sonnet 5`, `Fable 5`.

Gemini:

```
<type>(<scope>): <summary>

<body>

Co-Authored-By: Gemini <model> <gemini@localhost>
```

Replace `<model>` with the exact model name, e.g.:

```
Co-Authored-By: Gemini 3.7 Flash <gemini@localhost>
Co-Authored-By: Gemini 2.5 Pro <gemini@localhost>
```

## Known issues

| Issue | Status | Notes |
|---|---|---|
| Aqua `stale_deps` check | Disabled | Subprocess fails with EAGAIN under HPC load |
| Aqua `persistent_tasks` check | Disabled | False positive on Julia 1.12 HPC |

Verify any other suspected issue against current source before acting — this table is pruned when
items are fixed.
