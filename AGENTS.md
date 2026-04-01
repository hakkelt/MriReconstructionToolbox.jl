# Copilot Instructions for MriReconstructionToolbox

MriReconstructionToolbox (MRT) is a Julia package for MRI image reconstruction. It provides a modular pipeline: acquisition data → encoding operators → regularization → reconstruction via proximal algorithms.

## Architecture

```
AcquisitionInfo → Encoding Operators → Regularization → Reconstruction
     ↓                    ↓                  ↓               ↓
  Cartesian/         FFT/NFFT +         Tikhonov/L1/     ISTA/FISTA/
  NonCartesian     sensitivity maps     TV/Wavelet/      ADMM/CG
                                        LowRank
```

| Module | Directory | Role |
|---|---|---|
| Acquisition Data | `src/acquisition_data/` | AcquisitionInfo types, dimension utilities, copy constructors |
| Encoding | `src/encoding/` | Fourier (FFT/NFFT), sensitivity map, subsampling operators |
| Regularization | `src/regularization/` | Tikhonov, L1, wavelets, TV, temporal Fourier, low-rank |
| Reconstruction | `src/reconstruction/` | Config, model building, solver dispatch |
| Simulation | `src/simulation/` | Phantom generation, sampling patterns, coil sensitivities |

### Key Dependencies (all custom forks)
- **AbstractOperators.jl** — operator algebra (Eye, DFT, HCAT, VCAT, Reshape, BatchOp, etc.)
- **FFTWOperators** — DFT operator (note: uses `num_threads` kwarg, NOT `threaded`)
- **NFFTOperators** — NFFT for non-Cartesian trajectories
- **WaveletOperators** — Wavelet transforms
- **StructuredOptimization.jl** — Variable/Term/problem algebraic optimization interface
- **ProximalOperators.jl** / **ProximalAlgorithms.jl** — proximal functions and solvers

### Important API Notes
- `materialize` is NOT exported — use `MriReconstructionToolbox.materialize(reg, x; threaded)`
- `DFT` accepts `num_threads` keyword, not `threaded` — map via `num_threads = threaded ? Threads.nthreads() : 1`
- `Variable(T, dims...)` — splat dimensions, do NOT pass a tuple: `Variable(Float64, 8, 8, 10)` not `Variable(Float64, (8, 8, 10))`
- `Base.reshape` is defined for `AbstractOperator` and returns `Reshape(...)`
- `create_sampling_pattern` returns `(:, mask)` when `subsample_freq_encoding=false` (default)
- `@reexport using AbstractOperators` and `@reexport using ProximalOperators` both export `Sum` — resolved via explicit `using AbstractOperators: Sum`

## Code Standards

### Julia Best Practices
- Follow Julia naming conventions: lowercase with underscores for functions, CamelCase for types
- Write type-stable code; verify with `@code_warntype` and JET
- Use multiple dispatch effectively
- Prefer immutable structs when possible
- Keep functions focused and composable

### Comments and Documentation
- Only comment when purpose is not obvious from name and implementation
- Write docstrings for exported functions

### Code Structure
- Keep files under ~500 lines; split into logical units
- Format with Runic.jl before committing (see Formatting section)

### Testing Requirements
- Use **TestItems.jl** and **TestItemRunner.jl** — each test file contains `@testitem` blocks
- One `@testitem` per file works best; multiple `@testitem` blocks per file can cause parse issues with deeply nested begin/end blocks
- Tags: `:encoding`, `:regularization`, `:minimizer`, `:reconstruction`, `:integration`, `:nfft`, `:quality`, `:jet`, `:acquisition`, `:simulation`
- Each `@testitem` must `using MriReconstructionToolbox` and any other needed packages
- Run quality assurance: Aqua.jl (ambiguities=false, piracies=false, persistent_tasks=false) and JET.jl

## Development Workflow

### Building and Testing

**Install dependencies** (from package root):
```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=test -e 'using Pkg; Pkg.instantiate()'
```

**Run full test suite**:
```sh
julia --project=test -e 'using TestItemRunner; TestItemRunner.run_tests(".")'
```

**Run filtered tests** (by tag):
```sh
julia --project=test -e '
    using TestItemRunner
    TestItemRunner.run_tests("."; filter = ti -> :encoding in ti.tags)
'
```

**Run filtered tests** (by name):
```sh
julia --project=test -e '
    using TestItemRunner
    TestItemRunner.run_tests("."; filter = ti -> ti.name == "LowRank regularization")
'
```

### Formatting
- This project uses **Runic.jl** for code formatting
- Install: `julia --project=@runic --startup-file=no -e 'using Pkg; Pkg.add("Runic")'`
- Format src/: `julia --project=@runic --startup-file=no -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace src/`
- Format test/: `julia --project=@runic --startup-file=no -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace test/`
- Always format code before committing

## Known Issues

| Issue | Status | Notes |
|---|---|---|
| `dimnames` mutation in `temporal_fourier_reg.jl` | Open | `transformed_dimnames[time_dim] = :frequency` tries to mutate a Tuple — NamedDims path untested |
| `dimnames(::BatchOp, ::Int)` missing | Open | JET reports no matching method in `encoding_operators.jl` |
| Aqua `stale_deps` check | Flaky on HPC | Spawns subprocess that can fail with EAGAIN under load |
| Aqua `persistent_tasks` check | Disabled | Known false positive on Julia 1.12 HPC |

## Package Structure

```
src/
├── MriReconstructionToolbox.jl    # Main module, exports, includes
├── acquisition_data/
│   ├── acquisition_info.jl        # Abstract AcquisitionInfo type
│   ├── cartesian_acquisition_info.jl
│   ├── noncartesian_acquisition_info.jl
│   ├── acquisition_info_copy.jl   # Copy constructors
│   └── dimension_utils.jl         # get_image_size, get_time_dim, etc.
├── encoding/
│   ├── named_dims_op.jl           # NamedDimsOp wrapper
│   ├── fourier_operators.jl       # FFT-based encoding
│   ├── nfft_operators.jl          # NFFT-based encoding
│   ├── sensitivity_map_operators.jl
│   ├── subsampling_operators.jl
│   └── encoding_operators.jl      # Main encoding pipeline
├── regularization/
│   ├── regularization.jl          # Abstract Regularization type, fallbacks
│   ├── image_domain_reg.jl        # Tikhonov, L1Image
│   ├── wavelet_reg.jl             # L1Wavelet2D, L1Wavelet3D
│   ├── total_variation_reg.jl     # TotalVariation2D, TotalVariation3D
│   ├── temporal_fourier_reg.jl    # TemporalFourier
│   └── low_rank_reg.jl            # LowRank, RankLimit
├── reconstruction/
│   ├── config.jl                  # Config struct
│   ├── decomposition.jl           # SVD decomposition utilities
│   ├── build_model.jl             # Assemble optimization problem
│   ├── progress_utils.jl          # Progress logging
│   └── reconstruct.jl             # Main reconstruct() function
├── simulation/
│   ├── subsampling.jl             # Sampling pattern generation
│   ├── sensitivities.jl           # Coil sensitivity maps
│   └── simulate_acquisition.jl    # Full simulation pipeline
├── scaling.jl                     # Data scaling strategies
└── utils.jl                       # General utilities
test/
├── runtests.jl                    # TestItemRunner entry point
├── Project.toml                   # Test-specific dependencies
├── test_encoding_op.jl            # Encoding operator tests
├── test_regularizations.jl        # Core regularization tests
├── test_temporal_lowrank_reg.jl   # TemporalFourier, LowRank, RankLimit tests
├── test_acquisition_data.jl       # Acquisition info, dimensions, sampling tests
├── test_reconstruction_integration.jl  # End-to-end reconstruction tests
├── test_minimizer.jl              # Solver/minimizer tests
└── test_quality.jl                # Aqua + JET quality tests
```
