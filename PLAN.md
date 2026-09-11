# PLAN.md — what is still open

The single planning document for this repository. It carries only work that is **still live**;
finished work is recorded in git history and in `docs/src/high-level/performance.md`, which is why
the two earlier planning files (`IMPLEMENTATION_PLAN.md`, `TODO.md`) were deleted rather than
merged in — duplicating an applied record is how they went stale.

## Open work

### C9 — refresh the benchmark record

The multi-toolbox tables in `docs/src/high-level/performance.md` predate the threading work, and
their 8-thread columns are stale by construction. Re-run `benchmark/comparison/scripts/` at 1 and
8 threads across OpenBLAS and MKL on an exclusive node, refresh the accuracy race, and update the
page.

Four landed changes each move numbers and must be in the run: the bounded ADMM penalty
(`SpectralRadiusApproximationPenalty`'s `rho_min`/`rho_max`), the NestedThreading 0.1.1 thread
budgets, the scale-correct non-Cartesian warm start, and the warm-start scale proxy below (which
takes ~40 % off a pure-Krylov `reconstruct`).

### C7' — a preallocated SVD workspace for the block-SVD proximal terms

This replaces the original C7 ("cache the per-iteration proximal buffers"), which was measured and
found to be aiming at the wrong allocation — see the decisions section.

`svd!` allocates its own factors and LAPACK workspace on every call: ~32.6 kB for a 64×16
`ComplexF32` block, which at 256 blocks is 8.0 MiB of the 10.0 MiB a `LocallyLowRank` prox
allocates per call, and drives the GC share of prox time (4.9 % serial, 12.3 % at 4 threads,
19.0 % for `MultiScaleLowRank`). Removing it means calling the LAPACK driver directly with a
workspace held per task — one workspace per `@budgeted_threads` task index, which the
now-documented thread-entry contract (`materialize`'s docstring) permits.

Worth its own measurement of the win before committing to a LAPACK-level wrapper: the ceiling is
the GC share above, not the whole allocation.

## Findings that are decisions, not work

Recorded so nobody re-derives them. Each was measured; none is a defect.

- One `A` + `A'` application runs at 3.50 ms against a 2.95 ms two-FFT floor; MRIReco's fused
  `AHA` is 3.34 ms. The operator stack is at the achievable floor — do not optimize.
- `NamedDimsOp` is free: wrapping and unwrapping costs nothing measurable.
- ADMM applies 10 normal-operator products per outer iteration, against 11 for BART, MRIReco and
  SigPy. There is no product to remove.
- ADMM's per-iteration norm computations are 0.2 % of the solve, and its one-trip `@threads` is
  0.04 %. Both were explicitly dropped rather than left pending.
- **C7, the per-iteration proximal buffers, is not where the allocation is.** The gather buffers
  in `locally_low_rank_reg.jl` / `multi_scale_low_rank_reg.jl` are a fifth of the churn at most;
  `svd!`'s own allocations are the rest (numbers under C7' above). Caching the gather buffers
  alone would recover a couple of per cent of prox time, and would have to be indexed per task
  anyway. The blocker it used to carry — whether a materialized term may be entered from more
  than one thread — is now settled and written down in `materialize`'s docstring: one thread at a
  time from the outside, but the term's own internals thread across slabs and blocks, so shared
  state must be per task.
- **`PartitionedKSpace`'s special cases stay.** The alternative in the earlier plan — "an accessor
  that cannot be skipped rather than a guard that can be forgotten" — would mean hiding
  `kspace_data` behind an accessor (it is public API that users read) or splitting a separate
  partitioned acquisition type, both far larger than the problem. `_measurement` already *is* the
  accessor and is used at all 11 sites; the residual risk is a new entry point forgetting
  `_reject_partitioned`, and that is now covered by a test listing every guarded entry point
  (`test_reconstruction_integration.jl`, "Per-frame subsampling: unequal sample counts per frame").
- `estimate_opnorm` costs ~118 ms per `reconstruct` on a 192²×8 Cartesian acquisition (812 ms
  radial). Where the value is the algorithm's step size it is a real cost with no cheaper exact
  form. Where only the default warm start needs it, `_warm_start_scale_proxy` now replaces it with
  one normal-operator application: 11 ms and 62 ms respectively, 0.4 % / 3.2 % under the power
  estimate, which took a 30-iteration CG-SENSE solve from 0.448 s to 0.253 s (Cartesian) and
  2.73 s to 1.82 s (radial) at unchanged NRMSE.
- **`VCAT`'s adjoint is no longer quadratic in block count.** Measured before the fix, on a
  128×128×T domain with one `GetIndex` per frame: 0.81 ms at T = 4 rising to 162.6 ms at T = 64,
  against a forward that stayed under 0.5 ms. `AbstractOperators.add_mul!` (specialized for
  `GetIndex`) replaced the per-block "full-domain buffer, then add", giving 0.13 ms and 2.9 ms —
  linear, and bit-identical to the dense reference.
