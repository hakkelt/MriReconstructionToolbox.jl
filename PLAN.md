# PLAN.md — what is still open

The single planning document for this repository. It carries only work that is **still live**;
finished work is recorded in git history and in `docs/src/high-level/performance.md`, which is why
the two earlier planning files (`IMPLEMENTATION_PLAN.md`, `TODO.md`) were deleted rather than
merged in — duplicating an applied record is how they went stale.

## Open work

### C7 — per-iteration proximal buffers

`src/regularization/multi_scale_low_rank_reg.jl` (the `blocks`/`weights` fields and the
materialized prox), `src/regularization/locally_low_rank_reg.jl` (the per-block SVD workspace) and
`src/regularization/plug_and_play_reg.jl` (the denoiser's scratch image) each allocate their
working buffers once per proximal call rather than once per solve. Caching them per materialized
term would remove those allocations from the inner loop.

Blocked on the same prerequisite as when it was first written: nothing documents whether a
materialized term may be entered from more than one thread. Task splitting solves slabs
concurrently and each slab materializes its own terms today, but that is an implementation detail,
not a stated contract — a shared buffer hung off the term would be a data race the moment it stops
being true. Settle the contract first, then cache.

### C9 — refresh the benchmark record

The multi-toolbox tables in `docs/src/high-level/performance.md` predate the threading work, and
their 8-thread columns are stale by construction. Re-run `benchmark/comparison/scripts/` at 1 and
8 threads across OpenBLAS and MKL on an exclusive node, refresh the accuracy race, and update the
page.

Three landed changes each move numbers and must be in the run: the bounded ADMM penalty
(`SpectralRadiusApproximationPenalty`'s `rho_min`/`rho_max`), the NestedThreading 0.1.1 thread
budgets, and the scale-correct non-Cartesian warm start.

### Phase 6 items deferred past the notebook re-run

These were deliberately declined during the Phase 6 review pass because they restructure code that
the notebook run had not yet validated. That run has now happened, so they are actionable — each
still needs its own measurement before any rewrite.

- **`VCAT` makes the partitioned adjoint quadratic in frame count.** The unequal-per-frame-count
  path in `src/encoding/subsampling_operators.jl` builds a `VCAT` of `GetIndex` operators whose
  domains are the whole multi-frame k-space; `VCAT`'s adjoint accumulates through a buffer and
  `GetIndex`'s adjoint zeroes the full domain first, so T frames cost ~2T full k-space passes per
  adjoint application where the dense `BatchOp` path costs one scatter. The blocks are
  domain-disjoint, so the accumulation is pure waste. Measure adjoint wall time against frame
  count before rebuilding — the equal-count path (the common case) does not go through this at
  all.
- **`PartitionedKSpace`'s special cases.** `_measurement(...)` at twelve call sites, seven
  hand-placed `_reject_partitioned` guards, raggedness re-derived four times, and
  `_simulate_partitioned_acquisition` reimplementing the forward model. Each is survivable; the
  set is the maintenance cost of the unequal-count feature. The fix in each case is an accessor
  that cannot be skipped rather than a guard that can be forgotten.
- **The warm-start provenance flag.** `is_default_iterative_adjoint`, the `precomputed_L` keyword
  threaded down the call chain and `_warm_start_needs_operator_norm` shadowing
  `_should_estimate_operator_norm` all exist to carry one fact — which branch produced `x₀` — that
  a single function owning the default warm start and its scaling would not need. Related and
  worth its own number: pure-Krylov reconstructions now pay `estimate_opnorm` (~144 ms per slab)
  for the warm-start rescale alone, where `_should_estimate_operator_norm` used to skip it; the
  cheap proxy `dot(x̂, 𝒜'𝒜x̂)/‖x̂‖²` would give the correction with one operator application.

## Findings that are decisions, not work

Recorded so nobody re-derives them. Each was measured; none is a defect.

- One `A` + `A'` application runs at 3.50 ms against a 2.95 ms two-FFT floor; MRIReco's fused
  `AHA` is 3.34 ms. The operator stack is at the achievable floor — do not optimize.
- `NamedDimsOp` is free: wrapping and unwrapping costs nothing measurable.
- ADMM applies 10 normal-operator products per outer iteration, against 11 for BART, MRIReco and
  SigPy. There is no product to remove.
- ADMM's per-iteration norm computations are 0.2 % of the solve, and its one-trip `@threads` is
  0.04 %. Both were explicitly dropped rather than left pending.
- `estimate_opnorm` costs ~144 ms per `reconstruct` and is the whole of MRT's setup gap against the
  other toolboxes. It is a real cost with no cheaper exact form; the proxy under C9's third bullet
  is the only open idea.
