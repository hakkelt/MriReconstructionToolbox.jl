# MRT implementation plan — the open `TODO.md` items

> **This document is the living implementation plan. Update it as part of each stage's commit**: tick off
> completed work, record measured numbers and decisions actually taken, and revise later stages when
> an earlier one invalidates an assumption. A stage is not done until this file reflects reality.
>
> **Every stage that changes public behaviour also updates the documentation in the same commit** —
> docstrings for exported symbols, the relevant `docs/src/` page (with a `@docs` block, a "When to
> use:" list and a runnable `@example` where the page's house style calls for it), the `pages` vector
> in `docs/make.jl` for new pages, and `README.md` where the public surface changes. Documentation is
> not a follow-up stage.
>
> The `ReconstructionMethod` refactor and roadmap Phases 1–3 (the
> `comprehensive_literature_review_mri_toolboxes.md` §5 design, formerly Stages 0–10 of this file)
> are **complete** — the last of it landed in `2b1ffd8` — and have been removed from this file.
> Their record is the git history.

## Context

`TODO.md` at `fec37ad` carries three groups of *open* work: the residual threading items under
"Performance & Threading Findings" (finding 2 "Still open" 0–3, finding 3 "Not done"), the
cross-toolkit gaps §7–§9 plus the five "smaller MRT-internal gaps", and the deliberately-parked
§5 (per-iteration prox allocations) and the `Array{AbstractArray}` residual. Everything else in
that file is already applied and is kept there only as a record.

Most of the work lands in the vendored forks under `deps/` — `AbstractOperators` (+
`FFTWOperators`, `WaveletOperators`), `StructuredOptimization`, `ProximalAlgorithms`. Those are
ours to change, but each is a separate package with its own test suite: a change there is not
done until that package's own tests are green *and* the MRT suite is green.

## The two phases

**Phase 1 — simple (`S1`–`S7`).** One function or one file each, no contract change, no design
decision to take first, no measurement campaign, nothing upstream to coordinate. Each is
independently committable and independently revertible, and none blocks another. Two of them
(`S2`, `S3`) are among the largest measured per-iteration wins in the whole plan — "simple" here
means *bounded*, not *unimportant*.

**Phase 2 — complex (`C1`–`C9`).** Everything that changes a type's contract, plumbs a value
across package boundaries, needs a measurement before the right answer is known, needs a decision
recorded before code can be written, or lands upstream in a package we do not vendor. These carry
real risk of being wrong in a way tests do not catch, so each states what must be measured or
decided *before* the edit.

Phase 1 first, in full: it removes ~all of the §9 arithmetic cost, so Phase 2's measurements are
taken against the cheaper baseline that will actually ship — measuring `C5`'s setup cost against
a solve that still pays the un-fused sign passes would fit the wrong constant.

## Ground rules

- **Measure before and after, on the same node.** Every performance claim below comes
  with a `benchmarking/scripts/recon_bench.jl` number recorded in this file. The claims inherited
  from `TODO.md` are hypotheses until re-measured — that file already contains one retracted
  diagnosis (the "54M allocations") and one refuted mechanism ("Polyester spinning"), both of
  which survived for weeks because nobody re-ran them.
- **NRMSE is the gate.** Every optimization below must leave the accuracy race
  (`comparison/scripts/run_accuracy_race.jl`) bit-comparable or better; a change that moves NRMSE
  is a bug, not a speedup, unless the entry says otherwise.
- **One commit per numbered item**, `<type>(<scope>): …` per `AGENTS.md`, and `TODO.md` is edited
  in that same commit: the item moves out of "open" into the applied record, with the measured
  number. An item is not done while `TODO.md` still advertises it as open.
- Run tests through the `julia_*` MCP session, filtered by tag; full suite before each commit.
- **Phase order is a rule, not a preference.** No `C` item starts before every `S` item it names
  is merged; where an item lists no dependency, it still waits for Phase 1 to finish so its
  measurements are taken against the shipping baseline.

---

# Phase 1 — Simple

Bounded, single-site changes. Any order; all seven can be in flight at once. Each gets its own
commit and its own `TODO.md` edit.

### S1 — `WaveletOp` normal operator is the identity but is not declared as one

**Done** (2026-09-03). Guarded on `L.wavelet isa Wavelets.WT.OrthoFilter` (orthogonal
families only, e.g. `wavelet(WT.db4)`); a biorthogonal/lifting-scheme wavelet (e.g.
`wavelet(WT.cdf97, WT.Lifting)`) now reports `false` on `has_optimized_normalop`,
`is_AcA_diagonal`/`is_AAc_diagonal`, `has_fast_opnorm`, and throws from `diag_AcA`/`diag_AAc`/
`opnorm` instead of silently claiming the identity. Verified `norm(W'*(W*x) - x) <= 1e-12` and
`get_normal_op(op) isa Eye` for the orthogonal case.


`deps/AbstractOperators/WaveletOperators/src/WaveletOperators.jl:125-131` declares
`is_AcA_diagonal = true` and `diag_AcA = 1`, but not `has_optimized_normalop`, so neither the
`Compose` constructor nor ADMM's `get_cg_operator` can use it: an orthogonal wavelet transform's
`WᴴW` runs a real `dwt!`/`idwt!` pair per inner CG iteration for a mathematical identity.

Add `has_optimized_normalop(::WaveletOp) = true` and
`get_normal_op(L::WaveletOp) = Eye(domain_type(L), size(L, 2); array_type = domain_array_type(L))`
— exactly what `FFTWOperators` does for its shift operators (`Shift.jl:412-415`). **Guard it on
orthogonality**: only for wavelet families where `diag_AcA == 1` holds; if `WaveletOp` can be
constructed with a biorthogonal filter, the trait must be `false` there. Check the type's fields
first and, if the distinction is not currently representable, make it representable rather than
declaring a false identity.

- Test: `norm(W'*(W*x) - x)` at machine precision for the supported families, and a
  `get_cg_operator` shape test showing the wavelet pair is gone.

### S2 — `_alternate_sign!` inner loop (§9, first cause)

**Done** (2026-09-03). Rewritten in `deps/AbstractOperators/FFTWOperators/src/Shift.jl`: the
parity contribution from every dimension but the first is hoisted into a per-column value
computed once, and the true per-element alternation (dimension 1) is a branch-free `@simd`
multiply by a precomputed `±1` pattern; the threaded branch parallelises the outer (column) loop
only, never the inner `@simd` run. Measured on `ComplexF32` with `benchmarking`-style timing:
128²×8 old=406µs new=152µs (**2.68x**); 64²×4×8 old=555µs new=201µs (**2.76x**) — both above the
1.6-1.9x estimate in `TODO.md` §9. Property-tested against the naive per-element formula over
every non-empty subset of `1:ndims` for mixed even/odd extents including an axis of length 2,
self-inverse round-trip, both threaded and serial.


`deps/AbstractOperators/FFTWOperators/src/Shift.jl:302-378`. Both the in-place and out-of-place
kernels loop over `CartesianIndices` and recompute `sum(iseven(I[d]) ? 1 : 0 for d in dirs)` per
element. Measured 43% of a 20-iteration dynamic low-rank solve; a hoisted, `@simd` rewrite
measured C4–1.9x on the kernel alone.

Rewrite: when `dirs` covers a contiguous leading run of dimensions the parity is periodic in the
linear index, so walk the array as slabs — the parity of the outer dimensions is loop-invariant
per slab and the innermost dimension alternates every element, which vectorizes. General shape:
outer loop over `CartesianIndices(sz[maximum(dirs)+1:end])`, per-slab base parity computed once,
inner `@simd` over the leading contiguous block. Keep the existing `@batch` variant as the
threaded branch over the *outer* loop only (never over the innermost run).

- Correctness net: the existing doctests plus a property test asserting the new kernel matches
  `FFTW.fftshift` semantics for every `dirs` subset of a 3-D array with mixed even/odd extents,
  and `alternate_sign!(alternate_sign!(x, dirs), dirs) == x`.
- Measure with `benchmarking/scripts/probe.jl` on 128²×8 and 64²×4×8 `ComplexF32`.

### S3 — `Variation` adjoint is 12x slower than its forward

**Done** (2026-09-03). Rewritten in the forward's own flat/strided idiom
(`_variation_adjoint_dim1!` for dimension 1, mirroring the forward's own whole-array-shift
trick with two corrective passes; `_variation_adjoint_dim!`/`_variation_adjoint_slab!` for
dimensions `2:N`, mirroring the forward's `batch_length`/`k`-slab loop exactly, just
accumulating instead of assigning). Matches the old scalar `_variation_adjoint_term` exactly
across shapes including axis-of-length-2 boundaries (kept as the test reference), passes the
dot-test and the existing dense-transpose test, and is zero-allocating (`@allocated == 0`,
both threaded and serial). Measured on a 128² adjoint: old=247µs, new=24-27µs (**9-10x**);
forward itself (4.6-6.2µs) is untouched.


`deps/AbstractOperators/src/linearoperators/Variation.jl:195-215`. The forward is slab-wise
broadcasts (`@..`/`@views`, `mul!` at `:112-146`); the adjoint is a scalar loop that recomputes
`(cnt-1) ÷ stride % size(y,d)` per element per dimension (`_variation_adjoint_at!`,
`:186-200`). Measured 244 µs vs 21 µs on the benchmark image.

Rewrite the adjoint in the forward's own idiom: accumulate `y` dimension by dimension with
slab-shaped `@views` broadcasts, reproducing the three separable terms documented in
`_variation_adjoint_term`'s docstring (interior `+b_j − b_{j+1}`, first-row `−b_1`, the mirrored
`j == 2` term). Keep `_variation_adjoint_term` as the reference and use it in the test.

- Test: dot-test `⟨Ax, y⟩ == ⟨x, Aᴴy⟩` over 1-D/2-D/3-D including an axis of length exactly 2
  (the boundary case the current docstring records as a fixed `BoundsError`), and an exact
  elementwise comparison against the current scalar implementation, kept in the test file.
- This is TV/TGV/tTV-path work: re-measure those three rows of the accuracy race.

### S4 — Skip the restricting wrapper when there is nothing to restrict (item 0b)

**Done** (2026-09-03). Added `with_restricted_threads_if_needed` (`src/utils.jl`), mirroring
`with_serial_blas`'s own early-out: `LinearAlgebra.BLAS.get_num_threads() == 1` is checked before
entering the scope (every counted pool is always set to the same applied budget by
`NestedThreading._apply!`, so BLAS-is-serial is a valid proxy for the whole scope being a
no-op), and `solve_core.jl`'s gated branch now calls it instead of `with_restricted_threads`
directly. Verified interactively: skips the scope (no state change) when BLAS is already
serial, and still restricts-and-restores correctly otherwise.


`solve_core.jl:60-64` wraps a gated solve in `with_restricted_threads()`, which enters and exits
every counted pool plus the Polyester guard. When the enclosing scope is already serial (the
decomposed `MultiThreadingExecutor` path, or a `-t 1` process) the scope only re-sets what is
already set. Add the cheap early-out — compare the current budget with the target and run `f()`
directly when they match. `with_serial_blas` already does exactly this ("skips the save/restore
when BLAS is already serial", `src/utils.jl` docstring); mirror it.

### S5 — `results = Array{AbstractArray}` (residual observation)

**Done** (2026-09-03), landed with its own commit rather than folded into other
`decomposition.jl` work (both `execute` and `execute_two_phase`'s second `results` array had the
same defect, so both were worth doing together). Both sites now run the first slice/index
outside the loop to learn the concrete result type, mirroring `execute_two_phase`'s existing
`prelim` idiom, and allocate `results` concretely from it. Verified: decomposed reconstruction
(`DirectReconstruction` and a regularized `TotalVariation2D` solve, both multi-slice) still
matches a single-slice reconstruction of the same data exactly.


`src/reconstruction/decomposition.jl:72,179`. `execute_two_phase` already solves the same problem
for `prelim` by running the first item outside the loop to learn the concrete type
(`decomposition.jl:164-171`); apply that idiom to `results` in `execute`, or type it from
`plan.output_size` + `eltype(acq_data.kspace_data)`, which are both known before the loop. Noise
relative to a slice solve, so it goes in only as a tidy-up alongside other `decomposition.jl` work
— do not open a commit for it alone.

---

### S6 — Expose the NFFT operating point (§8)

**Done** (2026-09-03), API only — defaults unmoved (that is `C9`). `get_fourier_operator`
(the `(ksp, image_size, trajectory)` method, its `NamedDimsArray` wrapper, and the
`NonCartesianAcquisitionInfo` method) and `get_encoding_operator(::NonCartesianAcquisitionInfo)`
now take `m`, `sigma`, `precompute` keywords, all defaulting to `nothing` and only forwarded to
`NFFTOp` when set, so the un-parameterized call is provably unchanged. Verified interactively:
default-vs-`nothing`-explicit produces an identical operator; a custom `(m=3, sigma=1.25,
precompute=NFFT.TENSOR)` point actually changes `op.plan.params`; `get_encoding_operator`
forwards the same keywords through to the same `NFFTOp`. Docs: "Non-Cartesian accuracy / speed
trade-off" section added to `docs/src/high-level/performance.md`; test added
(`test/test_encoding_op.jl`, tag `:nfft`).


Not a speed gap: MRT takes NFFT.jl's defaults (`m = 5`, `σ = 2.0`, `POLYNOMIAL`) where MRIReco
hardcodes (`m = 3`, `σ = 1.25`, `TENSOR`) — 360x more forward accuracy (1.6e-7 vs 5.7e-5) for
4.28 ms vs 1.00 ms per coil, accuracy the reconstruction does not use (NRMSE 0.085 either way).
At MRIReco's operating point MRT is 8.35 ms against MRIReco's 47.1 ms. The defect is that MRT's
API gives the caller no way to choose.

- Add keyword pass-through in `get_fourier_operator(ksp, image_size, trajectory; …)`
  (`src/encoding/fourier_operators.jl:166-184`) — `m`, `σ` (`sigma`), `precompute` — forwarded to
  `NFFTOp`/`BatchOp`, with MRT's current values as defaults so nothing moves silently.
- Plumb them from `NonCartesianAcquisitionInfo` (a `nfft_options` field, or a keyword on
  `get_encoding_operator`/`build_encoding_operator`) so a caller reconstructing from an
  `AcquisitionInfo` can set them without hand-building operators.
- **Do not change the defaults here** — that is what keeps this a Phase 1 item. Changing them is a
  separate, measured decision belonging to `C9`:
  run the gridding row of the accuracy race across the (m, σ, precompute) grid and pick the point
  where NRMSE stops moving. If a lower-accuracy default is chosen, it changes every non-Cartesian
  result in the test suite and needs its own commit with the tolerances updated.
- Docs: a "Non-Cartesian accuracy / speed trade-off" section in
  `docs/src/high-level/reconstruction.md` (or `performance.md`) with the measured table, and
  docstring updates on both `get_fourier_operator` methods.
- Tests (`:nfft`): the operator built at the low-accuracy point still passes the adjoint dot-test
  and reconstructs the phantom within a stated, looser tolerance.

---

### S7 — Write down what `--gcthreads` does not do (finding 2, item 0c)

**Done** (2026-09-03). One paragraph added to `docs/src/high-level/performance.md`'s
"Notes for developers" section, alongside the S6 write-up.


`TODO.md` records that the residual 8T cost includes GC growing from ~10 ms to ~35–46 ms per
solve, and that this is **independent of `--gcthreads`** (tested at `--gcthreads=1`, no change).
That is a negative result worth one paragraph in `docs/src/high-level/performance.md`, precisely
because the next person to look at an allocation-heavy serial solve on a `-t 8` process will
reach for that flag first. Documentation only, no code, no measurement — the measurement is
already in `TODO.md`.

### Verification for Phase 1

Per item: the fork's own test suite → `julia_run_testitems` filtered to
`:regularization, :reconstruction, :encoding, :operators, :nfft` → full MRT suite → accuracy race
(`comparison/scripts/run_accuracy_race.jl`) with NRMSE unchanged → `recon_bench.jl` numbers into
`TODO.md`. Runic over `src/` and `test/` in every touched package.

`S5` and `S7` are exempt from the accuracy race (neither touches arithmetic); everything else is
not.

---

# Phase 2 — Complex

Contract changes, cross-package plumbing, measurement-driven decisions and upstream work. Each
item names its prerequisite and what must be settled before the edit.

### C1 — `SqrNormL2WithNormalOp` reports the wrong function value (correctness)

**Done** (2026-09-03, `c36c3b8`). Fixed as described below: `Aᴴd` (the normal operator's
displacement, which `get_normal_op(::AffineAdd)` already carries) and `‖d‖²/2` are computed
once at construction, and `gradient!` returns
`½Re⟨x, y⟩ + λ/2(Re⟨x, Aᴴd⟩ + ‖d‖²)` — one extra `dot` when `L` is affine, none at all when
it is purely linear (`Aᴴd` is `nothing` there, so the correction is skipped rather than
paying a dot against zeros). The array-λ methods were **removed** rather than fixed: a
weighted `½Σλ_k|y_k|²` has gradient `Aᴴ(λ⊙Ax)`, which a normal operator cannot express, and
what the code computed was `λ⊙(AᴴA x)` with `λ` indexed over the *domain* — not the gradient
of any such `f`. Nothing in this repository constructed one. Test added over
λ ∈ {1, 0.75} × {real, complex} × {linear, affine}, against the closed form and against a
finite-difference gradient.

The `g_z` half of this entry (below) is **not done** — measure it before acting.


`deps/StructuredOptimization/src/calculus/sqrNormL2WithNormalOp.jl:68-88`. Both `gradient!`
methods compute `y = λ·AᴴA·x` and then return `λ/2 · Σ|y_k|²` — that is `‖∇f(x)‖²/2`, not
`f(x) = ½‖A x − b‖²`. `ProximalCore.value_and_gradient!` contracts that `gradient!` returns the
value, so `FastForwardBackwardState.f_x` (`fast_forward_backward.jl:132`) and everything reading
it is wrong. It is silent today only because MRT runs FISTA with `adaptive = false` and
`tol = 0`, so `f_x` reaches nothing but the verbose display; turn on backtracking and the
line-search test compares garbage.

Fix, and it costs nothing: with `A = AffineAdd(𝒜, −b)`, `AᴴA` is likewise an `AffineAdd` whose
displacement is `−Aᴴb`, so `y = λ(𝒜ᴴ𝒜x − 𝒜ᴴb)` is already in hand and
`f(x) = ½⟨x, 𝒜ᴴ𝒜x⟩ − Re⟨x, 𝒜ᴴb⟩ + ½‖b‖²` needs one `dot(x, y)`, one `dot(x, 𝒜ᴴb)` and a
constant. Store `½‖b‖²` in the struct at construction (it is a fixed scalar) and keep `𝒜ᴴb`
— `AffineAdd` already holds it. New field, so the constructor and the `{T,SC,L,L2}` parameter
list change; `parse.jl:458` reads `.A`/`.lambda` only, so nothing downstream moves.

- Test (`deps/StructuredOptimization/test/`): for a random small `A`, `b`, assert
  `value_and_gradient!` agrees with `f(x)` evaluated through the callable form and with a
  finite-difference gradient. That test would have caught this.
- Then re-check the §7 claim "`FastForwardBackwardIteration` evaluates an extra `A` application
  nothing consumes": as written it is **wrong for `f`** — `gradient!` goes through `AᴴA` and
  computes value and gradient in one pass. What *is* real is `g_z`: `prox!` returns the
  regularizer's value on every iteration (`fast_forward_backward.jl:134`), an extra full pass
  over `z` (plus, for a wavelet term, an extra transform) that only the display consumes when
  `tol = 0` and `adaptive = false`. Measure it before acting; if it is above ~2% of the solve,
  add a `ProximalCore.prox!`-without-value path selected by an `iter.needs_g_value` flag set
  from `adaptive || verbose || tol > 0`.

### C2 — Cancel the sign-alternation pair inside `𝒜ᴴ𝒜` (§9, second cause)

**Done** (2026-09-03, `1f5c343`, plus the `Compose` fix `ec75b5c` it exposed). The rule went in
as written, with `L.dim_in == R.dim_in` added to the guard. Two corrections to the analysis
below, both found by probing the real operator rather than trusting the sketch:

- The `±` sits on the **k-space** side, not between `𝒮` and `ℱ`: the actual chain is
  `𝒜 = 𝒫 ∘ ± ∘ ℱ ∘ 𝒮 ∘ broadcast`. That is what makes the rule fire at all — had the `±`
  been where this entry originally said, the pair in `𝒜ᴴ𝒜` would have been separated by
  `ℱ, 𝒫ᴴ𝒫, ℱᴴ` (three operators, and not diagonal), and `± C ± = C` would have been *false*:
  conjugating a convolution by a modulation shifts its kernel. The rule as stated is only
  correct for a diagonal middle operator, which is exactly what the real chain has.
- `is_diagonal(::SignAlternation)` is still `false`, which is wrong (it is a ±1 diagonal), but
  fixing it also changes `_is_dft_op`'s `all(is_diagonal, ...)` test and hence which shifts
  `fftshift_op` converts. Left alone deliberately; noted here as a separate, unscheduled item.

Measured on the real chain: `get_normal_op(𝒜)` goes from 9 operators to 7, both
`SignAlternation`s gone. `recon_bench` rows, 1 thread, min of 3, C2 reverted vs applied with
`C1`/`C3` present in both:

| row | without C2 | with C2 | |
|---|---|---|---|
| CG-SENSE (10 it) | 94.66 ms | 46.05 ms | **2.06x** |
| Global Low-Rank (20 it) | 247.48 ms | 173.99 ms | 1.42x |
| Temporal TV (20 it) | 877.07 ms | 724.52 ms | 1.21x |
| Locally Low-Rank (20 it) | 250.38 ms | 224.06 ms | 1.12x |
| Total Variation (30 it) | 1490.84 ms | 1365.77 ms | 1.09x |
| L1-Wavelet (30 it) | 293.67 ms | 285.87 ms | 1.03x |
| TGV (30 it) | 2256.61 ms | 2239.83 ms | 1.01x |

The win tracks how much of a solve is normal-operator applications, which is why CG-SENSE
gains most and TGV least. NRMSE agrees to five significant figures on every row but is not
bit-identical: multiplying by ±1 *is* exact, but removing the pair changes buffer aliasing and
therefore the alignment-dependent FFTW code path.

**The `Compose` bug this exposed** (`ec75b5c`, its own commit): the constructor's
buffer-reallocation branch — the one that replaces a buffer when the removal makes two aliased
buffers adjacent — re-sliced the *already shortened* buffer tuple with `next_i`, the
pre-removal index. Right for a pairwise combination, off by one for a triple, producing a
`Compose{N, M}` with `M != N - 1`. Nothing rejected it: the entry check runs before the
combination loop, and `mul!` is `@generated` over `M`, so it applies `A[1:M]` then `A[N]` and
silently **skips** everything between — on the MRT operator, `𝒜ᴴ𝒜` losing its
sensitivity-map factor. Now spliced at the right index, and the constructor re-checks the
invariant after the loop. The existing `ShiftOp`/`SignAlternation`/`DFT` triples could reach
the same branch; nothing had.


The Cartesian encoding operator is `𝒜 = 𝒫 ∘ (ℱ ∘ ±) ∘ 𝒮` (`get_fourier_operator`
wraps the `DFT` with `ifftshift_op`, which becomes a `SignAlternation` on the other side when the
shifted extents are even — `Shift.jl:487-535`). `get_normal_op(::Compose)`
(`deps/AbstractOperators/src/calculus/Compose.jl:225-235`) folds only the outermost factor,
giving `(𝒮, ±, ℱ, 𝒫ᴴ𝒫, ℱᴴ, ±, 𝒮ᴴ)`; the `Compose` constructor's cancellation loop
(`Compose.jl:37-90`) only looks at *adjacent* pairs, so the `± … ±` pair separated by the
diagonal mask survives and every normal-op application pays two full sign passes it does not owe.

`𝒫ᴴ𝒫` is `NormalGetIndex`, and `is_diagonal(::NormalGetIndex) = true`
(`GetIndex.jl:201`). A sign alternation is a real ±1 diagonal, and diagonals commute, so
`± M ± = M` exactly whenever `M` is diagonal and both alternations carry the same `dirs`.

The constructor already has the hook: `can_be_combined(L, M, R)` / `combine(L, M, R)`
(`properties.jl:345,379`, dispatched at `Compose.jl:54`). Add to `FFTWOperators`:

```julia
can_be_combined(L::SignAlternation, M::AbstractOperator, R::SignAlternation) =
    L.dirs == R.dirs && is_linear(M) && is_diagonal(M) && size(M, 1) == size(M, 2)
combine(::SignAlternation, M::AbstractOperator, ::SignAlternation) = M
```

with the same pair for `FFTShift`/`IFFTShift` (`P M P⁻¹` is a *permuted* diagonal, **not** `M` —
so those get no rule; only `SignAlternation` is self-inverse *and* commuting). Note the argument
order convention: `can_be_combined(A[i+2], A[i+1], A[i])`, i.e. `(last, middle, first)` in
application order, which for this rule is symmetric anyway.

- Risk: the rule fires on any diagonal middle operator, including a `DiagOp` of sensitivity
  weights, which is also correct. The guard that matters is `L.dirs == R.dirs`; assert it in a
  test with mismatched `dirs` that must *not* combine.
- Tests (`deps/AbstractOperators/FFTWOperators/test/`): build `𝒫 ∘ ℱ ∘ ± ∘ 𝒮` for a real MRT-shaped
  problem, take `get_normal_op`, assert (a) the resulting `Compose` contains no `SignAlternation`,
  (b) `AᴴA * x` matches `A' * (A * x)` to `eps`-level, (c) an operator count regression test so a
  future `Compose` change cannot silently un-fuse it.
- Expected, from §9: dynamic low-rank 890 ms → ≈550 ms together with S2. Verify both separately
  so the attribution is real.

### C3 — `𝒜ᴴ𝒜` is built twice for every ADMM solve

**Done** (2026-09-03, `274b625`), as sketched: `LeastSquaresTerm` gained an optional `AHA`
symbol so only algorithms that actually form the normal operator ask for one (ADMM does; the
CG family does not and is unchanged), `ADMMIteration` gained an `AHA` field used by
`get_cg_operator`, and `prepare` emits `remove_displacement(f.AᴴA)` — point 3's question
answered: ADMM wants the linear part, since it carries `b` separately, and that is the same
`remove_displacement` already applied to `op` two lines up. Skipped when `lambda != 1`, as
the entry proposed.

**The measured saving is much smaller than §7 claims.** On the benchmark TV problem
(128²×8, 2× undersampled), one-outer-iteration solve, min of 3:

| | allocations | wall |
|---|---|---|
| without C3 | 78.18 MiB | 144.53 ms |
| with C3 | 74.19 MiB | 136.91 ms |

— ~4 MiB and ~7.6 ms, not the "57 MiB for a single-iteration solve" in `TODO.md`. Over 30
iterations it is 245.05 → 241.13 MiB and the wall time is indistinguishable (1378 vs 1381 ms,
inside run-to-run noise). So this is a setup-cost item, and it matters to `C5` (which reuses
the same plumbing from the other end) rather than to steady-state throughput. The 57 MiB
figure should be treated as retracted unless someone reproduces it on a different problem.


`SqrNormL2WithNormalOp` eagerly builds `AᴴA = A' * A` in its constructor
(`sqrNormL2WithNormalOp.jl:40`), then `parse.jl:458-461` hands ADMM only `f.A`, and
`admm.jl:265` builds `iter.A' * iter.A` all over again — two independent `Compose` chains, each
with its own k-space-sized buffers (57 MiB for a single-iteration solve).

Thread the existing one through:

1. `prepare(term, ::LeastSquaresTerm, …)` (`parse.jl:450-477`) additionally returns the
   precomputed normal operator under a new assumption key (e.g. `assumption.AHA`), populated only
   when `f isa SqrNormL2WithNormalOp` and `lambda == 1` (a `lambda != 1` rescales `op`, so the
   cached `AᴴA` no longer matches — either skip the cache there or scale it, and scaling a
   `Compose` is not free; skip is fine, `lambda == 1` is the MRT path).
2. `ADMMIteration` gains an optional `AHA` field defaulting to `nothing`;
   `get_cg_operator` (`admm.jl:262-269`) uses it when present instead of recomputing.
3. `remove_displacement`/`displacement` handling must stay consistent: the cached `AᴴA` carries
   the `AffineAdd` displacement, ADMM wants the linear part plus `AHb` separately. Verify which
   form `admm.jl` actually needs and cache that one, not the other.

- Test: allocation regression test on a one-iteration ADMM solve (`@allocated` under a fixed
  problem), plus an existing-solution equivalence test.

### C4 — ADMM per-iteration dead work

**Dropped** (2026-09-04) — measured, and the whole item is worth ≈0.25% of a solve, against the
2% floor this entry set for itself. No code change was kept; `admm.jl` is untouched.

Measurements on the benchmark node (8 threads, `benchmarking/src/ReconBench.jl` cases):

| | cost | share of the 30-iteration TV solve (~1300 ms) |
|---|---|---|
| one-trip `Threads.@threads` fork/join | 9.66 µs (inline body: 31 ns), ×2 per outer iteration | 0.58 ms — 0.04% |
| the six residual norms + the extra `Bᴴ` | `Bᴴ` 31.0 µs, `norm(Bx)` 12.6 µs ×4, `norm(x)` 6.2 µs ×2 = 94 µs per outer iteration | 2.81 ms — 0.2% |

The loop change was also implemented and A/B'd against `HEAD` end-to-end (three interleaved
rounds, hot-swapped through Revise inside one session, min-of-3 each, on TV / TGV / Temporal TV).
It produced no signal at all: run-to-run spread on this node is ±30-60% (Temporal TV ranged
817–1301 ms across six identical measurements), i.e. two orders of magnitude larger than the
0.04% being chased. The naive before/after run that preceded it appeared to show TV −6.3% and
Temporal TV −6.8% — pure node noise, and a good example of why the interleaved form is the only
usable one here.

Two corrections to the entry's premises, both found while measuring:

- The residual norms are **not** dead under MRT's defaults, and not for the reason given. MRT's
  default penalty is `SpectralRadiusApproximationPenalty`, whose `get_next_rho!` derives ρ from
  `u` and `z` directly and never reads `rᵏ_norm`/`sᵏ_norm`/`ϵᵖʳⁱ`/`ϵᵈᵘᵃ` — only
  `ResidualBalancingPenalty` and `WohlbergPenalty` do. What keeps them live is `tol`:
  `Config.tol` defaults to `1e-4` and `default_stopping_criterion` consumes all four vectors, so
  the proposed `needs_residuals` gate would be `true` on every MRT reconstruction. It could only
  ever fire for a caller who passes `tol = 0` with a fixed ρ — and buy them 0.2%.
- The `Bᴴ` application is 31 µs against a ~3.5 ms `𝒜`+`𝒜ᴴ` pair, so "7 vector passes out of ~10
  normal-op applications" overstates it by ~30x: the residual passes are over image- and
  `Bx`-sized arrays, not k-space-sized ones.

Original entry, for the record:

`admm.jl:376,403` and `:418-428`.

- Both `Threads.@threads for i in eachindex(iter.g)` loops are one-trip with a single
  regularizer, and pay a fork/join each. Replace with a plain loop when `length(iter.g) == 1`
  — cheapest form is a `@budgeted_threads`-style helper or simply
  `if length(iter.g) == 1 … else Threads.@threads … end`; keep both branches calling one shared
  body function so they cannot drift.
- The six residual norms and the extra `Bᴴ` application per outer iteration are dead under a
  `FixedPenalty` with `tol = 0` — nothing reads `rᵏ_norm`, `sᵏ_norm`, `ϵᵖʳⁱ`, `ϵᵈᵘᵃ`. They are
  *not* dead under an adaptive penalty (`get_next_rho!` consumes them) or a nonzero `tol`
  (`default_stopping_criterion`) or a verbose run. Gate on a flag computed once in `ADMMState`:
  `needs_residuals = !(iter.penalty_sequence isa FixedPenalty) || tol > 0 || verbose`. The
  `tol`/`verbose` values live in the stopping criterion closure, not the iteration, so this needs
  a small plumbing change — pass `needs_residuals` into `ADMMIteration` from
  `patch_algorithm_with_default_values` (`src/reconstruction/build_model.jl:57-78`), which already
  knows MRT's ADMM defaults.
- Measure first: on the benchmark TV solve this is ~7 vector passes out of ~10 normal-op
  applications per outer iteration, so the ceiling is a few percent, not a factor. If it measures
  under 2%, record that and drop the item rather than adding a flag.

## C5 — Setup cost: the operator norm (§7)

§7 measures ~137 ms of a 322 ms L1-wavelet solve in `normalize_op`, against MRIReco's ~7 ms for
the same information. `mrt-perf-opportunities` in project memory records the same 144 ms as
"MRT's whole setup gap". Three separate costs are tangled there and the first step is to
separate them.

### C5.1 — Measure the decomposition (no code change)

**Done** (2026-09-03). Measured on the L1-wavelet benchmark problem (128²×8, 2× undersampled),
1 thread, *after* `S2`/`C2`, so against the baseline that actually ships:

| | |
|---|---|
| (a) power iterations run | **20 of 20** — the `tol = 1e-3` exit never fires |
| (b) `AHA = A' * A` inside `powerit` | **0.41 ms** |
| one `AᴴA` application | 4.09 ms (× 20 = 81.8 ms) |
| (c) the `1/L * A` `Scale` wrapper | **+4.3%** per application |
| (d) `normalize_op` total | **72.40 ms** = 24.7% of the 293 ms solve |

§7's 137 ms is now 72 ms, because `S2` and `C2` made each of the 20 applications cheaper.

**(b) is not the lever**: 0.41 ms of 72 ms. The first branch of `C5.2` — "do not build the
normal operator twice" — is therefore **dropped**, and `C3`'s own measurement says the same
thing from the other end (~4 MiB, ~7.6 ms of setup). The cost is (a): twenty applications that
never converge.

**Why they never converge, and what that means for raising `tol`.** The trajectory (relative
error against a 60-iteration reference, three start vectors):

| k | 1 | 2 | 3 | 5 | 8 | 12 | 20 | 40 |
|---|---|---|---|---|---|---|---|---|
| 2× undersampled | 2.4e-1 | 1.0e-1 | 7.0e-2 | 4.2e-2 | 2.4e-2 | 1.4e-2 | 6.4e-3 – 7.1e-3 | 1.3e-3 – 1.7e-3 |
| fully sampled | 7.0e-2 | 4.6e-2 | 3.3e-2 | 2.0e-2 | 1.2e-2 | 7.2e-3 | 3.7e-3 – 4.0e-3 | 0.9e-3 – 1.1e-3 |

The top of the spectrum is close to degenerate, so convergence is linear and slow: at
`maxit = 20` the estimate is still 0.4–0.7% **below** the true norm, and the iterates approach
it from below always. The `tol = 1e-3` rule would first fire at iteration 22 (undersampled) or
16 (fully sampled) — i.e. `maxit` binds first, which is why (a) reads 20 of 20.

**And this is why "just raise `tol`" is not a free speedup.** MRT's convention makes `L` part
of the *problem*, not a preconditioner: it solves `½‖(𝒜/L)x − y‖² + R(x)`, and the equivalent
unnormalized form is `½‖𝒜x − Ly‖² + L²R(x)` — still `L`-dependent. A smaller `L` (fewer
iterations ⇒ larger under-estimate) both weakens the effective regularization by `L²` and
makes `Lf = n` an *under*-estimate of the true Lipschitz constant, which is the direction that
makes FISTA diverge. Cutting to 5 iterations would save ~54 ms of a 293 ms solve (18%) at the
price of a ~1% shift in `L` and hence ~2% in the effective `λ`. That is a recalibration of
every tuned `λ` in the tests, benchmarks and accuracy race — a deliberate decision with its own
re-baselining, not something to slip in under a performance item. **Left at 20 iterations.**

**What the measurement did find** (`c8f9a41`): `powerit` drew its start vector from the
**global RNG**, and since the iteration does not converge, that vector leaks into the *result*.
Measured: `estimate_opnorm` on one MRT encoding operator varied 6.5e-4 relative across six
calls in one session, and two otherwise identical `reconstruct` calls differed by 7.9e-4
relative. MRT reconstructions were not reproducible. `powerit`/`estimate_opnorm` now take an
`rng` keyword defaulting to a fresh fixed-seed generator, so the estimate is a deterministic
function of the operator and neither depends on nor consumes the global RNG. Verified: repeated
reconstructions are now bitwise identical. This also removes a confound from every measurement
in this file — the NRMSE wobble in the last digits of the `C2` table above is this.


`normalize_op` (`src/utils.jl:7-15`) → `estimate_opnorm` (`properties.jl:444`) → `powerit`
(`:452`). Instrument one L1-wavelet solve and record, separately:
(a) how many power iterations actually run before the `tol = 1e-3` early exit fires — §7 asserts
all 20, and the loop *does* have an early exit, so either the assertion or the exit is wrong;
(b) the cost of `AHA = A' * A` inside `powerit` (a full `Compose` build with k-space-sized
buffers, thrown away on return);
(c) the cost of the `1/L * A` `Scale` wrapper per application for the rest of the solve.

Everything below is conditional on what (a)–(c) say. Do not implement C5.2/C5.3 before this.

### C5.2 — Do not build the normal operator twice, and do not throw it away

**Dropped** (2026-09-03), on `C5.1`'s numbers: (b) is 0.41 ms of a 72 ms `normalize_op`, so
plumbing the normal operator into `estimate_opnorm` buys nothing measurable. The iteration
count *is* the cost, and raising `tol` to cut it changes the solved problem — see `C5.1`.


If (b) is significant: give `estimate_opnorm`/`powerit` an optional `normal_op` keyword, build
`𝒜ᴴ𝒜` once in `_iterative_reconstruct_core` (`src/reconstruction/solve_core.jl:21-25`), pass it to
the estimate, and hand the same object to the model builder so `SqrNormL2WithNormalOp` does not
rebuild it (this is the same plumbing as C3, from the other end — do C3 first and reuse it).
Note `get_normal_op(::Scale)` already forwards to the inner operator (`Scale.jl:128-131`), so the
normalized operator's normal op *is* the unnormalized one up to a real scalar; that identity is
what makes the reuse legal.

If (a) says the iteration count is the cost: raise `tol` from `1e-3` toward MRIReco's operating
point only if the accuracy race is unmoved — `Lf` too small makes FISTA diverge, too large only
makes it slow, so the safe direction is a deliberate over-estimate (`λ · (1 + tol)`), and that
should be written down where the tolerance is set.

### C5.3 — Do not normalize when only `Lf` is wanted — carefully

**Done** (2026-09-03), and this entry's framing was wrong in a way worth recording, because the
error was hiding a bug.

The entry says dropping the normalization "gives `½‖𝒜x − y‖² + R(x)`, which is a *different
problem* — the regularization is weaker by `L²`". The direction is right, the exponent is not,
and the important half was missed. Substituting `x = Lv` in what MRT actually solved gives

```
½‖(𝒜/L)x − y‖² + R(x)  =  L²·[ ½‖𝒜v − y‖² + L·λ‖Ψv‖₁ ]      (degree-1 homogeneous R)
```

so the effective weight was `λ·L`, not `λ·L²` — **and the returned image was `L` times the data's
own units**. Measured, as `λ → 0`: `‖x‖/‖x_true‖ = 1.5214` against `L = 1.5214`, for both TV and
L1-wavelet. Every benchmark uses amplitude-aligned NRMSE, which is why nobody saw it. Verified the
identity directly: the shipped result agreed with `L ×` (unnormalized solve at `λ·L`) to 7.6e-5.

So the choice was not "exact reformulation vs. skip" but "which of two conventions", and one of
them was returning images in arbitrary units. Taken: **`𝒜` is no longer rescaled at all.**
`_iterative_reconstruct_core` estimates `L = ‖𝒜‖` purely as a step size and passes `Lf = n·L²`;
the problem solved is `½‖𝒜x − y‖² + R(x)`. No `scale_regularization` guard is needed — that was
only required to keep the *old* λ convention while dropping the wrapper, and the old convention is
what we are leaving behind.

- The amplitude bug is fixed: the default path now reproduces the
  `disable_operator_normalization = true` path exactly (ratio 1.0000 on every row tested).
- `λ` no longer carries an `‖𝒜‖` factor. That factor was not benign: `L` is insensitive to matrix
  size (1.5248/1.5250/1.5253 at 64²/128²/256²) and to undersampling (1.5179–1.5253), but scales
  **linearly with the sensitivity maps' own scaling** (×3 ⇒ `L` ×3, exactly) and varies with coil
  count (1.5250 at 8 coils vs 1.0872 at 4). The same `λ` regularized ~40% harder with 8 coils
  than 4.
- Migration, documented in `docs/src/high-level/methods.md` ("Operator norm, step size and λ") and
  in `IterativeReconstruction`'s docstring: a `λ` tuned against the old behaviour reproduces it as
  `λ·L`.

Measured (1 thread, min of 3), against the `C1`/`C2`/`C3` baseline — more than `C5.1`'s +4.3%,
because the `Scale` wrapper is gone from every application *and* the solve path changes:

| row | before | after |
|---|---|---|
| Total Variation (30 it) | 1365.77 ms | 1178.08 ms (−13.7%) |
| Locally Low-Rank (20 it) | 224.06 ms | 199.24 ms (−11.1%) |
| L1-Wavelet (30 it) | 285.87 ms | 262.10 ms (−8.3%) |
| TGV (30 it) | 2239.83 ms | 2083.52 ms (−7.0%) |
| CG-SENSE (10 it) | 46.05 ms | 43.58 ms |

**NRMSE moves on the regularized rows** (TV 0.0116 → 0.0076, L1-wavelet 0.0084 → 0.0058, TGV
0.0097 → 0.0062) and this is *not* an accuracy improvement to claim: those benchmark `λ` now act
`L ≈ 1.52×` weaker. The tables must be re-baselined against the new convention in `C9`, and the
"NRMSE is the gate" ground rule does not apply to this one commit — it is a deliberate change of
what problem is being solved, which is the exception that rule allows for.

`normalize_op` (`src/utils.jl`) had no other caller and is removed; the `benchmark/benchmarks.jl`
entry that timed it now times `estimate_opnorm`, which is the cost that actually remains.


§7's framing ("FISTA only needs the number `Lf = ‖A‖²`, not a normalized operator") is right about
FISTA and **wrong about the objective**: MRT solves `½‖(𝒜/L)x − y‖² + R(x)`, and dropping the
normalization gives `½‖𝒜x − y‖² + R(x)`, which is a *different problem* — the regularization is
weaker by `L²`. Every calibrated `λ` in the tests, the benchmarks and the accuracy race is tied to
the normalized convention. So this is only safe as an exactly-equivalent reformulation:
solve with unnormalized `𝒜`, `b' = L·y` and `Lf = L²`, and scale each regularization term by `L²`
— which needs `scale_regularization` on every term in the problem, and that method exists only for
homogeneous terms (`AGENTS.md`, "Adding a regularizer").

Decision to take before writing code: either (i) restrict the unnormalized path to problems whose
regularizers all implement `scale_regularization`, falling back to today's path otherwise — a
correctness-preserving optimization with a trait guard; or (ii) skip C5.3 entirely and take only
C5.2's win. Recommend deciding *after* C5.1: if C5.2 alone closes most of the 137 ms, (ii) is the
right answer and this entry becomes a documented non-goal.

### Verification for C5

Accuracy race must be identical, not merely close — a changed `Lf` changes the iterate path.
Report before/after wall time for the L1-wavelet row (target: beat MRIReco's 91 ms/20 it or state
why not) and the TV row.

**Outcome**: the accuracy race cannot be identical here and is not meant to be — `C5.3` changes
the convention `λ` is expressed in, so the regularized rows move by construction (see its table).
That is the exception the ground rule allows, and `C9` re-baselines the tables. What *is* now
checkable is run-to-run identity: before the `powerit` fix the race could not reproduce itself
across two runs of the same code, because `L` was drawn from the global RNG.

L1-wavelet: 285.87 → 262.10 ms/30 it. Still short of MRIReco's 91 ms/20 it, and the reason is
recorded above — `estimate_opnorm` is 72 ms of it, the twenty power iterations are what that
buys, and cutting them changes `L`, which is now purely a step size but still one that a
too-small value makes divergent. Raising `tol` is therefore a safe *over*-estimate away
(`λ·(1+tol)`), and is the remaining lever if this row needs to come down.

---

## C6 — The threading residuals (finding 2 items 0–3, finding 3 "Not done")

The collapses are fixed; what is left is a 10–25% residual at `-t 8` and three known gaps in
where the gate applies.

### C6.1 — Size-gate the sequential executor (finding 2, item 3 — the real gap)

**Done** (2026-09-04). Both sites now call `slice_threading(plan, acq_data, config, executor)`
(`src/reconstruction/decomposition.jl`): a `MultiThreadingExecutor` still hard-returns `false`,
and otherwise the decision goes through `_should_thread_work_item(config, bytes)` —
extracted from `maybe_disable_undecomposed_threading` as the entry asked, so the decomposed and
undecomposed paths now share one predicate — applied to `slice_bytes(plan, acq_data)`
(`plan.variable_size` with the batch dimensions collapsed to one, times the k-space element
size). Nothing is measured at run time.

The same decision is also handed to `for_each_item!`'s `SequentialExecutor` method, which
previously read `config.threaded` directly: opening every pool with `with_full_threads` around a
slice loop whose body has just been gated serial is exactly the dead weight the gate exists to
remove, so the loop scope now follows `slice_threaded` too.

Measured on a 2-slice 128²×4 TV solve, 20 iterations, `-t 8`, interleaved rounds of min-of-3
(interleaved because this node's run-to-run spread is what killed `C4`): the gated build won
**11 of 12** rounds, with round medians 602 ms → 548 ms (**−9%**) on the noisiest campaign and
594/566/608 ms → 521/492/557 ms (**−13%**, disjoint ranges) on the quietest. Results are
bit-identical with `threaded = true` and `threaded = false` (`maximum(abs, Δ) == 0`), which is
the point: the gate is a performance decision only.

Test added (`test/test_reconstruction_integration.jl`, `:reconstruction`): asserts
`slice_bytes`, both executors' `slice_threading` verdicts at 128² (false) and at a synthetic
2048² plan (true under the sequential executor, false under the multi-threading one and false
with `threaded = false`), plus result identity across `threaded`.

`C6.3`'s documentation half also went in here, since it is one paragraph in the same file: the
`with_serial_blas` docstring now records *why* the gate keys on the work item rather than on
batch width (the two disagree only for a wide batch of large items, where the budget wins over
this scope anyway — the conservative direction).


`run_slices!` (`src/reconstruction/decomposition.jl:99-112`) and `execute_two_phase` (`:172-186`)
both set `slice_threaded = executor isa MultiThreadingExecutor ? false : config.threaded`. For the
`SequentialExecutor` (few slices, `length(plan) ≤ nthreads()`) that leaves `true`, and
`@conditionally_enable_threading` then opens every pool around a small per-slice work item. The
inner `with_restricted_threads` in `solve_core.jl` covers the *solve*, but the per-slice operator
build, the adjoint and the opnorm still run threaded.

Fix: reuse the existing predicate rather than inventing a second one. Extract from
`maybe_disable_undecomposed_threading` (`src/utils.jl:188`) a small
`_should_thread_work_item(config, bytes)` and call it in both places to compute `slice_threaded`,
using `plan.variable_size`/element type for the per-slice byte count (the plan knows the slice
shape; nothing needs to be measured at run time).

- Test (`:reconstruction`): a 2-slice 128² problem takes the sequential executor and reconstructs
  identically with `threaded = true` and `threaded = false`; assert on the *result*, and assert
  the chosen `slice_threaded` through a small internal accessor rather than by timing.

### C6.2 — Per-operator size vetoes beyond FFTW (item 2)

**Done** (2026-09-04) — and the entry's premise was stale in both directions.

The machinery it asks for already exists in the vendored `AbstractOperators`:
`src/threading_policy.jl` defines `threading_threshold(::Type{Op})` per operator, consulted
through `_resolve_threaded`, whose contract is that `threaded = true` is a **permission**, not a
command — the size policy still has the final say, and `is_threaded(L)` reports what the
operator will actually do. `Variation` (`2^10`), `FiniteDiff` (`2^16`), `DiagOp` (`2^17`),
`Scale` (`2^22`) and the elementwise operators all carry measured values. `WaveletOperators`
needs none: it has no Julia-level threading at all (`is_threaded(::WaveletOp) = false`, the work
is inside `Wavelets.jl`), so there is nothing to veto there.

What was actually wrong is that **`Variation`'s threshold was fitted before `S3`**, which
rewrote its adjoint in the forward's strided idiom and made the serial adjoint ~10x faster —
moving the crossover by seven powers of two. Re-measured with the package's own
`benchmark/operator_thresholds.jl` on the `--exclusive` `test` node (EPYC 7763, 8 threads, BLAS
serial), `serial / threaded` for the forward+adjoint pair:

| n | 2^10 | 2^12 | 2^14 | 2^15 | 2^16 | 2^17 | 2^19 | 2^22 |
|---|---|---|---|---|---|---|---|---|
| square, Float32 | 0.17x | 0.36x | 0.59x | 0.92x | 0.94x | **1.30x** | 1.50x | 2.47x |
| sliver, Float32 | 0.09x | 0.21x | 0.47x | 0.68x | 0.81x | **1.16x** | 1.36x | 1.95x |

So at the old `2^10` threading this operator cost up to **5.9x**, and it was a loss at every
size up to `2^16` — which covers the 128²-256² images a TV term is normally applied to.
`threading_threshold(::Type{<:Variation})` is now `2^17` (conservative across both shapes and
both element types, per the script's own rule), with the provenance comment rewritten and a
test in `deps/AbstractOperators/test/test_threading_policy.jl`.

Two further notes from the same run:

- `operator_thresholds.jl`'s `_build_variation` swept only an `(n/4, 4)` sliver. `Variation`
  makes one strided pass per dimension, so its cost is shape-sensitive and the shape it is
  actually used on is a square image; the script now sweeps both (`Variation`,
  `VariationSliver`). The two agree here, which is itself worth knowing.
- The script's suggested `threading_threshold(::Type{<:DiagOp}) = 2^15` was **not** taken. Its
  `crossover` rule takes the smallest size where threaded wins and keeps winning, which at
  `2^15` is a 1.00-1.01x tie; the real jump is 6.5x at `2^17`, exactly where the current value
  sits. `FiniteDiff`'s `2^16` was likewise confirmed unchanged.

Effect on MRT itself: **none measurable**, and for a reason worth recording — a 128²-256²
solve is already forced serial as a whole by `maybe_disable_undecomposed_threading`/`C6.1`, so
`Variation` never sees `threaded = true` at those sizes through MRT's own path (measured: TV
30 it, `-t 8`, four interleaved rounds, 1744-1793 ms before vs 1750-1771 ms after). The fix
matters for direct users of the operator and for any MRT path that reaches a mid-size
`Variation` with threading permitted — and it removes a 5.9x trap from the library.


Only FFTW consults a threshold (`_fftw_num_threads` / `fftw_threading_threshold`). Add the same
shape of veto to `DSPOperators` (the `Variation`/`FiniteDiff` `@batch` loops), `WaveletOperators`,
and the sensitivity `DiagOp`: a per-operator `threading_threshold(::T)` consulted where `threaded`
is read, so a large-volume solve can still thread the outer FFTs while a small stencil stays
serial. Less urgent now that whole small solves go serial — schedule after C6.1/S4 and only if
the large-volume benchmark (below) shows it.

### C6.3 — Re-fit `SERIAL_BLAS_THRESHOLD_BYTES` on a real solve (finding 3, "Not done")

**Done** (2026-09-04). Outcome: **the second one the entry allows** — the crossover is not a
single number, so the constant became a documented tunable and the shipped default stayed at
16 MiB.

Re-fitted with a new script, `benchmarking/scripts/serial_blas_threshold_sweep.jl`, on the
`--exclusive` `test` node (`x1001c4s3b0n1`), 8 threads, TV and temporal-TV solves at
10 iterations, A/B interleaved, min-of-2-of-2. No constant has to be redefined to run the A/B:
the threshold has exactly two effects and both are all-or-nothing per solve (below it,
`maybe_disable_undecomposed_threading`/`slice_threading` force `threaded = false`; above it,
`solve_core.jl` takes the `with_serial_blas` branch, which then also declines to narrow because
the item is over the same threshold), so `threaded = true` vs `false` *is* the A/B.

`serial / threaded` wall time — above 1 means threading won:

| case | item | OpenBLAS | MKL |
|---|---|---|---|
| TV 128² | 0.12 MiB | 1.00x | 1.01x |
| TV 256² | 0.50 MiB | 1.02x | 1.02x |
| TV 512² | 2 MiB | 1.02x | 1.00x |
| tTV 128²×16 | 2 MiB | 1.01x | 1.01x |
| tTV 128²×32 | 4 MiB | 1.00x | 1.00x |
| tTV 256²×16 | 8 MiB | **0.76x** | **1.08x** |
| TV 1024² | 8 MiB | **0.88x** | **1.38x** |
| tTV 256²×32 | 16 MiB | 1.13x | 1.35x |
| tTV 512²×32 | 64 MiB | 1.66x | 1.67x |

Two findings, neither of which a single re-fitted number could carry:

1. **Below 4 MiB the choice does not matter at all** on either backend (within ±2%, i.e. inside
   this node's noise). The whole "small solves must be serial" result that motivated the
   constant is a *wide-batch* effect — many slabs in flight, which is the decomposed path — and
   `C6.1` is what applies it there. For a single small solve the gate is nearly free either way.
2. **Between 4 and 16 MiB the backends disagree in direction.** Threading an 8 MiB solve is a
   1.3x *loss* on OpenBLAS and a 1.4x *win* on MKL, on the same node, same problem. That is
   larger than the octave the entry hoped to bracket, and no single constant is right for both.

So: `SERIAL_BLAS_THRESHOLD_BYTES` is now `DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES` (unchanged at
16 MiB, the value that is safe on both backends) plus `serial_blas_threshold_bytes()`,
`set_serial_blas_threshold_bytes!` and the `MRT_SERIAL_BLAS_THRESHOLD_BYTES` environment
variable, read in `__init__`. Documented in `docs/src/high-level/performance.md` and in
`set_serial_blas_threshold_bytes!`'s own docstring, which carries the table above; tested in
`test/test_reconstruction_integration.jl`.

The batch-width half of the entry landed with `C6.1`: the reasoning is now in
`with_serial_blas`'s docstring.


Today's 16 MiB is one number fitted on one node, on the *synthetic* reproducer, and the sweep only
brackets the crossover between 4 MiB and 16 MiB. Re-fit it against `recon_bench.jl` on real
solves at 128², 256², 512² and 512²×32, both backends, on the SLURM `test` node with
`--exclusive`. Two outcomes are acceptable: a better single number with the bracket recorded, or
the finding that the crossover is far enough inside the octave that the constant should become a
documented tunable (`ENV`- or `Config`-settable) rather than a `const`. Document whichever, in
`docs/src/high-level/performance.md`.

Also from the same list: the gate keys on the work item, not on batch width, because
`_iterative_reconstruct_core` cannot see how many slabs are in flight. Leave it — the budget wins
in the disagreeing case and that is the conservative direction — but write the reasoning into the
`with_serial_blas` docstring where a reader will find it.

### C6.4 — The residual 8T cost (item 0a, 0c)

**Closed as a record** (2026-09-04), no code of its own — which is what the entry says it is.
0c landed with `S7` (the `--gcthreads` paragraph is in `docs/src/high-level/performance.md`);
0a is `C7`'s prox-buffer work and is tracked there. This entry's remaining content is the
dependency itself: the residual 8T cost is an *allocation* problem, so it is closed by `C7`
and not by anything in `C6`.


`TODO.md` attributes the remaining 10–25% to the `-t 8` process itself: GC over 8 thread-local
arenas (~10 ms → ~35–46 ms per solve, independent of `--gcthreads`) against an ADMM/prox path
allocating 200–760 MiB per solve. So the lever is allocation, not threads: item 0a is `C7`'s
prox-buffer work, and this entry exists to record that dependency. 0c (`--gcthreads` guidance) is
documentation only — one paragraph in `performance.md` saying it was tested and does not help,
which is worth writing precisely because the next person will try it.

### Verification for C6

`benchmarking/scripts/threading_sweep.jl` at `-t 1` and `-t 8`, OpenBLAS and MKL, on the
`--exclusive` `test` node; the 8T ÷ 1T ratios in `TODO.md` (CG-SENSE 1.00, TV 0.81, TGV 0.94,
L1-Wav 0.87, LR 0.88, LLR 0.91, tTV 0.88) are the baseline to beat, and none may regress.

---

### C7 — Per-iteration prox allocation (TODO §5)

Parked "because the natural remedy (buffers owned by the term) needs a decision about thread
ownership first — terms are materialized per slice, but nothing currently documents whether a
materialized term may be entered from more than one thread." `C6.4` makes this the lever for
the residual 8T cost, so take the decision instead of re-parking it:

**Proposed contract: a materialized term is owned by exactly one task for the duration of a
solve.** That is already true of every path in the package — `execute_single_slice` materializes
per slice, `MultiThreadingExecutor` gives each slice to one task, the sequential executor runs
slices one at a time — and it is the contract the buffers need. Write it into
`src/regularization/regularization.jl`'s module docstring as part of the `Regularization`
contract, add it to `docs/src/high-level/regularization.md` and to the "Adding a regularizer"
section of `AGENTS.md`, and make `materialize`'s docstring say a term may hold scratch.

Then add the buffers, one file at a time, each with an `@allocated` regression test:
`multi_scale_low_rank_reg.jl:43,45,53`, `locally_low_rank_reg.jl:76,121,138-142`,
`plug_and_play_reg.jl:40,54-55`. Order by measured allocation share on the LLR benchmark row, and
stop when the remaining files are under a few percent — the goal is the 200+ MiB per solve, not
zero allocation.

### C8 — `NestedThreading.exclude` is a silent no-op for counted pools

`exclude` is consulted only in `_run_guarded`, which walks `GUARDED_POOLS`; counted pools go
through `_enter!`/`_apply!`, which take no `exclude`. Of the registered pools `:blas`, `:mkl`,
`:fftw`, `:nfft` are counted and `:polyester` is the only guarded one — so `exclude` can only ever
name `:polyester`, and unknown names are accepted without error. Two changes, upstream:

1. **Bug**: honour `exclude` in `_apply!` for counted pools, *or* validate the names and throw on
   one that cannot be excluded. Silently accepting a knob that does nothing is the worst option.
2. **Missing API**: an allowlist — `with_thread_budget(f, n; only = (:blas, :mkl))` — so "restrict
   exactly these pools, leave the rest alone" is expressible. A denylist cannot say it without
   enumerating every other pool, which breaks whenever a new pool registers.

`with_serial_blas` (`src/utils.jl`) is exactly `with_restricted_threads(only = (:blas, :mkl))` and
belongs upstream once the API exists: any library doing Julia-level parallelism over small work
items with BLAS-1 inside (Krylov, ODE, proximal solvers) hits the same wall, and
`LinearAlgebra`'s default budget follows CPU affinity rather than `-t`, which NestedThreading
already documents in `_with_blas_threading` without offering a primitive to act on.

Once landed: `@conditionally_enable_threading` (`src/utils.jl`) uses the allowlist and its long
warning block shrinks to a cross-reference; `with_serial_blas` becomes a thin forwarder or goes
away. NestedThreading is not under `deps/` — this is a real upstream PR against the package, so
it is sequenced last and nothing else in this plan may depend on it.

### C9 — Refresh the benchmark tables

The multi-toolbox table at the top of `TODO.md`'s performance section predates the threading
fixes; the 8T columns for TV/TGV/CG-SENSE are stale by construction. After Phases 1 and 2:

- Re-run `comparison/scripts/run_benchmarks.jl` at 1 and 8 threads, OpenBLAS and MKL, on the
  `--exclusive` `test` node, refreshing `comparison/results/benchmark_{openblas,mkl}_{1,8}threads.json`
  and `benchmarking/results/mrt_<backend>_<n>threads.json`.
- Re-run `comparison/scripts/run_accuracy_race.jl` and refresh the time-to-NRMSE table.
- Rewrite `TODO.md` around the new numbers: the applied record stays, the "still open" lists
  shrink to what is actually still open, and each entry that this plan closed is moved with its
  measured before/after.
- Update `docs/src/high-level/performance.md` with the same numbers, and drop the stale
  `AGENTS.md` gotcha "`DFT` accepts `num_threads`, not `threaded`" — `TODO.md` already records
  that `DFT` accepts both and `num_threads` wins, and `_axis_dft_op`
  (`src/encoding/fourier_operators.jl:196-201`) still passes the old form and should pass
  `threaded` so the FFTW size heuristic can run there too.

---

---

## Sequencing summary

| item | phase | depends on | why here |
|---|---|---|---|
| `S1` `WaveletOp` normal op | 1 | — | one trait + one method, guarded by an orthogonality check |
| `S2` `_alternate_sign!` kernel | 1 | — | one function, property-testable; largest single §9 win |
| `S3` `Variation` adjoint | 1 | — | one function, dot-testable against the existing scalar code |
| `S4` skip the no-op thread scope | 1 | — | mirrors `with_serial_blas`'s existing early-out |
| `S5` `Array{AbstractArray}` | 1 | — | idiom already used two functions away; ride along |
| `S6` NFFT knobs | 1 | — | additive keywords, defaults unmoved |
| `S7` `--gcthreads` note | 1 | — | documentation of an existing negative result |
| `C1` `SqrNormL2WithNormalOp` value | 2 | Phase 1 | **done** `c36c3b8` (the `g_z` half is still open) |
| `C2` cancel the `±` pair | 2 | `S2` | **done** `1f5c343` (+ `ec75b5c`); CG-SENSE 2.06x, LR 1.42x |
| `C3` `𝒜ᴴ𝒜` built twice | 2 | `C1` | **done** `274b625`; ~4 MiB/solve, not the claimed 57 MiB |
| `C4` ADMM dead work | 2 | `C3` | **dropped** (2026-09-04); measured at 0.25% of a solve, no code change |
| `C5` operator norm | 2 | `C3` | **done**; `C5.2` dropped, `C5.3` taken (λ convention changed), `powerit` made deterministic |
| `C6` threading residuals | 2 | Phase 1 | **done** (2026-09-04): `C6.1` gates the slice (~10%), `C6.2` re-fits `Variation`'s threshold (2^10 -> 2^17), `C6.3` makes the BLAS threshold a tunable, `C6.4` closes into `C7` |
| `C7` prox buffers | 2 | `C6.4` | needs the thread-ownership contract written down first |
| `C8` upstream `NestedThreading` | 2 | — | a PR against a package we do not vendor; nothing may depend on it |
| `C9` refresh the record | 2 | all | the tables are regenerated once, at the end |
