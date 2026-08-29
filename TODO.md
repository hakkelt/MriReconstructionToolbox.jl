# TODO — open observations

Context gathered from a code-quality review (reuse / simplification / efficiency / altitude) of the work
between `463c223` and `706042c` (vendored subtrees under `deps/` excluded). This file records **what was
observed and why it was left alone** — it is not a plan, and none of the entries below is scheduled or
sequenced. Line numbers are as of the commit that adds this file.

The cleanups that *were* applied in the same pass are listed at the end, so that an entry here is not
mistaken for something already dealt with.

---

## 1. The component path is a parallel copy of the single-variable path

`src/reconstruction/reconstruct.jl`

| component path | single-variable twin |
| --- | --- |
| `_reconstruct_dispatch_components` (:145) | `_reconstruct_dispatch` (:82) |
| `_reconstruct_components` (:198) | `_reconstruct` (:117) |
| `_direct_reconstruct_components` (:220) | `_direct_reconstruct` (:327) |
| `_iterative_reconstruct_components` (:276) | `_iterative_reconstruct` (:343) |

The two families mirror each other line-for-line: k-space scaling, `normalize_op`, `freq`/`ϵ`/`tol`/`stop`/
`display` setup, `patch_algorithm_with_default_values`, `solve`, inverse scaling, and the NamedDims rewrap.
What actually differs is three things: `x₀` is a tuple, a different `build_model` overload is chosen, and the
result is wrapped in a `DecomposedImage`.

Observed consequences:

- Every change to scaling, stopping criteria or executor policy has to be made twice, and divergence is
  silent.
- The paths have **already** diverged once: `config.disable_normalop_optimization` is honoured in
  `_iterative_reconstruct` (:351) and simply absent from `_iterative_reconstruct_components`, which applies
  `normalize_op` unconditionally at :283. Whether that is
  intentional (the component `build_model` never uses `normalop_ls`, see `build_model.jl:87-89`) or an
  oversight is not established.

Note that the *decomposition* half of this duplication (`execute_regularized` /
`execute_regularized_components`) has since been folded into a shared `execute_two_phase` skeleton, so the
remaining duplication is confined to `reconstruct.jl`.

Constraint on any future consolidation: `DecomposedImage` is part of the public return type, and a
single-component problem must keep returning a plain array.

## 2. `Lf = n_components` is derived at the caller

`src/reconstruction/reconstruct.jl:307` passes `length(components)` down to
`patch_algorithm_with_default_values`, which turns it into the Lipschitz constant at :417.

The reasoning (‖[𝒜 … 𝒜]‖² = n‖𝒜‖² when 𝒜 is normalized to unit norm) is documented at the use site, but the
value is a property of the model that was built, not of the component count:

- it is silently wrong when `disable_operator_normalization = true`;
- it stops being right the moment components no longer share one `𝒜`, or the data term changes shape.

`build_model` / `build_model_with_variables` are where the data term is assembled, and neither currently
exposes a Lipschitz estimate.

## 3. Encoding operator is built twice per slice in the decomposition path

`src/reconstruction/decomposition.jl:103` (plain) and `:123` (components) build a full encoding operator —
FFTW planning included — use it only for `𝒜'y` to derive the slice's scale, and discard it. Phase 2 then
builds an identical operator for the same slice inside `_reconstruct` / `_reconstruct_components`
(`reconstruct.jl:119`, `:202`).

The plain path additionally builds the phase-1 operator with `fast_planning = true` (FFTW `ESTIMATE`) and the
phase-2 one without, so the two are not even the same object to reuse.

Counter-pressure, and the reason this was not touched: phase 1 already retains every slice's `local_acq` and
warm start in `prelim` (`decomposition.jl:154`) for the whole run — peak memory is O(dataset), not O(slice),
as it was before the two-phase scheme. Carrying `𝒜` per slice as well would add to that. Trading the
duplicated build against the memory is a judgement call about typical dataset sizes, not a mechanical fix.

## 4. `prelim` is an `Array{Any}` of heterogeneous tuples

`src/reconstruction/decomposition.jl:154` stores `(id, local_acq, warm_start, scale)` per slice; the
unpacking in phase 2 (:168) is type-unstable, and inference gives up on the phase-2 closure body.

Related: `results = Array{AbstractArray}(...)` at :72 and :165.

Whether this matters is unmeasured — the loop body is one whole slice reconstruction, so the dynamic dispatch
per slice is likely noise against the solve itself. No benchmark was run either way.

## 5. Per-iteration allocation in the new prox implementations

None of these were changed: they are all inner-loop scratch, and the natural remedy (buffers owned by the
term) needs a decision about thread ownership first — terms are materialized per slice, but nothing currently
documents whether a materialized term may be entered from more than one thread.

- `src/regularization/multi_scale_low_rank_reg.jl:43,45` — `copy(x)` when the prox is in-place, plus
  `similar(y)`, on every call. Two full-image temporaries per iteration.
- `src/regularization/multi_scale_low_rank_reg.jl:53` — `return f(y)` re-runs every scale's full SVD sweep on
  the averaged point. This one is **deliberate**: the comment at :50-52 and the `ProximalAverage` docstring
  (:13-15) explain that averaging the per-scale values at their own prox points would return a different,
  strictly smaller number. Changing it changes the reported objective, so it is a behavioural decision, not a
  cleanup. Evaluating lazily (only when the caller actually displays the objective) would preserve the value
  but needs a `ProximalCore` contract that does not exist today.
- `src/regularization/locally_low_rank_reg.jl:121,138-142` — `zeros` per call, then per block per batch slice
  per iteration: a fresh block matrix, `svd!`'s `U/S/Vt`, and the thresholded `σ` vector.
- `src/regularization/locally_low_rank_reg.jl:76` — `_block_indices` returns `UnitRange`s or `Vector{Int}`s
  depending on a runtime check, so the gather/scatter views are dynamically typed.
- `src/regularization/plug_and_play_reg.jl:40,54-55` — `collect(slice)` materializes a slice that is already
  a contiguous strided view, and the `:magnitude` path adds `abs.`, the denoiser output, `angle.`, `cis.` and
  their product.

## 6. `x = copy(~x_var)` after `solve`

`src/reconstruction/reconstruct.jl:385` (and the per-component `map(v -> copy(~v), vars)` at :310).

`build_model_with_variables` already copies `x₀` before wrapping it in a `Variable` (`build_model.jl:54`), and
`build_model` for components does the same at :107, so the solution array is not aliased to the caller's `x₀`
and the extra copy may be redundant. It was added by `73a180f` ("stop reconstruct from writing its solution
through the caller's x₀") — that commit's regression test is the thing to read before concluding anything
here. Not touched, because the cost is one image copy per reconstruction and the downside of being wrong is a
silent aliasing bug.

## 7. `HardThreshold` / `SparsityLimit` decode a `Symbol` domain in four places

`src/regularization/hard_threshold_reg.jl` — `_sparsifying_operator` (:7), `_sparsifying_affected_dims` (:14),
`_sparsifying_repr` (:17) and `_check_sparsifying_domain` (:19) each enumerate `:image` / `:wavelet2d` /
`:wavelet3d`, and both terms carry `wavelet` + `levels` fields that are dead when the domain is `:image`.

A fourth transform means editing four functions and two constructors. The package already models "an operator
plus its affected dims" as a regularization type, so the enumeration is arguably re-implementing dispatch —
but the current form is also the public keyword API (`domain = :wavelet2d`), so changing it is an API
question, not a refactor.

Side note: the final `throw` in `_sparsifying_operator` (:11) is unreachable — both constructors run
`_check_sparsifying_domain` with an identical message. Left in place as defence in depth for internal callers.

## 8. Public entry point tests types instead of dispatching

`src/reconstruction/reconstruct.jl:68` does `any(r -> r isa Component, regularization)` and then re-asserts
`all(...)`, which also forces the `x₀`-type `@argcheck` to be written on both branches (:69, :74). Dispatching
on `Tuple{Vararg{Component}}` vs `Tuple{Vararg{Regularization}}`, with a narrow mixed-tuple method carrying
the "wrap loose regularization terms in a `Component`" error, would express the same thing in the signature.
Entangled with item 1 — if the two paths ever merge, the branch disappears on its own.

## 9. Test-side duplication

- `prox_of(reg, x, γ)` is defined five times: `test/test_reg_hard_threshold.jl` (×2),
  `test/test_reg_low_rank.jl` (×2), `test/test_reg_plug_and_play.jl`. All are
  `materialize` → `SO.extract_functions` → `PC.prox!`, differing only in whether the prox value is returned.
- Several `@testitem`s immediately open a `@testset` with the same name (`test/test_reg_shared.jl:3/10`,
  `39/46`, `71/78`, `104/111`), and each item repeats the same `using` preamble.

Each `@testitem` is its own module, so some of this duplication is structural to TestItemRunner. The suite has
no `@testsnippet` yet; introducing one is the piece of scaffolding these entries share.

---

## Already applied (do not re-open as findings)

From the same review pass, on top of the reviewed range:

- `get_affected_dims`: terms now implement the `::Nothing` method and the base layer forwards
  `::AcquisitionInfo` to it (`regularization.jl`). Ten forwarding methods and `_box_affected_dims` deleted.
  `EdgePreservingRoughness2D/3D`, `SecondOrderTotalVariation2D/3D` and `TotalGeneralizedVariation2D`, which
  had only the `::AcquisitionInfo` method, now work through the no-acquisition entry point as well.
- `identity_operator(x)` — replaced 7 copies of the `Eye` / `NamedDimsOp{names,names}(Eye(parent(x)))` pair.
- `_collapse_direction_axes(op, x, n)` — replaced 3 copies of the unwrap/reshape/rewrap idiom (TV,
  second-order TV, TGV).
- `_check_dim_spec(dim, name; allow_nothing)` — replaced `_check_lowrank_dims` and 3 inlined copies.
- `dims_of(x)` — replaced 6 copies of `x isa NamedDimsArray ? dimnames(x) : (1:ndims(x))`.
- `materialize_all(regs, x)` — shared by `build_model_with_variables` and `Component`'s fold.
- `execute_two_phase` — shared skeleton of `execute_regularized` and `execute_regularized_components`.
- `run_slices!` — one method built on `for_each_item!` instead of one per executor.
- `_resolve_scale` — shared by `_direct_reconstruct` and `_direct_reconstruct_components`.
- `get_component_x0s` — no longer re-validates what `check_x₀_components_size` already checked.
- `stack_image_slices` — dispatches on the first result instead of an `isa` test.
- ADMM defaults — single-key merge framework reduced to a membership test.
- `reduce(+, xs)` → `broadcast(+, xs...)` for the decomposition total.
