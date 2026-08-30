# TODO — open observations

Context gathered from a code-quality review (reuse / simplification / efficiency / altitude) of the work
between `463c223` and `706042c` (vendored subtrees under `deps/` excluded). This file records **what was
observed and why it was left alone** — it is not a plan, and none of the entries below is scheduled or
sequenced.

---

## Residual observations

- `results = Array{AbstractArray}` in `src/reconstruction/decomposition.jl:72, 179` (type instability in slice container; noise relative to slice solve).
- The `disable_normalop_optimization` divergence in the component path is **intentional and documented** at `build_model.jl:141-144` (plain `ls` is always used as fast normal-operator path for a sum of shared operators requires upstream `HCAT` normal-op fusion), not a defect.
- `two local prox_of copies` in `test/test_reg_low_rank.jl` and repeated `using Wavelets` in `test/test_reg_shared.jl` folded into `test_snippets.jl` in Stage 0.

---

## Out of Scope

### 5. Per-iteration allocation in the new prox implementations
None of these were changed: they are all inner-loop scratch, and the natural remedy (buffers owned by the
term) needs a decision about thread ownership first — terms are materialized per slice, but nothing currently
documents whether a materialized term may be entered from more than one thread.
- `src/regularization/multi_scale_low_rank_reg.jl:43,45`
- `src/regularization/multi_scale_low_rank_reg.jl:53`
- `src/regularization/locally_low_rank_reg.jl:121,138-142`
- `src/regularization/locally_low_rank_reg.jl:76`
- `src/regularization/plug_and_play_reg.jl:40,54-55`

---

## Already applied (applied in `d9964eb` and Stage 0)

- **1. Component path consolidation**: Shared iterative driver (`_iterative_reconstruct_core` in `src/reconstruction/reconstruct.jl:337`), folding duplicated setup.
- **2. Lipschitz constant estimation (`Lf`)**: Sourced directly from built variables (`reconstruct.jl:370`).
- **3. Operator caching in decomposition**: `execute_two_phase` caches encoding operator `𝒜` across phases (`src/reconstruction/decomposition.jl:147`).
- **4. Concrete type in `prelim`**: Typed tuples in decomposition prelim array (`src/reconstruction/decomposition.jl:164-167`).
- **6. Defensive copy on solution extraction (`copy(~x_var)`)**: Profiled in Stage 1 benchmark. Measured memory delta is 32 KB out of 15.5 MB (0.21%), well below the 2% threshold. Retained as won't-fix to guarantee callers and multi-variable solvers never alias internal solver state.
- **7. Sparsifying domain dispatch**: `hard_threshold_reg.jl` refactored with `Val`-dispatch.
- **8. Entry point dispatch**: Entry points dispatch on tuple shape (`reconstruct.jl:78-96`).
- **9. Test-side duplication**: Centralized test snippets in `test/test_snippets.jl` (`RegTestSetup`, `ProxOf`).
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
