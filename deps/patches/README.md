# Vendoring patches

One patch per vendored package, holding everything the copy under `deps/<package>` has that the
package's `integration` branch does not. `tools/vendor.jl sync` re-applies the patch after every
`git subtree pull`, so the pair (`integration` branch, patch) reproduces the vendored tree
exactly -- which is what makes a re-vendor a mechanical operation instead of a three-way merge.

Regenerate with:

```
julia tools/vendor.jl rebuild    # assemble each fork's `integration` branch from deps/vendor.toml
julia tools/vendor.jl patch      # diff each integration branch against deps/<package>
```

`patch` writes the diff of a clean `git archive` of `integration` against the working `deps/`
tree, so it is always exactly the current delta: nothing about it is hand-maintained.

## What is in them, and what should not stay

Two kinds of change are mixed in here on purpose, and telling them apart is the point of having
the delta in one file:

- **Adaptations**, which are permanent. Vendoring a package as an `include`d submodule forces
  changes no upstream would accept: every cross-package import has to be relative
  (`using ..AbstractOperators`), an `ext/` directory never loads so its contents have to be moved
  into `src/` and included by hand (`ProximalOperators`' OSQP and RecursiveArrayTools
  extensions), and material MRT does not ship -- the packages' own `docs/examples`, CI workflows,
  GPU test environments -- is pruned.
- **Drift**, which should not be here at all. A fix made while developing MRT and not yet carried
  back to the branch that owns the code shows up as a hunk in these patches. Push it to that
  branch, rebuild, and regenerate: the hunk disappears on its own, and that disappearance is the
  confirmation that the fix really did land upstream-bound.

A patch that stops applying after a rebuild is therefore good news, not a breakage: whatever it
carried is now in the branch.

## The integration branches these are relative to

`integration` is disposable and rebuilt from `deps/vendor.toml` on demand. Some of its merges
conflict -- mostly where a fork branch is still based on an older `master` rather than on the
branch the manifest says it is stacked on. Those conflicts were resolved once by taking the file
as it stands in `deps/`, on the grounds that the vendored tree is the version that is known to
work, and `git rerere` is enabled in each dependency checkout so the resolutions replay
automatically on the next rebuild. Resolving a conflict this way is a stopgap: the real fix is to
rebase the branch onto its manifest parent, after which the conflict does not arise.
