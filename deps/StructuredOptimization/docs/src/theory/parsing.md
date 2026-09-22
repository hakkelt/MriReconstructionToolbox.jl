# How problems are parsed

StructuredOptimization does **not** stuff your problem into a matrix. Instead it keeps
the algebraic structure you wrote and rewrites it into the form a first-order solver
expects. This page explains that rewriting so you can predict which problems parse and
why.

## The pipeline

```
Variable  →  Expression  →  Term  →  TermSet  →  solver call
```

- A [`Variable`](@ref) is a leaf holding an array.
- An **Expression** is a `Variable` composed with an `AbstractOperator` (and an
  optional additive displacement `d`): it represents an affine or non-linear map
  `A·x + d`. `operator`, `affine`, `displacement` and [`variables`](@ref) read its
  parts.
- A **Term** pairs a function `f` (from `ProximalOperators`) with an expression, plus
  a scalar weight `λ`: it represents `λ · f(A·x + d)`.
- A **TermSet** is a sum of terms — the whole cost plus constraints (constraints are
  terms whose `f` is a set indicator).

`problem(...)` flattens its arguments into one `TermSet`; `@minimize` expands to a
`solve` on that `TermSet`.

## Trait propagation

Each solver states its assumptions as *properties* the terms must satisfy. The
properties of a term are derived from its function and its operator, using a small
DCP-like ruleset:

| Term property | Rule |
|---|---|
| `is_smooth(λ f(A·x+d))` | `is_smooth(f)` |
| `is_convex(...)` | `is_convex(f) ∧ is_linear(A)` |
| `is_proximable(...)` | `is_proximable(f) ∧ is_AAᴴ_diagonal(A)` |
| `is_quadratic` / `is_generalized_quadratic` | `is_*(f) ∧ is_linear(A)` |
| `is_strongly_convex(...)` | `is_strongly_convex(f) ∧ is_full_column_rank(A)` |

The key consequence: **a non-linear operator destroys convexity and proximability**,
even when `f` itself is convex and proximable. That is why `ls(sin(x) - b)` is smooth
but not convex, and why a convex-only solver rejects it.

## Operator absorption (the "prox trick")

To match a proximal solver, the operator `A` inside `f(A·x + d)` must be folded into a
new function whose proximal map (or gradient) is still computable. There is one
canonical absorption transform, tried in this order:

| Case | Condition on `A` | Absorbed function |
|---|---|---|
| identity | `A = I` | `f` (displacement folded in) |
| diagonal | `A` diagonal | reweighted `f` (no displacement), else `PrecomposeDiagonal` |
| `AAᴴ`-diagonal | `A Aᴴ = diag` | `Precompose(f, A, …)` — prox still closed-form |
| normal operator | `f = ½‖·‖²`, `AᴴA` fuses and is cheaper | `SqrNormL2WithNormalOp` — gradient in one pass |
| general linear | `A` linear | `Precompose(f, A, 1, d)` — gradient only, no prox |
| non-linear | otherwise | `PrecomposeNonlinear(f, A+d)` — gradient only |

The invariant every case preserves is

```math
\\text{absorbed}(x) = λ \\cdot f(A x + d),
```

with the displacement carried **once** (by the operator) and `λ` applied **once**.
The `AAᴴ`-diagonal case is what makes `norm(fft(x), 1)` proximable: the DFT satisfies
`A Aᴴ = N·I`, so `prox_{f∘A}` has a closed form. A general `A` (e.g. a random matrix)
falls into the "general linear" row: only the gradient survives, so the term must be
routed to a solver that treats it as smooth, not proximal.

The order matters, and it is a preference order: every case above the "normal operator"
row keeps an exact prox, so the normal-operator rewrite is only considered once prox is
off the table anyway. It then applies when `AᴴA` collapses into a single operator (a Gram
matrix, a squared diagonal, a frequency-domain multiplication) *and* `A` maps into a
codomain at least as large as its domain, since `AᴴA` acts on the domain and forming it
squares the condition number. A least-squares term over several variables is judged the
same way, on the joint operator.

All of this happens when the problem is parsed. The syntax layer builds only
`λ · f(A·x + d)`: `ls(A*x - b)` is a plain `SqrNormL2` over the expression `A*x - b`, and
which of the rows above it lands in depends on the operator *and* on what the selected
algorithm asks of the term.

## Separable sums and sliced variables

When several terms touch the same variable, the problem is still separable if each
term reads a **disjoint slice** of that variable (via `getindex`). Overlapping,
non-sliced terms on one variable cannot be split and are reported as incompatible by
[`print_diagnostics`](@ref StructuredOptimization.print_diagnostics).

## Matching and diagnostics

For a given algorithm, parsing greedily assigns the **largest** subset of remaining
terms it can to each assumption, deterministically. If some terms remain unassigned,
the problem does not fit that algorithm; [`print_diagnostics`](@ref StructuredOptimization.print_diagnostics) then names the
unsatisfied property per term. Auto-selection (`solve` with no solver) tries every
algorithm and picks the first whose assumptions are fully met.
