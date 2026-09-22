# FAQ & Troubleshooting

## Which algorithm should I use?

Let `solve`/`@minimize` pick automatically when in doubt — they match the problem
against every algorithm's assumptions. To see the candidates, call
[`suggest_algorithm`](@ref). If you want to choose yourself, see the
[algorithm table](solvers.md#Choosing-an-algorithm). Short version: **`PANOCplus`**
is the safe default for `f(Ax) + g(x)` with a smooth `f`; **`FastForwardBackward`**
for purely proximal problems; **`CGNR`** for plain least squares.

## "Sorry, I cannot parse this problem" — now what?

`solve` throws this when no solver's assumptions are met. Diagnose it:

```julia
julia> print_diagnostics(problem(ls(sin(x) - b)), FastForwardBackward())
```

The output names the property each un-prepared term failed, for example
`f requires is_convex`. The usual culprits:

- **A non-linear operator inside a convex-only solver.** `sin(x)`, `sigmoid(...)`,
  `pow(x, 2)` make the composition non-convex, so convex-only algorithms
  (`FastForwardBackward`) reject it. Use `ZeroFPR` or `PANOCplus` instead.
- **A regularizer that is not proximable.** Not every function has a closed-form
  proximal map (e.g. `norm(A*x, 1)` with a general `A`). Either reformulate so the
  operator is absorbable (identity, diagonal, or `AAᴴ`-diagonal — see
  [How problems are parsed](theory/parsing.md)), or wrap the term in
  [`smooth`](@ref) to use its Moreau envelope.
- **An indicator that is not proximable.** `norm(x, 1) <= r` (an `IndBallL1`) has no
  guaranteed exact projection and is treated as non-proximable; `norm(x, 2) <= r`
  (`IndBallL2`) is.

## My solver runs but returns a wrong / non-converged answer

Check for a `stepsize gamma became too small` warning. `PANOC` and sometimes
`ZeroFPR` are prone to it. Re-run with `PANOCplus`. For nonconvex problems, confirm
the model really is smooth where the algorithm needs it.

## How do I warm-start?

Warm-starting is automatic: variables stay linked to their data arrays, so solving a
second problem over the same variables starts from the previous solution. To reset,
zero the variable first: `~x .= 0.0`.

## `Float64` vs `Float32`

Variables carry their element type (`Variable(Float32, n)`); the whole pipeline is
matrix-free and type-generic, so a `Float32` variable runs the solver in single
precision — faster and lower-memory, at reduced accuracy. Match the element type of
your data arrays to avoid silent promotion.

## Why is `norm(fft(x), 1)` proximable but `norm(A*x, 1)` is not?

Because the DFT satisfies `A Aᴴ = N·I` (it is `AAᴴ`-diagonal), the proximal map of
`f ∘ A` has a closed form; a general `A` does not. This "prox trick" and the exact
absorption rules are described in [How problems are parsed](theory/parsing.md).
