# Multithreading

Every operator whose kernel is worth parallelising takes a `threaded` keyword:

```julia
f = NormL1(1e-3)                  # may thread, if the input is large enough
g = NormL1(1e-3; threaded = false) # will never thread
```

The keyword is a *permission*, not an instruction. Whether a call actually threads also
depends on the input, which is not known when the operator is built, so the decision is
completed when `x` first arrives:

```julia
is_threaded(f, x)   # what will actually happen, for an input like x
supports_threading(f)  # whether f has a threaded path at all
```

`is_threaded` answers `false` whenever any of the following holds, and it is the honest
answer rather than the requested one:

* the operator was built with `threaded = false`;
* Julia is running with a single thread;
* `x` is shorter than the operator's own threshold;
* `x` does not live in CPU memory — a GPU array is already parallel, and a Julia-level
  thread pool on top of it would only serialise the work.

`threaded = false` is a hard veto and nothing overrides it. That matters because the
block-parallel calculus rules rely on it: a caller that parallelises over problems of its
own can build every operator with `threaded = false` and be certain no second thread pool
opens underneath.

## Thresholds

Threading a short array costs more than it saves, and the size at which that turns around
depends on how expensive one element is. Operators are classified by cost, and each class
carries a measured threshold:

| Class | Threshold | Kernel | Examples |
|---|---|---|---|
| `Transcendental` | `2^10` | `@turbo` / `@tturbo` | `LogBarrier`, `LogisticLoss`, `CrossEntropy`, `SqrHingeLoss` |
| `Arithmetic` | `2^15` | `@simd` / `@batch` | `NormL1`, `ElasticNet`, `IndBox`, `NormL0` |
| `MemoryBound` | `2^18` | `@simd` / `@batch` | `SqrNormL2`, `NormL2`, `IndBallL2`, `HuberLoss` |
| `BlockParallel` | `2^16` | `@batch` over blocks | `SlicedSeparableSum`, `SeparableSum`, `NormL21` |

The numbers are measured, not chosen: `benchmark/threading_sweep.jl` produces them, and its
`RESULTS` block records the machine, thread count and date, along with the crossover
measurements each one rests on. Re-run it before changing any of them.

The transcendental class is the one that gains most, and not only from threading: `exp` and
`log` vectorise through LoopVectorization where `Base` evaluates them one element at a time,
which is worth 3–7× even below the threading threshold and 4–10× above it. Complex inputs
never take that path — LoopVectorization does not vectorise them — and fall back to
Polyester.

## Block-parallel rules

`SlicedSeparableSum` is the rule that benefits most, because its slices are independent by
construction. It parallelises over them when there are at least `MIN_BLOCKS_FOR_PARALLEL`
(4) of them and enough total work; the block *count* is a separate condition from the size,
because a two-block loop does not amortise the spawn however large the blocks are.

`SeparableSum` parallelises the same way, but its block count is a compile-time tuple
length, so for the usual two- or three-term sum the decision folds away at compile time and
no task is ever created.

Neither rebuilds its children with `threaded = false`. They do not need to: the loop body
runs inside a [NestedThreading](https://github.com/hakkelt/NestedThreading.jl) budget scope,
so a child that threads internally, or that calls BLAS, shares the budget rather than
opening a second full pool on top of the first.

## Numerical reproducibility

A threaded reduction re-associates the sum, so a threaded call and a serial call agree to
rounding rather than bit for bit. The *written array* is identical either way — no reduction
is involved in it — but the returned function value can differ in the last digits. Code that
needs bitwise reproducibility across runs should build its operators with
`threaded = false`, which restores exactly the kernel the package used before threading
existed.

Preallocated operators are not thread-safe: all calls share one set of buffers. Use one
[`preallocate`](@ref) copy per task.

```@docs
is_threaded
supports_threading
```
