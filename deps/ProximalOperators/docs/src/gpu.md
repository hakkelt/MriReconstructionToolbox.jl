# GPU support

Every operator in this package works on GPU arrays. Pass a device array to `prox!` and the
result comes back on the device:

```julia
using CUDA, ProximalOperators

f  = NormL1(1e-3)
x  = CUDA.randn(1024, 1024)
y  = similar(x)

prox!(y, f, x, 0.1)      # runs on the GPU
```

No extra type, wrapper or constructor argument is involved: the operator is the same one
you would build for a `Matrix`, and it picks its path from the storage it is handed.
[`preallocate`](@ref) works the same way — its buffers are built with `similar`, so a device
prototype gives device buffers:

```julia
fp = preallocate(LogisticLoss(CUDA.randn(n)), x)
prox!(y, fp, x, 0.1)     # allocation-free, on the device
```

Julia-level threading turns itself off on device storage. The work is already parallel, and
a `@batch` loop over a device array would only serialise it through scalar indexing; see
[Multithreading](threading.md).

## What "works" means

Working is not the same as being accelerated, and the difference is worth being explicit
about. Every operator falls into exactly one of three tiers.

### Native

Runs as device kernels: broadcasts, reductions and device BLAS, with no scalar indexing and
no host transfer. This is most of the library.

`NormL1`, `NormL0`, `NormL2`, `NormL21`, `NormLinf`, `SqrNormL2`, `ElasticNet`,
`CubeNormL2`, `HuberLoss`, `LogBarrier`, `LogisticLoss`, `SqrHingeLoss`, `HingeLoss`,
`CrossEntropy`, `SumPositive`, `Linear`, `Maximum`, `IndBox`, `IndBallLinf`,
`IndNonnegative`, `IndNonpositive`, `IndZero`, `IndFree`, `IndBinary`, `IndPoint`,
`IndSimplex`, `IndBallL0`, `IndBallL1`, `IndBallL2`, `IndSphereL2`, `IndHalfspace`,
`IndHyperslab`, `IndAffine` (dense), `Quadratic` (direct), `SumLargest`, and every calculus
rule — `Conjugate`, `Postcompose`, `Precompose`, `PrecomposeDiagonal`, `Translate`, `Tilt`,
`Regularize`, `MoreauEnvelope`, `DistL2`, `SqrDistL2`, `PointwiseMinimum`, `ReshapeInput`,
`SeparableSum`, `SlicedSeparableSum`, `Sum` — whose tier is that of what they wrap.

Four of these deserve a note, because they are natively supported only through a *second*
algorithm. `IndSimplex` and `IndBallL1` are projections normally computed by Condat's
algorithm, which walks the input one element at a time through two stacks; `IndBallL0` and
`SumLargest` normally use `partialsortperm`. Neither can run on a device — and a sort is not
an option either, since GPUArrays ships no generic device `sort!`. Instead, each has an
equivalent formulation as a *threshold bisection*: the projection onto the simplex is
`max(x - τ, 0)` where `τ` solves the monotone equation `Σ max(xᵢ - τ, 0) = a`, and bisecting
`τ` needs only `sum`, `min`, `max` and a broadcast. The same trick gives the `L1` ball, and
the top-`k` selection reduces to bisecting a count.

The CPU keeps its exact single-pass algorithms; bisection costs about fifty passes over the
data and is chosen only for storage that cannot run the other one. Where the `k`-th and
`(k+1)`-th magnitudes are equal the two paths may keep a different subset, exactly as
`partialsortperm` is itself unspecified on ties.

### Device LAPACK

Runs natively when the array package provides the factorization, and takes a host round-trip
when it does not. Which happens is a property of the backend: CUDA.jl implements `svd!` and
`eigen!` for `CuArray`, so these are native there.

`NuclearNorm`, `IndPSD`, `IndStiefel`, `IndBallRank`.

### Host round-trip

Copied to the host, computed there, and copied back. Correct, but not accelerated — and it
says so, with a one-time warning per operator type, so the cliff is not something you
discover by profiling.

`IndSOC`, `IndRotatedSOC`, `IndExpPrimal`, `IndExpDual`, `TotalVariation1D`, `LeastSquares`
(direct), `IndGraph`, `IndPolyhedral`.

Each is here for a concrete reason rather than for want of effort: `IndPolyhedral` calls
into OSQP's C library; `IndSOC`, `IndExp` and `TotalVariation1D` are sequential in the
length of a single problem with no independent blocks to spread over work-items; the direct
`LeastSquares` and `IndGraph` hold a factorization computed once at construction, which a
per-call conversion would destroy.

For the cone projections this matters less than it looks. A single cone is a handful of
scalars, and the realistic use is many cones at once through `SlicedSeparableSum`, where the
parallelism is across slices and lives at that level.

Operators in this tier that carry their own data — `LeastSquares(A, b)`, `IndGraph(A)`,
`IndPolyhedral(...)` — should be built from **host** arrays. Only the input and the output
are moved per call; their factorization stays where it was computed.

## Querying the tier

```julia
ProximalOperators.device_tier(f)      # :native, :device_lapack or :host
ProximalOperators.supports_device(f)  # false only for :host
```

The tier propagates through calculus rules, so a wrapper reports what it actually does:

```julia
device_tier(Postcompose(NuclearNorm(1.0), 2.0))      # :device_lapack
device_tier(SeparableSum(NormL1(1.0), IndSOC()))     # :host — one term forces it
```

`test/test_gpu.jl` enumerates every operator and checks both that it runs on a device array
and that its declared tier matches its behaviour, so an operator added later without a
device story fails the suite rather than surprising a user.

## Mixed storage

A host output with a device input is always a mistake, and is rejected with a message naming
both types rather than failing deep inside a broadcast:

```julia
prox!(Array{Float64}(undef, n), NormL1(0.3), CUDA.randn(n), 0.7)
# ERROR: ArgumentError: mixed storage: the output is Array{Float64, 1} but the input is
# CuArray{Float64, 1, ...}. Both must live in the same memory.
```

This is the only case in which an operator refuses to run. Nothing is rejected for being on
a device.

## Testing without a GPU

The test suite uses [JLArrays](https://github.com/JuliaGPU/GPUArrays.jl), a reference GPU
array that executes on the CPU, under `GPUArrays.allowscalar(false)`. That combination is
stricter than a real device about the failure that matters most — a surviving scalar index
is an error rather than a slow path — and it runs on CI with no GPU runner.

What it cannot exercise is the `:device_lapack` tier taking its *native* branch, since
JLArrays implements neither `svd!` nor `eigen!`; there those operators correctly fall back
to the host. Confirming the native branch needs a real device.
