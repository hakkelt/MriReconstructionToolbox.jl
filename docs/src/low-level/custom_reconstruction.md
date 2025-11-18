# Custom Reconstruction with StructuredOptimization

This page shows how to experiment with custom reconstruction problems using `StructuredOptimization.jl`, leveraging its convenient bindings to `AbstractOperators.jl` (operators like FFT, Wavelets, reshape, slicing) and `ProximalOperators.jl` (norms and penalties with fast proximal maps).

## Essentials

- `Variable`: declares decision variables (scalars, vectors, matrices, tensors).
- `@minimize`: builds and solves an optimization problem from expressions.
- `problem(...)` and `StructuredOptimization.parse_problem(...)`: build and inspect the parsed problem for a given algorithm.
- Convenient bindings:
  - Smooth terms: `ls(ex)` for 1/2||ex||^2
  - Nonsmooth norms: `norm(ex, 1)`, `norm(ex, 2)`, `norm(ex, Inf)`, mixed `norm(ex, 2, 1)`, nuclear norm `norm(ex, *)`
  - Operator composition: `op * expr`, `reshape(expr, ...)`, `fft`, `dct`, `finitediff`, etc.

```@setup so
using Random
Random.seed!(0)
```

## Two-Variable Example: Sparse + Low-Rank Wavelet Prior

We build a toy reconstruction with two variables and two regularizers:
- Variable `x`: L1 sparsity prior.
- Variable `z`: nuclear norm prior after a wavelet transform and reshaping to a matrix.
- Data fidelity: simple least-squares to synthetic observation `b`.

```@example so
using StructuredOptimization
using AbstractOperators
using ProximalOperators
using WaveletOperators: WaveletOp, WT, wavelet
using SparseArrays: sprandn
using ProximalAlgorithms: FastForwardBackward, PANOCplus

# Problem size (kept small for docs)
nx, ny = 32, 32

# Variables
x = Variable(nx, ny)           # sparse image component
z = Variable(nx, ny)           # low-rank (after transform) component

# Synthetic observation b = x* + z* + small noise (unknown in practice)
x_true = sprandn(nx, ny, 0.1)
z_true = randn(nx, ny) .* 0.1
b = x_true + z_true + 0.01 .* randn(nx, ny)

# Wavelet transform operator (orthonormal, tight frame)
W = WaveletOp(Float64, wavelet(WT.db4), (nx, ny))

# Regularization strengths
λ1 = 0.05
λ2 = 0.2

# Build problem:
#   min_{x,z} 1/2 || (x + z) - b ||^2 + λ2 * nuclearnorm( W*z ) s.t. norm(x, 0) < 50
# It is hard to solve, so first we solve a relaxed version without the L0 constraint:
#   min_{x,z} 1/2 || (x + z) - b ||^2 + λ1 * norm(x, 1) + λ2 * nuclearnorm( W*z )
# Bindings used:
#   - ls(...)       -> least-squares
#   - norm(.,0)     -> L0 "norm" (count of nonzeros)
#   - norm(., 1)    -> L1 norm
#   - norm(., *)    -> nuclear norm (sum of singular values)
#   - W * z         -> operator application

# Parse first (inspect what a solver expects) -- optional, only for demonstration
p = problem( ls(x + z - b), λ2 * norm(W * z, *), norm(x, 0) <= 50 )
alg, kwargs, vars = StructuredOptimization.parse_problem(p, PANOCplus())
println("Prepared keys for PANOCplus: ", keys(kwargs))

# Solve relaxed problem with FISTA
(x̂, ẑ), it = @minimize ls(x + z - b) + λ1 * norm(x, 1) + λ2 * norm(W * z, *) with FastForwardBackward(maxit=50, verbose=false)
println("FISTA Iterations (relaxed problem): ", it)
# Solve original problem with PANOCplus
(x̂, ẑ), it = @minimize ls(x + z - b) + λ2 * norm(W * z, *) st norm(x, 0) <= 50 with PANOCplus(maxit=20, tol=1e-6, verbose=false)
println("Iterations: ", it)
println("Solution sizes: ", size(~x̂), ", ", size(~ẑ))

# Quick sanity: objective components (not rigorous tests)
val_l1 = NormL0(λ1)(~x̂)
# For nuclear norm value evaluate explicitly on the reshaped transform
using LinearAlgebra
S = svdvals(W * ~ẑ)
val_nuc = λ2 * sum(S)
println("L1 term: ", round(val_l1, digits=4), "; Nuclear term: ", round(val_nuc, digits=4))
```

### What Just Happened
- `ls(x + z - b)` is the data fidelity term.
- `norm(x, 1)` applies the L1 norm to `x` via `ProximalOperators.NormL1`.
- `norm(., 0)` applies the L0 "norm" (count of nonzeros) via `ProximalOperators.NormL0`.
- `norm(W*z, *)` applies the nuclear norm through a `Term(NuclearNorm(), ...)` binding; the reshape ensures a matrix domain.
- `W` is an orthonormal/Parseval wavelet transform (tight frame), enabling efficient proximal splitting.
- The problem separates across variables (`x` and `z`), so FISTA (= FastForwardBackward) can handle both nonsmooth terms.


## Anatomy: Basics in One Place

```@example so
using StructuredOptimization
using AbstractOperators
using ProximalOperators
using ProximalAlgorithms: FastForwardBackward

# Variables and access
u = Variable(10); v = Variable(10)

# Smooth term (LS)
term_smooth = ls(u + v - randn(10))

# Common nonsmooths
term_l1  = norm(u, 1)        # L1
term_l2  = norm(v, 2)        # L2
term_l21 = norm(reshape(v, 2, 5), 2, 1)  # group sparsity
term_nuc = norm(reshape(v, 5, 2), *)     # nuclear

# AbstractOperators bindings
ex_fft  = fft(u)   # Fourier transform binding
ex_resh = reshape(ex_fft, 5, 2)                   # reshape expression

# Build and solve explicitly via `problem` + `solve`
q = problem(term_smooth, 0.1*term_l1, 0.05*term_nuc)
sol, it2 = solve(q, FastForwardBackward(maxit=50, verbose=false))
println("Solved in ", it2, " iterations. Size(~u): ", size(~u))
```

## Where to Go Next

- More norms and penalties: see `low-level/proximal_operators.md`.
- Operator catalog and composition patterns: see `low-level/abstract_operators.md` and `low-level/operators.md`.
- For full MRI forward models and reconstruction entry points, see `high-level/reconstruction.md`.

## See Also

- [ProximalOperators.jl Summary](@ref): list of common proximal functions and custom implementation pattern.
- [AbstractOperators.jl Summary](@ref abstract_operators): base operator abstractions and composition rules.
- [MRI Operators](@ref): concrete Fourier, sensitivity map, and subsampling operators.
- [High level Reconstruction](@ref reconstruction): unified reconstruction interface tying everything together.
