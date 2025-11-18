# Optimization Algorithms

MriReconstructionToolbox supports multiple iterative optimization algorithms for solving MRI reconstruction problems. This guide helps you choose and configure the right algorithm for your needs.

## Quick Algorithm Selection

**Not sure which to use?** Let `reconstruct()` choose automatically:

```julia
img = reconstruct(acq, regularization)
# Automatically selects appropriate algorithm
```

**How does it decides?** Here's a decision tree:

```
Is your problem smooth (no L1, TV, etc.)?
├─ Yes → Use CGNR (Conjugate Gradient Normal Residual)
└─ No → Does it have a single non-smooth regularizer where the wrapped operator is symmetric* (e.g. wavelets, temporal Fourier)?
    ├─ Yes → Use FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
    └─ No → Use ADMM (Alternating Direction Method of Multipliers)
```

*Symmetric means the operator satisfies `E' * E = E * E'`, e.g. Fourier-based operators. But actually a more loose condition (`is_AAc_diagonal` from `OperatorCore.jl`) is used: `E' * E = diag(d)` for some `d`, i.e. the normal operator is equal to element-wise scaling.

## ProximalAlgorithms.jl Interface

MriReconstructionToolbox builds on [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl). All algorithms from that package can be used directly. There are three recommended algorithms for MRI reconstruction used by default in `reconstruct()`:
- `CGNR`: Conjugate Gradient Normal Residual for smooth problems
- `FISTA`: Fast Iterative Shrinkage-Thresholding Algorithm for single non-smooth regularizer
- `ADMM`: Alternating Direction Method of Multipliers for multiple regularizers

All algorithms from this library share a common interface with the following parameters:
* `maxit::Int`: maximum number of iteration
* `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
* `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
* `verbose::Bool`: whether the algorithm state should be displayed
* `freq::Int`: every how many iterations to display the algorithm state
* `summary::Function`: function returning a summary of the iteration state, `summary(k::Int, iter::T, state)` should return a vector of pairs `(name, value)`
* `display::Function`: display function, `display(k::Int, alg, iter::T, state)` should display a summary of the iteration state

## Default Algorithms

```@setup imports
using MriReconstructionToolbox
using MIRTjim: jim
using Plots
using Random

Random.seed!(0)

x = shepp_logan(128, 128);
x_noisy = x + 0.02f0 * randn(ComplexF32, 128, 128);
smaps = coil_sensitivities(128, 128, 8);
acq_full = AcquisitionInfo(
    image_size=(128, 128)
)
data_full = simulate_acquisition(x_noisy, acq_full)

pdf = VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05)
pattern = create_sampling_pattern(pdf, (128, 128))
acq = AcquisitionInfo(
    image_size=(128, 128),
    subsampling=pattern,
    sensitivity_maps=smaps
)
data = simulate_acquisition(x_noisy, acq)
```

### Conjugate Gradient Normal Residual (CGNR)

This algorithm solves linear systems of the form

	argminₓ ‖Ax - b‖₂² + ‖λx‖₂² 

where `A` is a symmetric positive definite linear operator, and `b` is the measurement vector,
and `λ` is the L2 regularization parameter. `λ` might be scalar or an array of the same size
as `x`. If `λ` is zero, the problem reduces to a least-squares problem:

	argminₓ ‖Ax - b‖₂²

**Best for:** Least-squares problems with optional Tikhonov regularization

**Properties:**
- Solves normal equations: A'A·x = A'·b
- Good for ill-conditioned problems
- Variant of CG with different mathematical properties

**Parameters:**
- `λ=0`: L2 regularization parameter (default: 0)
- `P`: preconditioner (optional)
- `P_is_inverse`: whether `P` is the inverse of the preconditioner (default: `false`)

**References:**
1. Hestenes, M.R. and Stiefel, E., "Methods of conjugate gradients for solving linear systems."
   Journal of Research of the National Bureau of Standards 49.6 (1952): 409-436.

**Pros:**
- ✅ Very fast convergence for appropriate problems
- ✅ No hyperparameter tuning
- ✅ Memory efficient

**Cons:**
- ❌ Only for a small class of problems
- ❌ Can't handle L1, TV, or other non-smooth terms

**Example:**
```@example imports
reconstruct(data, Tikhonov(1e-4), CGNR(maxit=2), verbose=false) # hide
GC.gc() # hide
img = reconstruct(data, Tikhonov(1e-4), CGNR(maxit=20));
nothing # hide
```

### Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

**Best for:** Single non-smooth regularizer (e.g. L1) with symmetric operator (e.g. wavelets)

!!! note
    FISTA is only an alias for FastForwardBackward from ProximalAlgorithms.jl.

**Properties:**
- Accelerated gradient method
- Handles non-smooth regularization
- Faster convergence than basic ISTA

**Parameters:**
- `mf=0`: convexity modulus `f` (the smooth part of the objective, usually data fidelity term)
- `Lf=nothing`: Lipschitz constant of the gradient of `f` (usually equals to 1 because the Lipschitz contant of squared L2 norm is 1 and `get_encoding_operator` returns a normalized operator)
- `gamma=nothing`: stepsize, defaults to `1/Lf` if `Lf` is set, and `nothing` otherwise.
- `adaptive=true`: makes `gamma` adaptively adjust during the iterations; this is by default `gamma === nothing`.
- `minimum_gamma=1e-7`: lower bound to `gamma` in case `adaptive == true`.
- `reduce_gamma=0.5`: factor by which to reduce `gamma` in case `adaptive == true`, during backtracking.
- `increase_gamma=1.0`: factor by which to increase `gamma` in case `adaptive == true`, before backtracking.
- `extrapolation_sequence=nothing`: sequence (iterator) of extrapolation coefficients to use for acceleration.

**References:**
1. Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave Optimization" (2008).
2. Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009).

**Pros:**
- ✅ Fast for single regularizer
- ✅ Proven convergence guarantees
- ✅ Accelerated compared to basic gradient descent

**Cons:**
- ❌ Only one regularizer
- ❌ Requires Lipschitz constant (usually auto-estimated)
- ❌ Can be sensitive to step size

**Example:**
```@example imports
reconstruct(data, L1Wavelet2D(5e-3), FISTA(maxit=2), verbose=false) # hide
GC.gc() # hide
img = reconstruct(data, L1Wavelet2D(5e-3), FISTA(maxit=100));
nothing # hide
```

### Alternating Direction Method of Multipliers (ADMM)

This algorithm solves optimization problems of the form

	minimize ½‖Ax - b‖²₂ + ∑ᵢ gᵢ(Bᵢx)

where:
- `A` is a linear operator
- `b` is the measurement vector
- `gᵢ` are proximable functions with associated linear operators `Bᵢ`

**Best for:** Multiple regularizers or complex constraints

**Properties:**
- Splits problem into simpler subproblems
- Handles multiple regularizers naturally
- Handles contraints with non-symmetric operators (e.g. total variation)

**Parameters:**
- `P=nothing`: preconditioner for CG (optional)
- `P_is_inverse=false`: whether `P` is the inverse of the preconditioner
- `eps_abs=0`: absolute tolerance for convergence
- `eps_rel=1`: relative tolerance for convergence
- `cg_tol=1e-6`: CG tolerance
- `cg_maxit=100`: maximum CG iterations
- `y0=nothing`: initial dual variables
- `z0=nothing`: initial auxiliary variables
- `penalty_sequence=nothing`: penalty sequence for adaptive rho updating. The following options are available:
  - `FixedPenalty(rho)`: fixed penalty sequence with specified rho values
  - `ResidualBalancingPenalty(rho; mu=10.0, tau=2.0)`: adaptive penalty sequence based on residual balancing [2]
  - `SpectralRadiusBoundPenalty(rho; tau=10.0, eta=100.0)`: adaptive penalty sequence based on spectral radius bounds [3]
  - `SpectralRadiusApproximationPenalty(rho; tau=10.0)`: adaptive penalty sequence based on spectral radius approximation [4]
  Note: rho can be specified either as the `rho` parameter or within the penalty sequence constructor, but not both.

The adaptive penalty parameter schemes are implemented through the penalty sequence types, 
following various strategies from the literature. See the individual penalty sequence types 
for their specific update rules and references.

**References:**
1. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1), 1-122.
2. He, B. S., Yang, H., & Wang, S. L. (2000). Alternating direction method with self-adaptive penalty parameters for monotone variational inequalities. Journal of Optimization Theory and applications, 106(2), 337-356.
3. Lorenz, D. A., & Tran-Dinh, Q. (2019). Non-stationary Douglas–Rachford and alternating direction method of multipliers: Adaptive step-sizes and convergence. Computational Optimization and Applications, 74(1), 67–92. https://doi.org/10.1007/s10589-019-00106-9
4. Mccann, M. T., & Wohlberg, B. (2024). Robust and Simple ADMM Penalty Parameter Selection. IEEE Open Journal of Signal Processing, 5, 402–420. https://doi.org/10.1109/OJSP.2023.3349115

**Pros:**
- ✅ Handles multiple regularizers
- ✅ Very robust and stable
- ✅ Good for complex problems

**Cons:**
- ❌ Slower than FISTA for single regularizer
- ❌ More parameters to tune
- ❌ Each iteration more expensive
- ❌ No convergence guarantees in general

**Example:**
```@example imports
# Multiple regularizers
reg = (L1Wavelet2D(5e-3), TotalVariation2D(1e-3))
reconstruct(data, reg, ADMM(maxit=2), verbose=false) # hide
GC.gc() # hide
img = reconstruct(data, reg, ADMM(maxit=50));
nothing # hide
```

## Tuning Algorithm Parameters

### Maximum Iterations

**How many iterations do you need?**

Typical ranges:
- CG/CGNR: 10-50 iterations
- FISTA: 50-200 iterations
- ADMM: 20-100 iterations

**Strategy:**
```julia
# Start with more iterations to see convergence behavior
img = reconstruct(acq, reg, FISTA(maxit=200), verbose=true)
# Check output to see when convergence plateaus

# Then use fewer iterations in production
img = reconstruct(acq, reg, FISTA(maxit=80))
```

### Convergence Tolerance

Controls early stopping:

```julia
# Stricter convergence
img = reconstruct(acq, reg, FISTA(maxit=200, tol=1e-6))

# Looser convergence (faster but less accurate)
img = reconstruct(acq, reg, FISTA(maxit=200, tol=1e-3))

# Disable early stopping
img = reconstruct(acq, reg, FISTA(maxit=100, tol=0))
```

**Practical tip:** Default `tol=1e-4` is usually good. Tighten to 1e-5 or 1e-6 if you need higher accuracy.

### Verbosity and Monitoring

Track convergence:

```julia
# Show progress every iteration
img = reconstruct(acq, reg, algorithm; verbose=true, freq=1)

# Show progress every 10 iterations
img = reconstruct(acq, reg, algorithm; verbose=true, freq=10)

# No output
img = reconstruct(acq, reg, algorithm; verbose=false)
```

**What to look for:**
- Objective value decreasing
- Changes becoming smaller
- Reasonable convergence rate

## Advanced Usage

### Auto-Selecting Multiple Algorithms

Try multiple algorithms automatically:

```julia
# Provide tuple of algorithms to try
algorithms = (CG(maxit=20), FISTA(maxit=100), ADMM(maxit=50))
img = reconstruct(acq, reg, algorithms)

# Package automatically selects best for problem
# - CG tried first for smooth problems
# - FISTA for single regularizer
# - ADMM for multiple regularizers
```

### Custom Stopping Criteria

```julia
using ProximalAlgorithms

# Custom stopping function
function my_stop(iter, state)
    # Stop if objective doesn't change much
    if iter > 1
        rel_change = abs(state.objective - prev_obj) / abs(prev_obj)
        return rel_change < 1e-5
    end
    prev_obj = state.objective
    return false
end

# Use with low-level interface
# (requires working directly with ProximalAlgorithms.jl)
```

### Warm Starting

Use previous solution as initialization:

```julia
# First reconstruction
img1 = reconstruct(acq, L1Wavelet2D(5e-3))

# Use as initialization for refined reconstruction
img2 = reconstruct(acq, L1Wavelet2D(3e-3); x₀=img1)
```

**When useful:**
- Parameter sweeps
- Iterative refinement
- Multi-stage reconstruction
