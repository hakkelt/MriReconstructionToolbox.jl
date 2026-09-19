# Solvers

## Minimizing a function

```@docs
@minimize
```

!!! note "Problem warm-starting"

    By default *warm-starting* is always enabled.
    For example, if two problems that involve the same variables are solved consecutively,
    the second one will be automatically warm-started by the solution of the first one.
    That is because the variables are always linked to their respective data vectors.
    If one wants to avoid this, the optimization variables needs to be manually re-initialized
    before solving the second problem e.g. to a vector of zeros: `~x .= 0.0`.


## Specifying solver and options

You can pick the algorithm to use as a `Solver` object from the
[`ProximalAlgorithms.jl`](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
package — for example `PANOCplus()`, `ZeroFPR()`, `PANOC()`,
`FastForwardBackward()`, or `CGNR()`. Each accepts options such as `maxit` and `tol`
(see the ProximalAlgorithms documentation), which you may also override at
[`solve`](@ref) time via keyword arguments. See
[Choosing an algorithm](@ref) below for guidance on which to use.


## Parse and solve

The macro [`@minimize`](@ref) automatically parse and solve the problem.
An alternative syntax is given by the function [`problem`](@ref) and [`solve`](@ref).

```@docs
problem
solve
```

Once again, the `Solver` objects is to be picked from
[`ProximalAlgorithms.jl`](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)).

## Choosing an algorithm

If you do not pass a solver, `solve`/`@minimize` **auto-select** one by matching the
problem structure against each algorithm's assumptions. You can inspect that matching
directly:

```@docs
suggest_algorithm
StructuredOptimization.print_diagnostics
```

As a rule of thumb:

| Problem type | Recommended solver |
|---|---|
| `f(Ax) + g(x)`, `f` smooth (convex or not) | `PANOCplus` |
| Purely proximal (`g(x)` only, or a sum of proximable terms) | `FastForwardBackward` |
| Nonconvex smooth `f` | `ZeroFPR` or `PANOCplus` |
| Least squares `‖Ax-b‖²` (+ optional ridge) | `CGNR` |

!!! warning "PANOC / ZeroFPR stepsize"

    `PANOC` and (less often) `ZeroFPR` can hit a "stepsize `gamma` became too small"
    warning and return an unreliable point on some problems. Prefer `PANOCplus` for
    convergence-critical work; reach for `PANOC`/`ZeroFPR` mainly when a problem is
    nonconvex and `PANOCplus` struggles.

## When parsing fails

If no solver's assumptions can be satisfied, `solve` raises an error. Call
[`print_diagnostics`](@ref StructuredOptimization.print_diagnostics) to see *why*: it lists each term that could not be
prepared together with the property it failed to certify (`is_convex`,
`is_proximable`, `is_smooth`, …). A common cause is asking a solver that requires
convexity to handle a nonlinear (hence non-convex) composition such as
`ls(sin(x) - b)` — the diagnostic reports `f requires is_convex`.

If a term is *almost* usable but not proximable in closed form, `smooth(f)` replaces
it with its Moreau envelope, which is smooth and can then be handled by a
gradient-based solver — see [Functions](functions.md).

## References

[[1]](http://www.mit.edu/~dimitrib/PTseng/papers/apgm.pdf) Tseng, *On Accelerated Proximal Gradient Methods for Convex-Concave Optimization* (2008).

[[2]](http://epubs.siam.org/doi/abs/10.1137/080716542) Beck, Teboulle, *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*, SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009).

[[3]](https://arxiv.org/abs/1606.06256) Themelis, Stella, Patrinos, *Forward-backward envelope for the sum of two nonconvex functions: Further properties and nonmonotone line-search algorithms*, arXiv:1606.06256 (2016).

[[4]](https://doi.org/10.1109/CDC.2017.8263933) Stella, Themelis, Sopasakis, Patrinos, *A simple and efficient algorithm for nonlinear model predictive control*, 56th IEEE Conference on Decision and Control (2017).
