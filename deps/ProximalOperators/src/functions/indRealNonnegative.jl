# indicator of the nonnegative orthant, after discarding the imaginary part

export IndRealNonnegative

"""
    IndRealNonnegative()

For a **complex** input, return the indicator of the set of complex numbers whose imaginary
part is zero and whose real part is nonnegative,
```math
C = \\{ x \\in \\mathbb{C} : \\mathrm{Im}(x) = 0,\\ \\mathrm{Re}(x) \\geq 0 \\}.
```
The projection zeros the imaginary part and clamps the real part at `0`, matching
RegularizedLeastSquares.jl's `PositiveRegularization` (`enfReal!` then `enfPos!`). For a
real-valued input, use [`IndNonnegative`](@ref) instead — that indicator's set already lives
in ``\\mathbb{R}``, so nothing needs discarding.

The nonnegative orthant is the `low = 0`, `up = Inf` case of [`IndRealBox`](@ref); this is
just that constructor under its own name.
"""
IndRealNonnegative() = IndRealBox(0, Inf)
