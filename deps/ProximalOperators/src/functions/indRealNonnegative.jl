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
"""
struct IndRealNonnegative end

is_separable(f::Type{<:IndRealNonnegative}) = true
is_convex(f::Type{<:IndRealNonnegative}) = true
is_cone_indicator(f::Type{<:IndRealNonnegative}) = true

function (::IndRealNonnegative)(x::AbstractArray{<:Complex})
    R = real(eltype(x))
    for k in eachindex(x)
        xk = x[k]
        if imag(xk) != 0 || real(xk) < 0
            return R(Inf)
        end
    end
    return R(0)
end

function prox!(y, ::IndRealNonnegative, x::AbstractArray{<:Complex}, gamma)
    R = real(eltype(x))
    for k in eachindex(x)
        y[k] = complex(max(R(0), real(x[k])), R(0))
    end
    return R(0)
end

function prox_naive(::IndRealNonnegative, x::AbstractArray{<:Complex}, gamma)
    R = real(eltype(x))
    y = complex.(max.(R(0), real.(x)), R(0))
    return y, R(0)
end
