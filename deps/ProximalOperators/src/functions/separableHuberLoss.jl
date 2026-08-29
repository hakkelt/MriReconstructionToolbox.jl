# Separable (element-wise) Huber loss function

export SeparableHuberLoss

"""
    SeparableHuberLoss(ρ=1, μ=1)

Return the function
```math
f(x) = \\sum_i h(x_i), \\qquad
h(t) = \\begin{cases}
    \\tfrac{μ}{2}|t|^2 & \\text{if}\\ |t| ⩽ ρ \\\\
    ρμ(|t| - \\tfrac{ρ}{2}) & \\text{otherwise},
\\end{cases}
```
where `ρ` and `μ` are positive parameters.

This is the element-wise (separable) Huber loss, i.e. the Huber function is applied to every entry of `x` and
the results are summed. It differs from [`HuberLoss`](@ref), which applies the Huber function once to the
Euclidean norm of the whole array. The two agree when `x` has a single entry.

The separable form is the one used as an edge-preserving potential in image reconstruction: it behaves like
`μ/2 ⋅ ‖x‖²` on small entries (so it is smooth, and quadratic penalties do not over-penalize noise-level
differences) and like `ρμ‖x‖₁` on large ones (so large entries — edges — are penalized only linearly). Being
smooth everywhere, it also lets gradient-based algorithms handle a term that would otherwise require a
proximal step.
"""
struct SeparableHuberLoss{R, S}
    rho::R
    mu::S
    function SeparableHuberLoss{R, S}(rho::R, mu::S) where {R, S}
        if rho <= 0 || mu <= 0
            error("parameters rho and mu must be positive")
        else
            new(rho, mu)
        end
    end
end

is_convex(f::Type{<:SeparableHuberLoss}) = true
is_smooth(f::Type{<:SeparableHuberLoss}) = true
is_separable(f::Type{<:SeparableHuberLoss}) = true

SeparableHuberLoss(rho::R=1, mu::S=1) where {R, S} = SeparableHuberLoss{R, S}(rho, mu)

function _huber(f::SeparableHuberLoss, absx::R) where R <: Real
    return absx <= f.rho ? f.mu / R(2) * absx^2 : f.rho * f.mu * (absx - f.rho / R(2))
end

function (f::SeparableHuberLoss)(x)
    R = real(eltype(x))
    v = R(0)
    for k in eachindex(x)
        v += _huber(f, R(abs(x[k])))
    end
    return v
end

function gradient!(y, f::SeparableHuberLoss, x)
    R = real(eltype(x))
    v = R(0)
    for k in eachindex(x)
        absx = R(abs(x[k]))
        if absx <= f.rho
            y[k] = f.mu * x[k]
        else
            # The gradient of |t| is t/|t|, which extends to complex entries as the phase of x[k].
            y[k] = (f.mu * f.rho) / absx * x[k]
        end
        v += _huber(f, absx)
    end
    return v
end

function prox!(y, f::SeparableHuberLoss, x, gamma)
    R = real(eltype(x))
    mugam = f.mu * gamma
    v = R(0)
    for k in eachindex(x)
        absx = R(abs(x[k]))
        # Same shrinkage as HuberLoss, but driven by |x[k]| instead of the norm of the whole array: below
        # the kink the quadratic branch scales by 1/(1+γμ), above it the linear branch subtracts γμρ.
        scal = R(1) - min(mugam / (R(1) + mugam), iszero(absx) ? R(1) : mugam * f.rho / absx)
        y[k] = scal * x[k]
        v += _huber(f, scal * absx)
    end
    return v
end

function prox_naive(f::SeparableHuberLoss, x, gamma)
    R = real(eltype(x))
    absx = abs.(x)
    scal = R(1) .- min.(f.mu * gamma / (R(1) + f.mu * gamma), (f.mu * gamma * f.rho) ./ max.(absx, eps(R)))
    y = scal .* x
    return y, sum(_huber.(Ref(f), R.(abs.(y))))
end
