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

(f::SeparableHuberLoss)(x) = reduce_call(f, xk -> _huber(f, abs(xk)), x)

# The gradient of |t| is t/|t|, which extends to complex entries as the phase of t.
function _huber_gradient(f::SeparableHuberLoss, xk)
    absx = abs(xk)
    return absx <= f.rho ? f.mu * xk : (f.mu * f.rho) / absx * xk
end

# Same shrinkage as HuberLoss, but driven by |t| instead of the norm of the whole array: below the
# kink the quadratic branch scales by 1/(1+γμ), above it the linear branch subtracts γμρ.
function _huber_shrink(f::SeparableHuberLoss, mugam, xk)
    absx = abs(xk)
    R = typeof(absx)
    scal = one(R) - min(R(mugam / (1 + mugam)), iszero(absx) ? one(R) : R(mugam * f.rho / absx))
    return scal * xk
end

function gradient!(y, f::SeparableHuberLoss, x)
    # The value is read before `y` is written, since `y` may alias `x`.
    v = f(x)
    map_prox!(f, y, xk -> _huber_gradient(f, xk), x)
    return v
end

function prox!(y, f::SeparableHuberLoss, x, gamma)
    mugam = f.mu * gamma
    # |y[k]| is the shrunk |x[k]|, so the value is the loss of `y` itself.
    return map_reduce_prox!(f, y, xk -> _huber_shrink(f, mugam, xk), yk -> _huber(f, abs(yk)), x)
end

function prox_naive(f::SeparableHuberLoss, x, gamma)
    R = real(eltype(x))
    absx = abs.(x)
    scal = R(1) .- min.(f.mu * gamma / (R(1) + f.mu * gamma), (f.mu * gamma * f.rho) ./ max.(absx, eps(R)))
    y = scal .* x
    return y, sum(_huber.(Ref(f), R.(abs.(y))))
end
