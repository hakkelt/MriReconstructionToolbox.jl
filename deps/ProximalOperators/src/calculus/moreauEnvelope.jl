export MoreauEnvelope

"""
    MoreauEnvelope(f, γ=1)

Return the Moreau envelope (also known as Moreau-Yosida regularization) of function `f` with parameter `γ` (positive), that is
```math
f^γ(x) = \\min_z \\left\\{ f(z) + \\tfrac{1}{2γ}\\|z-x\\|^2 \\right\\}.
```
If ``f`` is convex, then ``f^γ`` is a smooth, convex, lower approximation to ``f``, having the same minima as the original function.
"""
struct MoreauEnvelope{R, T, B}
    g::T
    lambda::R
    buf::B
    function MoreauEnvelope{R, T, B}(g::T, lambda::R, buf::B) where {R, T, B}
        if lambda <= 0 error("parameter lambda must be positive") end
        new(g, lambda, buf)
    end
end

MoreauEnvelope(g::T, lambda::R=1; buf=nothing) where {R, T} =
    MoreauEnvelope{R, T, typeof(buf)}(g, lambda, buf)

MoreauEnvelope{R, T}(g::T, lambda::R) where {R, T} =
    MoreauEnvelope{R, T, Nothing}(g, lambda, nothing)

is_convex(::Type{<:MoreauEnvelope{R, T}}) where {R, T} = is_convex(T)
is_smooth(::Type{<:MoreauEnvelope{R, T}}) where {R, T} = is_convex(T)
is_generalized_quadratic(::Type{<:MoreauEnvelope{R, T}}) where {R, T} = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:MoreauEnvelope{R, T}}) where {R, T} = is_strongly_convex(T)

preallocate(f::MoreauEnvelope, x) = MoreauEnvelope(
    preallocate(f.g, x), f.lambda; buf = (sig = input_signature(x), y = similar(x))
)

function (f::MoreauEnvelope)(x)
    R = eltype(x)
    z = get_buffers(f, x).y
    g_prox = prox!(z, f.g, x, f.lambda)
    return g_prox + R(1) / (2 * f.lambda) * normdiff2(z, x)
end

function gradient!(grad, f::MoreauEnvelope, x)
    R = eltype(x)
    g_prox = prox!(grad, f.g, x, f.lambda)
    grad .= (x .- grad)./f.lambda
    fx = g_prox + f.lambda / R(2) * norm(grad)^2
    return fx
end

function prox!(u, f::MoreauEnvelope, x, gamma)
    # See: Thm. 6.63 in A. Beck, "First-Order Methods in Optimization", MOS-SIAM Series on Optimization, SIAM, 2017
    R = eltype(x)
    gamma_lambda = gamma + f.lambda
    y = get_buffers(f, x).y
    fy = prox!(y, f.g, x, gamma_lambda)
    alpha = gamma / gamma_lambda
    u .= ((1 - alpha) .* x) .+ (alpha .* y)
    return fy + R(1) / (2 * f.lambda) * normdiff2(u, y)
end

# NOTE the following is just so we can use certain test helpers
# TODO properly implement the following
prox_naive(f::MoreauEnvelope, x, gamma) = prox(f, x, gamma)

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:MoreauEnvelope{<:Any, T}}) where T = device_tier(T)
