# logarithmic barrier function

export LogBarrier

"""
    LogBarrier(a=1, b=0, μ=1; threaded=true)

Return the function
```math
f(x) = -μ⋅∑_i\\log(a⋅x_i+b),
```
for a nonnegative parameter `μ`.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct LogBarrier{R, S, T, Th}
    a::R
    b::S
    mu::T
    function LogBarrier{R, S, T, Th}(a::R, b::S, mu::T) where {R, S, T, Th}
        if mu <= 0
            error("parameter mu must be positive")
        else
            new(a, b, mu)
        end
    end
end

is_separable(f::Type{<:LogBarrier}) = true
is_convex(f::Type{<:LogBarrier}) = true
is_locally_smooth(f::Type{<:LogBarrier}) = true

@threadable LogBarrier{<:Any, <:Any, <:Any, Th} Transcendental

LogBarrier(a::R=1, b::S=0, mu::T=1; threaded::Bool=true) where {R, S, T} =
    LogBarrier{R, S, T, threaded}(a, b, mu)

function (f::LogBarrier)(x)
    T = eltype(x)
    a, b = f.a, f.b
    sumf = T(0)
    vmin = T(Inf)
    strategy = execution_strategy(f, x)
    # The domain check is a `min` reduction rather than an `&` over `Bool`s: `@tturbo`
    # miscompiles a boolean reduction (it yields `UInt8` and then uses it in a boolean
    # context), while `min` is supported by every backend. `log` is clamped so the
    # vectorised form never evaluates it at a negative argument speculatively; the value
    # that produces is discarded by the check below.
    @transcendental_loop strategy reduction = ((+, sumf), (min, vmin)) for i in eachindex(x)
        v = a * x[i] + b
        vmin = min(vmin, v)
        sumf += log(max(v, T(0)))
    end
    vmin > T(0) || return T(Inf)
    return -f.mu * sumf
end

function prox!(y, f::LogBarrier, x, gamma)
    T = eltype(x)
    a, b = f.a, f.b
    par = 4 * gamma * f.mu * a * a
    sumf = T(0)
    strategy = execution_strategy(f, x)
    @transcendental_loop strategy reduction = ((+, sumf),) for i in eachindex(x, y)
        z = a * x[i] + b
        v = (z + sqrt(z * z + par)) / 2
        y[i] = (v - b) / a
        sumf += log(v)
    end
    return -f.mu * sumf
end

function prox!(y, f::LogBarrier, x, gamma::AbstractArray)
    T = eltype(x)
    a, b = f.a, f.b
    par = 4 * f.mu * a * a
    sumf = T(0)
    strategy = execution_strategy(f, x)
    @transcendental_loop strategy reduction = ((+, sumf),) for i in eachindex(x, y)
        par_i = gamma[i] * par
        z = a * x[i] + b
        v = (z + sqrt(z * z + par_i)) / 2
        y[i] = (v - b) / a
        sumf += log(v)
    end
    return -f.mu * sumf
end

function gradient!(y, f::LogBarrier, x)
    sumf = eltype(x)(0)
    a, b, mu = f.a, f.b, f.mu
    strategy = execution_strategy(f, x)
    @transcendental_loop strategy reduction = ((+, sumf),) for i in eachindex(x, y)
        logarg = a * x[i] + b
        y[i] = -mu * a / logarg
        sumf += log(logarg)
    end
    return -f.mu * sumf
end

function prox_naive(f::LogBarrier, x, gamma)
    asqr = f.a * f.a
    z = f.a * x .+ f.b
    y = ((z .+ sqrt.(z .* z .+ 4 * gamma * f.mu * asqr)) / 2 .- f.b) / f.a
    fy = -f.mu * sum(log.(f.a .* y .+ f.b))
    return y, fy
end
