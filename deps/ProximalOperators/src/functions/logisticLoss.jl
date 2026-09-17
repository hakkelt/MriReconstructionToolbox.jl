# Logistic loss function

export LogisticLoss

"""
    LogisticLoss(y, μ=1; threaded=true)

Return the function
```math
f(x) = μ⋅∑_i log(1+exp(-y_i⋅x_i))
```
where `y` is an array and `μ` is a positive parameter.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct LogisticLoss{T, R, B, Th}
    y::T
    mu::R
    buf::B
    function LogisticLoss{T, R, B, Th}(y::T, mu::R, buf::B) where {T, R, B, Th}
        if mu <= R(0)
            error("parameter mu must be positive")
        end
        new(y, mu, buf)
    end
end

@threadable LogisticLoss{<:Any, <:Any, <:Any, Th} Transcendental

LogisticLoss(y::T, mu::R=1; buf=nothing, threaded::Bool=true) where {R, T} =
    LogisticLoss{T, R, typeof(buf), threaded}(y, mu, buf)

LogisticLoss{T, R}(y::T, mu::R) where {T, R} =
    LogisticLoss{T, R, Nothing, true}(y, mu, nothing)

preallocate(f::LogisticLoss{<:Any, <:Any, <:Any, Th}, x::AbstractArray) where Th = LogisticLoss(
    f.y, f.mu;
    buf = (sig = input_signature(x), expyz = similar(x), Fz = similar(x)),
    threaded = Th,
)

is_separable(f::Type{<:LogisticLoss}) = true
is_convex(f::Type{<:LogisticLoss}) = true
is_smooth(f::Type{<:LogisticLoss}) = true
is_proximable(f::Type{<:LogisticLoss}) = false

# f(x)  =  mu log(1 + exp(-y x))

function (f::LogisticLoss)(x)
    R = eltype(x)
    val = R(0)
    fy = f.y
    strategy = execution_strategy(f, x)
    @transcendental_loop strategy reduction = ((+, val),) for k in eachindex(x)
        expyx = exp(fy[k] * x[k])
        val += log(R(1) + R(1) / expyx)
    end
    return f.mu * val
end

# f'(x) = -mu y exp(-y x) / (1 + exp(-y x))
#       = -mu y / (1 + exp(y x))
#
# Lipschitz constant of gradient: (mu y)

function gradient!(g, f::LogisticLoss, x)
    R = eltype(x)
    val = R(0)
    fy, mu = f.y, f.mu
    strategy = execution_strategy(f, x)
    @transcendental_loop strategy reduction = ((+, val),) for k in eachindex(x, g)
        expyx = exp(fy[k] * x[k])
        g[k] = -mu * fy[k] / (R(1) + expyx)
        val += log(R(1) + R(1) / expyx)
    end
    return f.mu * val
end

# Computing proximal operator:
# z = prox(f, x, gamma)
# <==> f'(z) + (z - x)/gamma = 0
# <==> (z - x)/gamma - mu y / (1 + exp(y z)) = 0
# <==> z - x - mu gamma y / (1 + exp(y z)) = 0
#
# Indicating the above condition as F(z) = 0, then
# ==> F'(z) = 1 - (mu gamma y^2 exp(y z))/(1+exp(y z))^2
#
# Newton's method (no damping) to compute z reads:
# z_{k+1} = z_k - F(z_k)/F'(z_k)
#
# To ensure convergence of Newton's method a damping is required.
# The damping coefficient could be computed by backtracking.
#
# Alternatively we can use gradient methods with constant step size.

function prox!(z, f::LogisticLoss, x, gamma)
    R = eltype(x)
    c = R(1) / gamma # convexity modulus
    L = f.mu * maximum(abs, f.y) + c # Lipschitz constant (f.mu > 0)
    fy, mu = f.y, f.mu
    z .= x
    strategy = execution_strategy(f, x)
    if is_cpu_storage(x)
        # Written as an inline `@transcendental_loop` rather than the three-array broadcast
        # below: on device storage the broadcast is the right shape (fused, no scalar
        # indexing), but on the host it neither threads nor vectorises through
        # LoopVectorization, since `Base.broadcast` doesn't route through this package's
        # execution strategy at all. The loop below is otherwise the same iteration.
        #
        # `check_input`, not `get_buffers`: this path needs no scratch array, and calling
        # `get_buffers` anyway would allocate one on every call for a non-preallocated `f`
        # (see its docstring) for no reason. `check_input` keeps the mismatch check without
        # that cost.
        check_input(f, x)
        for _ in 1:20
            @transcendental_loop strategy for k in eachindex(x, z)
                e = exp(fy[k] * z[k])
                Fzk = z[k] - x[k] - (mu * gamma) * (fy[k] / (R(1) + e))
                z[k] -= Fzk / L
            end
        end
        val = R(0)
        @transcendental_loop strategy reduction = ((+, val),) for k in eachindex(x, z)
            e = exp(fy[k] * z[k])
            val += log(R(1) + R(1) / e)
        end
        return f.mu * val
    end
    b = get_buffers(f, x)
    expyz, Fz = b.expyz, b.Fz
    for k = 1:20
        expyz .= exp.(fy .* z)
        Fz .= z .- x .- (mu * gamma) .* (fy ./ (1 .+ expyz))
        z .-= Fz ./ L
    end
    expyz .= exp.(fy .* z)
    val = R(0)
    @transcendental_loop strategy reduction = ((+, val),) for k in eachindex(expyz)
        val += log(R(1) + R(1)/expyz[k])
    end
    return f.mu * val
end
