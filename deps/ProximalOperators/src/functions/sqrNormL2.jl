# squared L2 norm (times a constant, or weighted)

export SqrNormL2

"""
    SqrNormL2(λ=1; threaded=true)

With a nonnegative scalar `λ`, return the squared Euclidean norm
```math
f(x) = \\tfrac{λ}{2}\\|x\\|^2.
```
With a nonnegative array `λ`, return the weighted squared Euclidean norm
```math
f(x) = \\tfrac{1}{2}∑_i λ_i x_i^2.
```
An array `λ` whose size differs from `x`'s broadcasts against it, as `λ .* x` would: weights per
row of a matrix `x` are a column `λ`, and they are never expanded to the size of `x`.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct SqrNormL2{T,SC,Th}
    lambda::T
    function SqrNormL2{T,SC,Th}(lambda::T) where {T,SC,Th}
        if any(lambda .< 0)
            error("coefficients in λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_proximable(::Type{<:SqrNormL2}) = true
is_convex(::Type{<:SqrNormL2}) = true
is_smooth(::Type{<:SqrNormL2}) = true
is_separable(::Type{<:SqrNormL2}) = true
is_generalized_quadratic(::Type{<:SqrNormL2}) = true
is_strongly_convex(::Type{<:SqrNormL2{T,SC}}) where {T,SC} = SC

@threadable SqrNormL2{<:Any,<:Any,Th} MemoryBound

SqrNormL2(lambda::T=1; threaded::Bool=true) where T =
    SqrNormL2{T,all(lambda .> 0),threaded}(lambda)

function (f::SqrNormL2{<:Real})(x)
    return f.lambda / real(eltype(x))(2) * norm(x)^2
end

function (f::SqrNormL2{<:AbstractArray})(x)
    R = real(eltype(x))
    lambda = f.lambda
    size(lambda) == size(x) || return _weighted_sqrnorm(lambda, x) / R(2)
    sqnorm = reduce_call_idx(f, (k, xk) -> lambda[k] * abs2(xk), x)
    return sqnorm / R(2)
end

function gradient!(y, f::SqrNormL2{<:Real}, x)
    R = real(eltype(x))
    lambda = f.lambda
    # The reduction is over `x`, not over the written `y`, so no `map_reduce_prox!` seam
    # fits and the loop is spelled out. It is therefore guarded by `is_cpu_storage`, like
    # the inlined loops in `sqrHingeLoss.jl`: on device storage it would scalar-index.
    if is_cpu_storage(x)
        sqnx = R(0)
        strategy = execution_strategy(f, x)
        @elementwise_loop strategy reduction = ((+, sqnx),) for k in eachindex(x, y)
            y[k] = lambda * x[k]
            sqnx += abs2(x[k])
        end
        return f.lambda / R(2) * sqnx
    end
    y .= lambda .* x
    return f.lambda / R(2) * sum(abs2, x)
end

function gradient!(y, f::SqrNormL2{<:AbstractArray}, x)
    R = real(eltype(x))
    lambda = f.lambda
    if size(lambda) != size(x)
        y .= lambda .* x
        return _weighted_sqrnorm(lambda, x) / R(2)
    end
    if is_cpu_storage(x)
        sqnx = R(0)
        strategy = execution_strategy(f, x)
        @elementwise_loop strategy reduction = ((+, sqnx),) for k in eachindex(x, y)
            y[k] = lambda[k] * x[k]
            sqnx += lambda[k] * abs2(x[k])
        end
        return sqnx / R(2)
    end
    y .= lambda .* x
    return _weighted_sqrnorm(lambda, x) / R(2)
end

function prox!(y, f::SqrNormL2{<:Real}, x, gamma::Number)
    R = real(eltype(x))
    gl = gamma * f.lambda
    sqny = map_reduce_prox!(f, y, xk -> xk / (1 + gl), abs2, x)
    return f.lambda / R(2) * sqny
end

function prox!(y, f::SqrNormL2{<:AbstractArray}, x, gamma::Number)
    R = real(eltype(x))
    lambda = f.lambda
    if size(lambda) != size(x)
        y .= x ./ (1 .+ gamma .* lambda)
        return _weighted_sqrnorm(lambda, y) / R(2)
    end
    wsqny = map_reduce_prox_idx!(
        f, y, (k, xk) -> xk / (1 + gamma * lambda[k]), (k, yk) -> lambda[k] * abs2(yk), x
    )
    return wsqny / R(2)
end

function prox!(y, f::SqrNormL2{<:Real}, x, gamma::AbstractArray)
    R = real(eltype(x))
    lambda = f.lambda
    sqny = map_reduce_prox_idx!(
        f, y, (k, xk) -> xk / (1 + gamma[k] * lambda), (k, yk) -> abs2(yk), x
    )
    return f.lambda / R(2) * sqny
end

function prox!(y, f::SqrNormL2{<:AbstractArray}, x, gamma::AbstractArray)
    R = real(eltype(x))
    lambda = f.lambda
    if size(lambda) != size(x)
        y .= x ./ (1 .+ gamma .* lambda)
        return _weighted_sqrnorm(lambda, y) / R(2)
    end
    wsqny = map_reduce_prox_idx!(
        f, y, (k, xk) -> xk / (1 + gamma[k] * lambda[k]), (k, yk) -> lambda[k] * abs2(yk), x
    )
    return wsqny / R(2)
end

# `∑ λᵢ |xᵢ|²` with `λ` broadcast against `x`, reduced without materialising the product.
_weighted_sqrnorm(lambda, x) =
    real(sum(Broadcast.instantiate(Broadcast.broadcasted((l, xk) -> l * abs2(xk), lambda, x))))

function prox_naive(f::SqrNormL2, x, gamma)
    R = real(eltype(x))
    y = x./(R(1) .+ f.lambda .* gamma)
    return y, real(sum(f.lambda .* abs2.(y))) / R(2)
end
