# elastic-net regularization

export ElasticNet

"""
    ElasticNet(μ=1, λ=1; threaded=true)

Return the function
```math
f(x) = μ\\|x\\|_1 + (λ/2)\\|x\\|^2,
```
for nonnegative parameters `μ` and `λ`.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct ElasticNet{R, S, Th}
    mu::R
    lambda::S
    function ElasticNet{R, S, Th}(mu::R, lambda::S) where {R, S, Th}
        if lambda < 0 || mu < 0
            error("parameters `μ` and `λ` must be nonnegative")
        else
            new(mu, lambda)
        end
    end
end

is_separable(f::Type{<:ElasticNet}) = true
is_proximable(f::Type{<:ElasticNet}) = true
is_convex(f::Type{<:ElasticNet}) = true
is_locally_smooth(f::Type{<:ElasticNet}) = true

@threadable ElasticNet{<:Any, <:Any, Th} Arithmetic

ElasticNet(mu::R=1, lambda::S=1; threaded::Bool=true) where {R, S} =
    ElasticNet{R, S, threaded}(mu, lambda)

function (f::ElasticNet)(x)
    R = real(eltype(x))
    return f.mu * norm(x, 1) + f.lambda / R(2) * norm(x, 2)^2
end

function prox!(y, f::ElasticNet, x, gamma)
    R = real(eltype(x))
    gm = gamma * f.mu
    gl = gamma * f.lambda
    sqnorm2x, norm1x = map_reduce2_prox!(
        f, y, xi -> (xi + (xi <= -gm ? gm : (xi >= gm ? -gm : -xi))) / (1 + gl), abs2, abs, x
    )
    return f.mu * norm1x + f.lambda / R(2) * sqnorm2x
end

function prox!(y, f::ElasticNet, x, gamma::AbstractArray)
    R = real(eltype(x))
    mu, lambda = f.mu, f.lambda
    sqnorm2x, norm1x = map_reduce2_prox_idx!(
        f, y,
        function (i, xi)
            gm = gamma[i] * mu
            gl = gamma[i] * lambda
            (xi + (xi <= -gm ? gm : (xi >= gm ? -gm : -xi))) / (1 + gl)
        end,
        abs2, abs, x,
    )
    return f.mu * norm1x + f.lambda / R(2) * sqnorm2x
end

function prox!(y, f::ElasticNet, x::AbstractArray{<:Complex}, gamma)
    R = real(eltype(x))
    gm = gamma * f.mu
    gl = gamma * f.lambda
    sqnorm2x, norm1x = map_reduce2_prox!(
        f, y, xi -> sign(xi) * max(0, abs(xi) - gm) / (1 + gl), abs2, abs, x
    )
    return f.mu * norm1x + f.lambda / R(2) * sqnorm2x
end

function prox!(y, f::ElasticNet, x::AbstractArray{<:Complex}, gamma::AbstractArray)
    R = real(eltype(x))
    mu, lambda = f.mu, f.lambda
    sqnorm2x, norm1x = map_reduce2_prox_idx!(
        f, y,
        function (i, xi)
            gm = gamma[i] * mu
            gl = gamma[i] * lambda
            sign(xi) * max(0, abs(xi) - gm) / (1 + gl)
        end,
        abs2, abs, x,
    )
    return f.mu * norm1x + f.lambda / R(2) * sqnorm2x
end

function gradient!(y, f::ElasticNet, x)
    R = real(eltype(x))
    # Gradient of 1 norm
    y .= f.mu .* sign.(x)
    # Gradient of 2 norm
    y .+= f.lambda .* x
    return f.mu * norm(x, 1) + f.lambda / R(2) * norm(x, 2)^2
end

function prox_naive(f::ElasticNet, x, gamma)
    R = real(eltype(x))
    uz = max.(0, abs.(x) .- gamma .* f.mu)./(1 .+ f.lambda .* gamma)
    return sign.(x) .* uz, f.mu * norm(uz, 1) + f.lambda / R(2) * norm(uz)^2
end
