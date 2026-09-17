# L1 norm (times a constant, or weighted)

export NormL1

"""
    NormL1(λ=1; threaded=true)

With a nonnegative scalar parameter λ, return the ``L_1`` norm
```math
f(x) = λ\\cdot∑_i|x_i|.
```
With a nonnegative array parameter λ, return the weighted ``L_1`` norm
```math
f(x) = ∑_i λ_i|x_i|.
```

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct NormL1{T, B, Th}
    lambda::T
    buf::B
    function NormL1{T, B, Th}(lambda::T, buf::B) where {T, B, Th}
        if !(eltype(lambda) <: Real)
            error("λ must be real")
        end
        if any(lambda .< 0)
            error("λ must be nonnegative")
        else
            new(lambda, buf)
        end
    end
end

is_proximable(f::Type{<:NormL1}) = true
is_separable(f::Type{<:NormL1}) = true
is_convex(f::Type{<:NormL1}) = true
is_positively_homogeneous(f::Type{<:NormL1}) = true
is_locally_smooth(f::Type{<:NormL1}) = true

@threadable NormL1{<:Any, <:Any, Th} Arithmetic

NormL1(lambda::R=1; buf=nothing, threaded::Bool=true) where R =
    NormL1{R, typeof(buf), threaded}(lambda, buf)
NormL1{R}(lambda::R) where R = NormL1{R, Nothing, true}(lambda, nothing)

# only the weighted variant needs scratch space, for the ∑_i λ_i|y_i| it returns
preallocate(f::NormL1{<:AbstractArray, <:Any, Th}, x::AbstractArray) where Th = NormL1(
    f.lambda;
    buf = (sig = input_signature(x), lam_abs_y = similar(x, real(eltype(x)))),
    threaded = Th,
)

(f::NormL1)(x) = f.lambda * norm(x, 1)

(f::NormL1{<:AbstractArray})(x) = norm(f.lambda .* x, 1)

# ∑_i λ_i |y_i|, computed through the buffer so that the summation order (and
# therefore the result, bit for bit) is the same as `sum(lambda .* abs.(y))`
@inline function weighted_l1(f::NormL1{<:AbstractArray}, x, y)
    t = get_buffers(f, x).lam_abs_y
    t .= f.lambda .* abs.(y)
    return sum(t)
end

# soft-thresholding, the one kernel this whole file is made of
@inline soft_threshold(xi::Real, gl) = xi + (xi <= -gl ? gl : (xi >= gl ? -gl : -xi))
@inline soft_threshold(xi::Complex, gl) = sign(xi) * (abs(xi) <= gl ? zero(abs(xi)) : abs(xi) - gl)

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Real}, gamma)
    @assert length(y) == length(x) == length(f.lambda)
    map_prox_idx!(f, y, (i, xi) -> soft_threshold(xi, gamma * f.lambda[i]), x)
    return weighted_l1(f, x, y)
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Complex}, gamma)
    @assert length(y) == length(x) == length(f.lambda)
    map_prox_idx!(f, y, (i, xi) -> soft_threshold(xi, gamma * f.lambda[i]), x)
    return weighted_l1(f, x, y)
end

function prox!(y, f::NormL1, x::AbstractArray{<:Real}, gamma)
    @assert length(y) == length(x)
    gl = gamma * f.lambda
    n1y = map_reduce_prox!(f, y, xi -> soft_threshold(xi, gl), yi -> yi > 0 ? yi : -yi, x)
    return f.lambda * n1y
end

function prox!(y, f::NormL1, x::AbstractArray{<:Complex}, gamma)
    @assert length(y) == length(x)
    gl = gamma * f.lambda
    n1y = map_reduce_prox!(f, y, xi -> soft_threshold(xi, gl), abs, x)
    return f.lambda * n1y
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Real}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(f.lambda) == length(gamma)
    map_prox_idx!(f, y, (i, xi) -> soft_threshold(xi, gamma[i] * f.lambda[i]), x)
    return weighted_l1(f, x, y)
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Complex}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(f.lambda) == length(gamma)
    map_prox_idx!(f, y, (i, xi) -> soft_threshold(xi, gamma[i] * f.lambda[i]), x)
    return weighted_l1(f, x, y)
end

function prox!(y, f::NormL1, x::AbstractArray{<:Real}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(gamma)
    n1y = map_reduce_prox_idx!(
        f, y, (i, xi) -> soft_threshold(xi, gamma[i] * f.lambda), (i, yi) -> yi > 0 ? yi : -yi, x
    )
    return f.lambda * n1y
end

function prox!(y, f::NormL1, x::AbstractArray{<:Complex}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(gamma)
    n1y = map_reduce_prox_idx!(
        f, y, (i, xi) -> soft_threshold(xi, gamma[i] * f.lambda), (i, yi) -> abs(yi), x
    )
    return f.lambda * n1y
end

function gradient!(y, f::NormL1, x)
    y .= f.lambda .* sign.(x)
    return f(x)
end

function prox_naive(f::NormL1, x, gamma)
    y = sign.(x).*max.(0, abs.(x) .- gamma .* f.lambda)
    return y, norm(f.lambda .* y,1)
end
