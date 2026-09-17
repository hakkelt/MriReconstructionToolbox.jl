# L2 norm (times a constant)

export NormL2

"""
    NormL2(λ=1; threaded=true)

With a nonnegative scalar parameter λ, return the ``L_2`` norm
```math
f(x) = λ\\cdot\\sqrt{x_1^2 + … + x_n^2}.
```

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct NormL2{R, Th}
    lambda::R
    function NormL2{R, Th}(lambda::R) where {R, Th}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_convex(f::Type{<:NormL2}) = true
is_positively_homogeneous(f::Type{<:NormL2}) = true
is_locally_smooth(f::Type{<:NormL2}) = true

@threadable NormL2{<:Any, Th} MemoryBound

NormL2(lambda::R=1; threaded::Bool=true) where R = NormL2{R, threaded}(lambda)

(f::NormL2)(x) = f.lambda * norm(x)

function prox!(y, f::NormL2, x, gamma)
    normx = norm(x)
    scale = max(0, 1 - f.lambda * gamma / normx)
    map_prox!(f, y, xi -> scale * xi, x)
    return f.lambda * scale * normx
end

function gradient!(y, f::NormL2, x)
    fx = norm(x) # Value of f, without lambda
    if fx == 0
        y .= 0
    else
        y .= (f.lambda / fx) .* x
    end
    return f.lambda * fx
end

function prox_naive(f::NormL2, x, gamma)
    normx = norm(x)
    scale = max(0, 1 -f.lambda * gamma / normx)
    y = scale * x
    return y, f.lambda * scale * normx
end
