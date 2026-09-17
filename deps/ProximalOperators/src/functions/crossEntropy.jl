# Cross Entropy loss function

export CrossEntropy

"""
    CrossEntropy(b; threaded=true)

Return the function
```math
f(x) = -\\frac{1}{N} \\sum_{i = 1}^{N} b_i \\log (x_i)+(1-b_i) \\log (1-x_i),
```
where `b` is an array of length `N` such that `0 ≤ b ≤ 1` component-wise.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct CrossEntropy{T, Th}
    b::T
    function CrossEntropy{T, Th}(b::T) where {T, Th}
        if !(all(0 .<= b .<= 1. ))
            error("b must be 0 ≤ b ≤ 1 ")
        else
            new(b)
        end
    end
end

is_convex(f::Type{<:CrossEntropy}) = true
is_smooth(f::Type{<:CrossEntropy}) = true

@threadable CrossEntropy{<:Any, Th} Transcendental

CrossEntropy(b::T; threaded::Bool=true) where {T} = CrossEntropy{T, threaded}(b)

function (f::CrossEntropy)(x)
    T = eltype(x)
    fsum = T(0)
    b = f.b
    strategy = execution_strategy(f, x)
    # `b` is commonly a `Bool` array; it is converted elementwise so the kernel does all of
    # its arithmetic in `T` (`Bool` arithmetic inside `@turbo` recurses infinitely in
    # VectorizationBase)
    @transcendental_loop strategy reduction = ((+, fsum),) for i in eachindex(b)
        bi = T(b[i])
        fsum += bi * log(x[i]) + (1 - bi) * log(1 - x[i])
    end
    return -fsum/length(f.b)
end

function gradient!(y, f::CrossEntropy, x)
    T = eltype(x)
    fsum = T(0)
    b = f.b
    invn = T(1/length(f.b))
    strategy = execution_strategy(f, x)
    @transcendental_loop strategy reduction = ((+, fsum),) for i in eachindex(x, y)
        bi = T(b[i])
        y[i] = invn * (-bi/x[i] + (1 - bi)/(1 - x[i]))
        fsum += bi * log(x[i]) + (1 - bi) * log(1 - x[i])
    end
    return -1/length(f.b)*fsum
end
