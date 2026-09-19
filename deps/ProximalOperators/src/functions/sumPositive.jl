# Sum of the positive components

export SumPositive

"""
    SumPositive(; threaded=true)

Return the function
```math
f(x) = ∑_i \\max\\{0, x_i\\}.
```

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct SumPositive{Th} end

SumPositive(; threaded::Bool=true) = SumPositive{threaded}()

is_separable(f::Type{<:SumPositive}) = true
is_convex(f::Type{<:SumPositive}) = true
is_positively_homogeneous(f::Type{<:SumPositive}) = true
is_locally_smooth(f::Type{<:SumPositive}) = true

@threadable SumPositive{Th} Arithmetic

@threadable SumPositive{Th} Arithmetic

function (::SumPositive)(x)
    return sum(xi -> max(xi, eltype(x)(0)), x)
end

function prox!(y, f::SumPositive, x, gamma)
    R = eltype(x)
    fsum = map_reduce_prox!(
        f, y,
        xi -> xi < gamma ? (xi > 0 ? R(0) : xi) : xi - gamma,
        yi -> yi > 0 ? yi : R(0),
        x,
    )
    return fsum
end

function gradient!(y, f::SumPositive, x)
    R = eltype(x)
    y .= max.(0, sign.(x))
    return sum(xi -> max(xi, R(0)), x)
end

function prox_naive(::SumPositive, x, gamma)
    R = eltype(x)
    y = copy(x)
    indpos = x .> 0
    y[indpos] = max.(R(0), x[indpos] .- gamma)
    return y, sum(max.(R(0), y))
end

# ######################### #
# Prox with multiple gammas #
# ######################### #

function prox!(y, f::SumPositive, x, gamma::AbstractArray)
    R = eltype(x)
    fsum = map_reduce_prox_idx!(
        f, y,
        (i, xi) -> xi < gamma[i] ? (xi > 0 ? R(0) : xi) : xi - gamma[i],
        (i, yi) -> yi > 0 ? yi : R(0),
        x,
    )
    return fsum
end

function prox_naive(::SumPositive, x, gamma::AbstractArray)
    R = eltype(x)
    y = copy(x)
    indpos = x .> 0
    y[indpos] = max.(R(0), x[indpos] .- gamma[indpos])
    return y, sum(max.(R(0), y))
end
