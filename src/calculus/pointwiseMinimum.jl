export PointwiseMinimum

"""
    PointwiseMinimum(f_1, ..., f_k)

Given functions `f_1` to `f_k`, return their pointwise minimum, that is function
```math
g(x) = \\min\\{f_1(x), ..., f_k(x)\\}
```
Note that `g` is a nonconvex function in general.
"""
struct PointwiseMinimum{T, B}
    fs::T
    buf::B
    # explicit inner constructor: it suppresses the default two-argument outer
    # one, which would otherwise capture `PointwiseMinimum(f1, f2)` with `f2` as
    # the buffer
    PointwiseMinimum{T, B}(fs::T, buf::B) where {T, B} = new(fs, buf)
end

PointwiseMinimum(fs...) = PointwiseMinimum{typeof(fs), Nothing}(fs, nothing)

pointwise_minimum(fs::T; buf=nothing) where T <: Tuple =
    PointwiseMinimum{T, typeof(buf)}(fs, buf)

component_types(::Type{<:PointwiseMinimum{T}}) where T = fieldtypes(T)

preallocate(g::PointwiseMinimum, x) = pointwise_minimum(
    map(f -> preallocate(f, x), g.fs);
    buf = (sig = input_signature(x), y_temp = similar(x)),
)

@generated is_set_indicator(::Type{T}) where T <: PointwiseMinimum = return all(is_set_indicator, component_types(T)) ? :(true) : :(false)
@generated is_cone_indicator(::Type{T}) where T <: PointwiseMinimum = return all(is_cone_indicator, component_types(T)) ? :(true) : :(false)

function (g::PointwiseMinimum{T})(x) where T
    return minimum(f(x) for f in g.fs)
end

function prox!(y, g::PointwiseMinimum, x, gamma)
    R = real(eltype(x))
    y_temp = get_buffers(g, x).y_temp
    minimum_moreau_env = Inf
    for f in g.fs
        f_y_temp = prox!(y_temp, f, x, gamma)
        moreau_env = f_y_temp + R(1)/(2*gamma)*normdiff2(x, y_temp)
        if moreau_env <= minimum_moreau_env
            copyto!(y, y_temp)
            minimum_moreau_env = moreau_env
        end
    end
    return g(y)
end

function prox_naive(g::PointwiseMinimum, x, gamma)
    R = real(eltype(x))
    proxes = [prox_naive(f, x, gamma) for f in g.fs]
    moreau_envs = [f_y + R(1)/(R(2)*gamma)*norm(x - y)^2 for (y, f_y) in proxes]
    _, i_min = findmin(moreau_envs)
    y = proxes[i_min][1]
    return y, minimum(f(y) for f in g.fs)
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{T}) where T <: PointwiseMinimum = _combine_tiers(map(device_tier, component_types(T))...)
