# Hinge loss function

export SqrHingeLoss

"""
    SqrHingeLoss(y, μ=1; threaded=true)

Return the squared Hinge loss
```math
f(x) = μ⋅∑_i \\max\\{0, 1 - y_i ⋅ x_i\\}^2,
```
where `y` is an array and `μ` is a positive parameter.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct SqrHingeLoss{R, T, Th}
    y::T
    mu::R
    function SqrHingeLoss{R, T, Th}(y::T, mu::R) where {R, T, Th}
        if mu <= 0
            error("parameter mu must be positive")
        else
            new(y, mu)
        end
    end
end

is_separable(f::Type{<:SqrHingeLoss}) = true
is_convex(f::Type{<:SqrHingeLoss}) = true
is_smooth(f::Type{<:SqrHingeLoss}) = true

@threadable SqrHingeLoss{<:Any, <:Any, Th} Arithmetic

SqrHingeLoss(b::T, mu::R=1; threaded::Bool=true) where {R, T} =
    SqrHingeLoss{R, T, threaded}(b, mu)

function (f::SqrHingeLoss)(x)
    R = eltype(x)
    return f.mu * sum(max.(R(0), (R(1) .- f.y .* x)).^2)
end

function gradient!(g, f::SqrHingeLoss, x)
    R = eltype(x)
    fy, mu = f.y, f.mu
    # Inline `@elementwise_loop`, guarded by `is_cpu_storage`, rather than unconditionally
    # calling `map_reduce_prox_idx!`'s `h`/`g` closures: those closures are a call boundary
    # that neither `@inbounds` nor `@fastmath` crosses, so their division/branch cost was
    # paid unmitigated on the threaded path (see `prox!` below for the measurement). Device
    # storage still goes through `map_reduce_prox_idx!`, whose `_map_reduce_prox_idx!` has a
    # GPU-native override in `GpuExt`; inlining the loop here would bypass that override and
    # try to scalar-index the device array instead.
    if is_cpu_storage(x)
        acc = R(0)
        @elementwise_loop execution_strategy(f, x) reduction = ((+, acc),) for i in eachindex(x, g)
            xi = x[i]
            zz = 1 - fy[i] * xi
            g[i] = zz > 0 ? -2 * mu * fy[i] * zz : R(0)
            acc += max(R(0), 1 - fy[i] * xi)^2
        end
        return f.mu * acc
    end
    acc = map_reduce_prox_idx!(
        f, g,
        function (i, xi)
            zz = 1 - fy[i] * xi
            zz > 0 ? -2 * mu * fy[i] * zz : R(0)
        end,
        (i, _) -> max(R(0), 1 - fy[i] * x[i])^2,
        x,
    )
    return f.mu * acc
end

# Inline `@elementwise_loop`, guarded by `is_cpu_storage`: going through
# `map_reduce_prox_idx!`'s `h`/`g` closures unconditionally left this kernel's division and
# bounds checks outside both `@inbounds` and `@fastmath` (neither crosses a call boundary),
# which on the threaded path serialized the division-heavy, branchy body badly enough to lose
# to the serial `@simd` path outright -- measured at 2^20 elements over 4 threads: 2192us
# threaded against 1022us serial. Writing the loop inline puts the same body under the
# threaded branch's `@inbounds @fastmath` (see `_inbounds_body` in `execution.jl`), which
# fixes it: 283us threaded against 945us serial. Device storage keeps going through
# `map_reduce_prox_idx!`, whose GPU-native override in `GpuExt` the inline loop would
# otherwise bypass, scalar-indexing the device array instead.
function prox!(z, f::SqrHingeLoss, x, gamma)
    R = eltype(x)
    fy, mu = f.y, f.mu
    if is_cpu_storage(x)
        v = R(0)
        @elementwise_loop execution_strategy(f, x) reduction = ((+, v),) for k in eachindex(x, z)
            yk = fy[k]
            xk = x[k]
            zk = yk * xk >= 1 ? xk : (xk + 2 * mu * gamma * yk) / (1 + 2 * mu * gamma * yk^2)
            z[k] = zk
            inactive = yk * xk >= 1
            r = 1 - yk * zk
            v += inactive ? R(0) : r * r
        end
        return f.mu * v
    end
    v = map_reduce_prox_idx!(
        f, z,
        function (k, xk)
            yk = fy[k]
            yk * xk >= 1 ? xk : (xk + 2 * mu * gamma * yk) / (1 + 2 * mu * gamma * yk^2)
        end,
        function (k, zk)
            yk = fy[k]
            inactive = yk * x[k] >= 1
            r = 1 - yk * zk
            inactive ? R(0) : r * r
        end,
        x,
    )
    return f.mu * v
end

function prox_naive(f::SqrHingeLoss, x, gamma)
    flag = f.y .* x .<= 1
    z = copy(x)
    z[flag] = (x[flag] .+ 2 .* f.mu .* gamma .* f.y[flag]) ./ (1 + 2 .* f.mu .* gamma .* f.y[flag].^2)
    return z, f.mu * sum(max.(0, 1 .- f.y .* z).^2)
end
