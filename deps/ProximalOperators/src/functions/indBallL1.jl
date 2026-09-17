# indicator of the L1 norm ball with given radius

export IndBallL1

"""
    IndBallL1(r=1.0)

Return the indicator function of the ``L_1`` norm ball
```math
S = \\left\\{ x : \\sum_i |x_i| \\leq r \\right\\}.
```
Parameter `r` must be positive.
"""
struct IndBallL1{R, B}
    r::R
    buf::B
    function IndBallL1{R, B}(r::R, buf::B) where {R, B}
        if r <= 0
            error("parameter r must be positive")
        else
            new(r, buf)
        end
    end
end

is_convex(f::Type{<:IndBallL1}) = true
is_set_indicator(f::Type{<:IndBallL1}) = true

IndBallL1(r::R=1.0; buf=nothing) where R = IndBallL1{R, typeof(buf)}(r, buf)
IndBallL1{R}(r::R) where R = IndBallL1{R, Nothing}(r, nothing)

function preallocate(f::IndBallL1, x::AbstractArray{<:Real})
    is_cpu_storage(typeof(x)) || return IndBallL1(f.r; buf = (sig = input_signature(x),))
    abs_x = similar(x)
    return IndBallL1(f.r; buf = (
        sig = input_signature(x),
        abs_x = abs_x,
        condat_buffers(abs_x)...,
    ))
end

function preallocate(f::IndBallL1, x::AbstractArray{<:Complex})
    R = real(eltype(x))
    is_cpu_storage(typeof(x)) || return IndBallL1(f.r; buf = (sig = input_signature(x),))
    abs_x = similar(x, R)
    return IndBallL1(f.r; buf = (
        sig = input_signature(x),
        abs_x = abs_x,
        y_temp = similar(x, R),
        condat_buffers(abs_x)...,
    ))
end

function (f::IndBallL1)(x)
    R = real(eltype(x))
    # Same tolerance the other ball indicators use (`IndBallL2`, `IndSphereL2`, `IndSOC`,
    # `IndHalfspace`). The previous `f.r*eps(R)` was tight enough that a point produced by this
    # function's own prox could be reported as outside the ball: composing the prox with a linear
    # mapping (`Precompose`) leaves a rounding error of a few `n*eps` in the ℓ1 norm, which is orders
    # of magnitude above `eps(R)`.
    if isapprox_le(norm(x, 1), f.r, atol=eps(R), rtol=sqrt(eps(R)))
        return R(0)
    end
    return R(Inf)
end

# Two algorithms for one projection, chosen by storage, exactly as in `indSimplex.jl`: the
# Condat path projects `abs.(x)` onto the simplex and puts the signs back, and needs a
# scratch array plus Condat's stacks; the bisection path solves `Σ max(|xᵢ| - τ, 0) = r`
# directly out of reductions and writes the answer in one broadcast, needing no scratch and
# no scalar indexing.
function prox!(y, f::IndBallL1, x::AbstractArray{<:Real}, gamma)
    R = eltype(x)
    check_input(f, x)
    if norm(x, 1) <= f.r
        y .= x
        return R(0)
    end
    if is_cpu_storage(typeof(x))
        b = get_buffers(f, x)
        w = condat_work(f, x)
        b.abs_x .= abs.(x)
        simplex_proj_condat!(y, f.r, b.abs_x, w.v, w.v_tilde)
        y .*= sign.(x)
    else
        tau = l1_ball_threshold(x, f.r, R)
        y .= sign.(x) .* max.(abs.(x) .- tau, zero(R))
    end
    return R(0)
end

function prox!(y, f::IndBallL1, x::AbstractArray{<:Complex}, gamma)
    R = real(eltype(x))
    check_input(f, x)
    if norm(x, 1) <= f.r
        y .= x
        return R(0)
    end
    if is_cpu_storage(typeof(x))
        b = get_buffers(f, x)
        w = condat_work(f, x)
        b.abs_x .= real.(abs.(x))
        simplex_proj_condat!(b.y_temp, f.r, b.abs_x, w.v, w.v_tilde)
        y .= b.y_temp .* sign.(x)
    else
        tau = l1_ball_threshold(x, f.r, R)
        y .= sign.(x) .* max.(abs.(x) .- tau, zero(R))
    end
    return R(0)
end

function prox_naive(f::IndBallL1, x, gamma)
    R = real(eltype(x))
    # do a simple bisection (aka binary search) on λ
    L = R(0)
    U = maximum(abs, x)
    λ = L
    v = R(0)
    maxit = 120
    for _ in 1:maxit
        λ = (L + U) / 2
        v = sum(max.(abs.(x) .- λ, R(0)))
        # modify lower or upper bound
        (v < f.r) ? U = λ : L = λ
        # exit condition
        if abs(L - U) < (1 + abs(U))*eps(R)
            break
        end
    end
    return sign.(x) .* max.(R(0), abs.(x) .- λ), R(0)
end
