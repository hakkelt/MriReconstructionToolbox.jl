# Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems
# with Applications to Imaging", Journal of Mathematical Imaging and Vision,
# vol. 40, no. 1, pp. 120-145 (2011).

"""
    ChambollePockIteration(; <keyword-arguments>)

Iterator implementing the Chambolle-Pock primal-dual algorithm (Algorithm 1 of [1]).

This iterator solves convex optimization problems of the form

    minimize g(x) + h(L x),

where `g` and `h` are possibly nonsmooth and proximable, and `L` is a linear mapping. Each
iteration applies `L` and its adjoint once:

    y ← prox[σh*](y + σ L x̄)
    x⁺ ← prox[τg](x - τ L' y)
    x̄ ← x⁺ + θ (x⁺ - x)

A sum of terms each composed with its own operator, `Σᵢ hᵢ(Lᵢ x)`, is this form with `L` the
vertical concatenation of the `Lᵢ` and `h` their separable sum. A least-squares term among them
is then handled through its proximal mapping, so the primal step is limited by `‖L‖` alone and not
by a gradient Lipschitz constant; [`VuCondat`](@ref) is the variant that takes a smooth term by its
gradient instead.

When `g` is `mg`-strongly convex, `mg > 0` turns on the accelerated variant (Algorithm 2 of [1]):
after each primal step, with `θ = 1/√(1 + 2 mg τ)`,

    τ ← θ τ,  σ ← σ / θ,  x̄ ← x⁺ + θ (x⁺ - x)

which keeps `τσ` fixed and improves the guaranteed rate on `‖x - x*‖²` from `O(1/N)` to
`O(1/N²)`. The guarantee is a worst case: on a problem where the unaccelerated iteration already
converges linearly, the shrinking primal step makes the accelerated one slower.

Points `x0` and `y0` are the initial primal and dual iterates. `y0` defaults to zero.

See also: [`ChambollePock`](@ref).

# Arguments
- `x0`: initial primal point.
- `y0=nothing`: initial dual point (zero when `nothing`).
- `g=Zero()`: proximable objective term.
- `h=Zero()`: proximable objective term, composed with `L`.
- `L=I`: linear operator (e.g. a matrix).
- `normL=opnorm(L)`: the operator norm `‖L‖`.
- `ratio=1`: the ratio `σ/τ` of the dual and primal step sizes of the default step sizes.
- `tau`, `sigma`: primal and dual step sizes; by default `τ = 0.99/(‖L‖√ratio)` and
  `σ = 0.99√ratio/‖L‖`, so that `τσ‖L‖² < 1`. With `h` a separable sum whose dual is an
  `ArrayPartition`, `sigma` may also be a tuple with one step per block, each a number or an
  array of the block's size (a diagonal preconditioner, as in [2]); `tau` must then be given.
- `theta=1`: extrapolation parameter; unused when `mg > 0`, which sets it every iteration.
- `mg=0`: strong convexity modulus of `g`; positive values select the accelerated variant. The
  initial steps must still satisfy `τσ‖L‖² < 1`, which the defaults do.

# References
1. Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", Journal of Mathematical Imaging and Vision, vol. 40, no. 1, pp. 120-145 (2011).
2. Pock, Chambolle, "Diagonal preconditioning for first order primal-dual algorithms in convex optimization", ICCV (2011).
"""
Base.@kwdef struct ChambollePockIteration{Tx, Ty, Tg, Th, TL, TLt, Tn, Tr, Tt, Ts, Tθ, Tm}
    g::Tg = Zero()
    h::Th = Zero()
    L::TL = I
    Lt::TLt = L'
    x0::Tx
    y0::Ty = nothing
    normL::Tn = real(eltype(x0))(opnorm(L))
    ratio::Tr = real(eltype(x0))(1)
    tau::Tt = real(eltype(x0))(0.99 / (normL * sqrt(ratio)))
    sigma::Ts = real(eltype(x0))(0.99 * sqrt(ratio) / normL)
    theta::Tθ = real(eltype(x0))(1)
    mg::Tm = real(eltype(x0))(0)
end

Base.IteratorSize(::Type{<:ChambollePockIteration}) = Base.IsInfinite()

get_assumptions(::Type{<:ChambollePockIteration}) = AssumptionGroup(
    OperatorTerm(:h => (is_proximable, is_convex), :L => (is_linear,)),
    SimpleTerm(:g => (is_proximable, is_convex)),
)

mutable struct ChambollePockState{Tx, Ty, Tt, Ts, Tsi}
    x::Tx
    x_prev::Tx
    xbar::Tx
    y::Ty
    y_prev::Ty
    temp_x::Tx
    temp_y::Ty
    tau::Tt
    sigma::Ts
    sigma_inv::Tsi
    step_scale::Tt  # σ/σ₀ = τ₀/τ, which the accelerated variant grows from 1
end

function ChambollePockState(iter::ChambollePockIteration)
    x = copy(iter.x0)
    y = iter.y0 === nothing ? zero(iter.L * x) : copy(iter.y0)
    # the accelerated variant rescales the dual steps in place, so it works on its own copy
    sigma = iszero(iter.mg) ? iter.sigma : _copy_step(iter.sigma)
    return ChambollePockState(
        x, similar(x), copy(x), y, similar(y), similar(x), similar(y), iter.tau, sigma, _inv_step(sigma),
        one(iter.tau),
    )
end

_inv_step(σ::Number) = inv(σ)
_inv_step(σ::AbstractArray) = inv.(σ)
_inv_step(σ::Tuple) = map(_inv_step, σ)

_copy_step(σ::Number) = σ
_copy_step(σ::AbstractArray) = copy(σ)
_copy_step(σ::Tuple) = map(_copy_step, σ)

# `σ` times `c`, in place for an array step.
_scale_step!(σ::Number, c) = σ * c
_scale_step!(σ::AbstractArray, c) = (σ .*= c; σ)
_scale_step!(σ::Tuple, c) = map(s -> _scale_step!(s, c), σ)

# `f(blocks..., σ)` over the dual: on the whole of each array under one scalar step, or block by
# block of `ArrayPartition`s under a tuple of steps, one per block.
_dual_blocks(f, σ, ys...) = (f(ys..., σ); nothing)
_dual_blocks(f, σ::Tuple, ys...) = (foreach(f, map(y -> y.x, ys)..., σ); nothing)

function Base.iterate(iter::ChambollePockIteration, state::ChambollePockState = ChambollePockState(iter))
    # dual step, from the extrapolated primal point, through Moreau's identity
    # prox[σh*](v) = σ (v/σ - prox[h/σ](v/σ)): the prox of `h` itself keeps a separable `h`
    # separable, where its conjugate's prox would go through the generic, allocating path
    mul!(state.temp_y, iter.L, state.xbar)
    _dual_blocks((t, y, σ) -> (t .= y ./ σ .+ t), state.sigma, state.temp_y, state.y)
    state.y, state.y_prev = state.y_prev, state.y
    prox!(state.y, iter.h, state.temp_y, state.sigma_inv)
    _dual_blocks((y, t, σ) -> (y .= σ .* (t .- y)), state.sigma, state.y, state.temp_y)

    # primal step
    mul!(state.temp_x, iter.Lt, state.y)
    state.temp_x .= state.x .- state.tau .* state.temp_x
    state.x, state.x_prev = state.x_prev, state.x
    prox!(state.x, iter.g, state.temp_x, state.tau)

    # step update of the accelerated variant, then extrapolation
    theta = iter.theta
    if !iszero(iter.mg)
        theta = oftype(theta, 1 / sqrt(1 + 2 * iter.mg * state.tau))
        state.tau *= theta
        state.sigma = _scale_step!(state.sigma, inv(theta))
        state.sigma_inv = _scale_step!(state.sigma_inv, theta)
        state.step_scale /= theta
    end
    state.xbar .= state.x .+ theta .* (state.x .- state.x_prev)

    return state, state
end

# The largest change of the last step, in the scratch arrays (free between iterations).
# `maximum(abs, ·)` rather than `norm(·, Inf)`: the dual of a sum of terms is an `ArrayPartition`,
# whose `norm` reads it element by element, which a GPU array does not allow.
function _cp_changes(state::ChambollePockState)
    state.temp_x .= state.x .- state.x_prev
    state.temp_y .= state.y .- state.y_prev
    return maximum(abs, state.temp_x), maximum(abs, state.temp_y)
end

# Each change is measured in the units of the initial steps: the primal change over `τ/τ₀`, the dual
# one over `σ/σ₀`. The accelerated variant shrinks `τ` towards zero, so its raw primal change would
# fall below `tol` long before the iterates settle.
function default_stopping_criterion(tol, ::ChambollePockIteration, state::ChambollePockState)
    dx, dy = _cp_changes(state)
    return dx * state.step_scale + dy / state.step_scale <= tol
end
default_solution(::ChambollePockIteration, state::ChambollePockState) = (state.x, state.y)
function default_iteration_summary(it, ::ChambollePockIteration, state::ChambollePockState)
    dx, dy = _cp_changes(state)
    return ("" => it, "‖x - x⁻‖" => dx, "‖y - y⁻‖" => dy)
end

"""
    ChambollePock(; <keyword-arguments>)

Constructs the Chambolle-Pock primal-dual algorithm [1].

This algorithm solves convex optimization problems of the form

    minimize g(x) + h(L x),

where `g` and `h` are possibly nonsmooth, and `L` is a linear mapping.

The returned object has type `IterativeAlgorithm{ChambollePockIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`ChambollePockIteration`](@ref), [`VuCondat`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-5`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `ChambollePockIteration` constructor upon call

# References
1. Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", Journal of Mathematical Imaging and Vision, vol. 40, no. 1, pp. 120-145 (2011).
"""
ChambollePock(;
    maxit = 10_000,
    tol = 1.0e-5,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    ChambollePockIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)
