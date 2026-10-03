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
- `theta=1`: extrapolation parameter.

# References
1. Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", Journal of Mathematical Imaging and Vision, vol. 40, no. 1, pp. 120-145 (2011).
2. Pock, Chambolle, "Diagonal preconditioning for first order primal-dual algorithms in convex optimization", ICCV (2011).
"""
Base.@kwdef struct ChambollePockIteration{Tx, Ty, Tg, Th, TL, TLt, Tn, Tr, Tt, Ts, Tθ}
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
end

Base.IteratorSize(::Type{<:ChambollePockIteration}) = Base.IsInfinite()

get_assumptions(::Type{<:ChambollePockIteration}) = AssumptionGroup(
    OperatorTerm(:h => (is_proximable, is_convex), :L => (is_linear,)),
    SimpleTerm(:g => (is_proximable, is_convex)),
)

mutable struct ChambollePockState{Tx, Ty, Ts}
    x::Tx
    x_prev::Tx
    xbar::Tx
    y::Ty
    y_prev::Ty
    temp_x::Tx
    temp_y::Ty
    sigma_inv::Ts
end

function ChambollePockState(iter::ChambollePockIteration)
    x = copy(iter.x0)
    y = iter.y0 === nothing ? zero(iter.L * x) : copy(iter.y0)
    return ChambollePockState(x, similar(x), copy(x), y, similar(y), similar(x), similar(y), _inv_step(iter.sigma))
end

_inv_step(σ::Number) = inv(σ)
_inv_step(σ::AbstractArray) = inv.(σ)
_inv_step(σ::Tuple) = map(_inv_step, σ)

# `f(blocks..., σ)` over the dual: on the whole of each array under one scalar step, or block by
# block of `ArrayPartition`s under a tuple of steps, one per block.
_dual_blocks(f, σ, ys...) = (f(ys..., σ); nothing)
_dual_blocks(f, σ::Tuple, ys...) = (foreach(f, map(y -> y.x, ys)..., σ); nothing)

function Base.iterate(iter::ChambollePockIteration, state::ChambollePockState = ChambollePockState(iter))
    # dual step, from the extrapolated primal point, through Moreau's identity
    # prox[σh*](v) = σ (v/σ - prox[h/σ](v/σ)): the prox of `h` itself keeps a separable `h`
    # separable, where its conjugate's prox would go through the generic, allocating path
    mul!(state.temp_y, iter.L, state.xbar)
    _dual_blocks((t, y, σ) -> (t .= y ./ σ .+ t), iter.sigma, state.temp_y, state.y)
    state.y, state.y_prev = state.y_prev, state.y
    prox!(state.y, iter.h, state.temp_y, state.sigma_inv)
    _dual_blocks((y, t, σ) -> (y .= σ .* (t .- y)), iter.sigma, state.y, state.temp_y)

    # primal step
    mul!(state.temp_x, iter.Lt, state.y)
    state.temp_x .= state.x .- iter.tau .* state.temp_x
    state.x, state.x_prev = state.x_prev, state.x
    prox!(state.x, iter.g, state.temp_x, iter.tau)

    # extrapolation
    state.xbar .= state.x .+ iter.theta .* (state.x .- state.x_prev)

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

default_stopping_criterion(tol, ::ChambollePockIteration, state::ChambollePockState) =
    sum(_cp_changes(state)) <= tol
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
