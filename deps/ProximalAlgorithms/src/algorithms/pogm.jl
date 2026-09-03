# Kim, Fessler, "Adaptive Restart of the Optimized Gradient Method for
# Convex Optimization", J. Optim. Theory Appl. (2018) (POGM, without restart).

"""
    POGMIteration(; <keyword-arguments>)

Iterator implementing the proximal optimized gradient method (POGM) [1].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth. Currently only supports the `mf == 0`, fixed-stepsize,
no-restart case.

See also: [`POGM`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of `f`.
- `gamma=nothing`: stepsize to use, defaults to `1/Lf` if not set (but `Lf` is).

# References
1. Kim, Fessler, "Adaptive Restart of the Optimized Gradient Method for Convex Optimization", Journal of Optimization Theory and Applications (2018).
"""
Base.@kwdef struct POGMIteration{Tx,Tf,Tg,TLf,Tgamma}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    Lf::TLf = nothing
    gamma::Tgamma = Lf === nothing ? nothing : (1 / Lf)
end

Base.IteratorSize(::Type{<:POGMIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct POGMState{R,Tx}
    x::Tx             # iterate
    f_x::R            # value f at x
    grad_f_x::Tx      # gradient of f at x
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    g_z::R            # value of g at z
    res::Tx           # fixed-point residual at iterate (= z - x)
    theta::R = one(gamma)  # extrapolation "theta" parameter
    y_prev::Tx = copy(y)   # forward point at the previous iteration
    w_prev::Tx = copy(x)   # pre-prox composite point at the previous iteration
    zeta_prev::R = gamma   # prox stepsize at the previous iteration
end

function Base.iterate(iter::POGMIteration)
    x = copy(iter.x0)
    f_x, grad_f_x = value_and_gradient(iter.f, x)
    R = real(eltype(x))
    gamma = R(iter.gamma === nothing ? 1 / lower_bound_smoothness_constant(iter.f, I, x, grad_f_x) : iter.gamma)
    y = x - gamma .* grad_f_x

    # First POGM update: theta_0 = 1, so beta = 0 and only the "eta" (OGM)
    # momentum term is active; see [1] in the docstring above.
    theta = R(1)
    theta_new = (1 + sqrt(1 + 4 * theta^2)) / 2
    eta = theta / theta_new
    w = y .+ eta .* (y .- x)
    zeta = gamma * (1 + eta)
    z, g_z = prox(iter.g, w, zeta)

    state = POGMState(
        x = x,
        f_x = f_x,
        grad_f_x = grad_f_x,
        gamma = gamma,
        y = y,
        z = z,
        g_z = g_z,
        res = x - z,
        theta = theta_new,
        y_prev = copy(y),
        w_prev = w,
        zeta_prev = zeta,
    )
    return state, state
end

function Base.iterate(iter::POGMIteration, state::POGMState{R,Tx}) where {R,Tx}
    # Carry the previous prox output forward as the point where the
    # gradient is evaluated: POGM has no separate momentum-on-x step,
    # the momentum is folded directly into the pre-prox point below.
    state.x .= state.z
    state.f_x = value_and_gradient!(state.grad_f_x, iter.f, state.x)
    state.y .= state.x .- state.gamma .* state.grad_f_x

    theta_new = (1 + sqrt(1 + 4 * state.theta^2)) / 2
    beta = (state.theta - 1) / theta_new
    eta = state.theta / theta_new

    # Pre-prox composite point; safe to update w_prev in place since the
    # update is elementwise (see [1] in the docstring above).
    coef = beta * state.gamma / state.zeta_prev
    state.w_prev .=
        state.y .+ beta .* (state.y .- state.y_prev) .+ eta .* (state.y .- state.x) .-
        coef .* (state.x .- state.w_prev)
    zeta = state.gamma * (1 + beta + eta)

    state.y_prev .= state.y
    state.g_z = prox!(state.z, iter.g, state.w_prev, zeta)
    state.res .= state.x .- state.z
    state.zeta_prev = zeta
    state.theta = theta_new

    return state, state
end

default_stopping_criterion(tol, ::POGMIteration, state::POGMState) =
    norm(state.res, Inf) / state.gamma <= tol
default_solution(::POGMIteration, state::POGMState) = state.z
default_iteration_summary(it, ::POGMIteration, state::POGMState) =
    ("" => it, "f(x)" => state.f_x, "g(z)" => state.g_z, "‖x - z‖/γ" => norm(state.res, Inf) / state.gamma)

"""
    POGM(; <keyword-arguments>)

Constructs the proximal optimized gradient method (POGM) [1].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth. It has a worst-case rate a factor of 2 better than
FISTA's; see [`FastForwardBackward`](@ref) for the FISTA-type accelerated
forward-backward splitting algorithm, of which this is a momentum-update
variant.

The returned object has type `IterativeAlgorithm{POGMIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`POGMIteration`](@ref), [`FastForwardBackward`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `POGMIteration` constructor upon call

# References
1. Kim, Fessler, "Adaptive Restart of the Optimized Gradient Method for Convex Optimization", Journal of Optimization Theory and Applications (2018).
"""
POGM(;
    maxit = 10_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    POGMIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:POGMIteration}) = AssumptionGroup(
    SimpleTerm(:f => (is_smooth, is_convex)),
    SimpleTerm(:g => (is_proximable, is_convex,))
)
