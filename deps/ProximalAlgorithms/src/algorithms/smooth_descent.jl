# Nocedal, Wright, "Numerical Optimization", 2nd edition, Springer (2006),
# chapters 3 (line search), 5 (nonlinear conjugate gradient) and 7 (L-BFGS).
#
# Fessler, Booth, "Conjugate-gradient preconditioning methods for shift-variant PET image
# reconstruction", IEEE Transactions on Image Processing, vol. 8, no. 5, pp. 688-699 (1999):
# a line search along d for a cost Σᵢ fᵢ(Lᵢ x) that reuses Lᵢ x and Lᵢ d.

# Both algorithms minimize Σᵢ fᵢ(Lᵢ x) with every fᵢ smooth, keeping uᵢ = Lᵢ x and vᵢ = Lᵢ d. Along
# a direction d the cost is φ(α) = Σᵢ fᵢ(uᵢ + α vᵢ), so the line search evaluates φ and φ' on those
# cached arrays and applies no operator: an iteration costs one Lᵢ and one Lᵢ' per term, as a
# gradient step does. A quadratic fᵢ contributes a quadratic in α, fixed by fᵢ(uᵢ), ⟨vᵢ, ∇fᵢ(uᵢ)⟩
# and the curvature ⟨vᵢ, ∇fᵢ(uᵢ + vᵢ) - ∇fᵢ(uᵢ)⟩, so its gradient is evaluated once per iteration
# however many trials the search takes. That matters when an fᵢ carries an operator of its own
# (a least-squares term through its normal operator, with Lᵢ the identity).
#
# The gradient followed is Σᵢ Lᵢ'∇fᵢ(Lᵢ x). When an `Lᵢ'` is a positive multiple of the adjoint,
# `Lᵢ' = Lᵢᴴ/sᵢ` (a Fourier transform whose adjoint is normalised as its inverse, say), that is the
# gradient of Σᵢ fᵢ(Lᵢ x)/sᵢ, and the line search must minimise that cost for the two to agree. Each
# sᵢ is measured once, from the first direction with Lᵢ d ≠ 0, as ‖Lᵢ d‖² / ⟨d, Lᵢ'Lᵢ d⟩; it is 1
# for a true adjoint pair.

_as_tuple(x::Tuple) = x
_as_tuple(x) = (x,)

_terms(f, ::Nothing) = (_as_tuple(f), map(_ -> I, _as_tuple(f)))
_terms(f, L) = (_as_tuple(f), _as_tuple(L))

_is_quadratic(f) = ProximalCore.is_generalized_quadratic(f) && is_smooth(f)

mutable struct SmoothDescentState{Tx, Tu, Tq, Q, R, H}
    x::Tx
    d::Tx                     # search direction
    grad::Tx                  # Σᵢ Lᵢ'∇fᵢ(Lᵢ x)
    grad_prev::Tx
    temp::Tx
    u::Tu                     # Lᵢ x, per term
    v::Tu                     # Lᵢ d
    w::Tu                     # Lᵢ x + α Lᵢ d, the line search's trial point
    gu::Tu                    # ∇fᵢ(uᵢ)
    gw::Tu                    # ∇fᵢ(wᵢ)
    gq::Tq                    # ∇fᵢ(uᵢ + vᵢ) - ∇fᵢ(uᵢ) of a quadratic fᵢ, `nothing` otherwise
    quadratic::Q
    fu::Vector{R}             # fᵢ(uᵢ)
    fw::Vector{R}             # fᵢ(wᵢ)
    slope::Vector{R}          # ⟨vᵢ, ∇fᵢ(uᵢ)⟩
    curvature::Vector{R}      # ⟨vᵢ, gqᵢ⟩
    scaling::Vector{R}        # sᵢ, NaN until measured
    f_x::R
    alpha::R
    step::R                   # ‖α d‖∞ of the last step
    hessian::H                # L-BFGS inverse-Hessian approximation, `nothing` for NCG
    it::Int
end

function SmoothDescentState(f, L, Lt, x0, scaling, hessian)
    R = real(eltype(x0))
    x = copy(x0)
    u = map(Li -> Li * x, L)
    gu = map(similar, u)
    quadratic = map(_is_quadratic, f)
    gq = map((q, ui) -> q ? similar(ui) : nothing, quadratic, u)
    m = length(f)
    fu = [R(value_and_gradient!(gu[i], f[i], u[i])) for i in 1:m]
    grad = zero(x)
    temp = similar(x)
    for i in 1:m
        mul!(temp, Lt[i], gu[i])
        grad .+= temp
    end
    s = scaling === nothing ? fill(R(NaN), m) : R.(collect(_as_tuple(scaling)))
    return SmoothDescentState(
        x, -grad, grad, similar(x), temp, u, map(similar, u), map(similar, u), gu, map(similar, u),
        gq, quadratic, fu, copy(fu), zeros(R, m), zeros(R, m), s,
        sum(fu), zero(R), R(Inf), hessian(x), 0,
    )
end

_scale(state, i) = isnan(state.scaling[i]) ? one(state.alpha) : state.scaling[i]

# φ(α) and φ'(α). A non-quadratic term leaves its trial point and gradient in `state.w` / `state.gw`;
# a quadratic one is evaluated from its coefficients and touches no array.
function _phi!(state, f, α)
    R = typeof(state.alpha)
    val, der = zero(R), zero(R)
    for i in eachindex(f)
        s = _scale(state, i)
        if state.quadratic[i]
            val += (state.fu[i] + α * state.slope[i] + α^2 * state.curvature[i] / 2) / s
            der += (state.slope[i] + α * state.curvature[i]) / s
        else
            state.w[i] .= state.u[i] .+ α .* state.v[i]
            state.fw[i] = R(value_and_gradient!(state.gw[i], f[i], state.w[i]))
            val += state.fw[i] / s
            der += real(dot(state.v[i], state.gw[i])) / s
        end
    end
    return val, der
end

# A step α along d with |φ'(α)| ≤ eta·|φ'(0)|: secant steps on φ', extrapolating until φ' changes
# sign and then Illinois regula falsi inside the bracket. For a quadratic cost φ' is affine and the
# first secant step is exact. The last `_phi!` evaluated is at the α returned.
function _line_search!(state, f, dphi0, alpha0, eta, maxit)
    R = typeof(state.alpha)
    target = eta * abs(dphi0)
    lo, dlo = zero(R), dphi0                     # φ'(lo) < 0
    hi, dhi = R(Inf), zero(R)                    # φ'(hi) > 0 once found
    side = 0
    α = alpha0
    phi, dphi = _phi!(state, f, α)
    best = (abs(dphi), α)
    for _ in 1:maxit
        abs(dphi) <= target && return α, phi
        if dphi < 0
            prev, dprev = lo, dlo
            lo, dlo = α, dphi
            side == -1 && isfinite(hi) && (dhi /= 2)
            side = -1
        else
            hi, dhi = α, dphi
            side == 1 && (dlo /= 2)
            side = 1
        end
        if isfinite(hi)
            α = lo - dlo * (hi - lo) / (dhi - dlo)
        else
            # extrapolate from the two lowest points; φ' not increasing means no curvature to use
            α = dlo > dprev ? lo - dlo * (lo - prev) / (dlo - dprev) : 4 * lo
            α = clamp(α, 2 * lo, 1000 * lo)
        end
        phi, dphi = _phi!(state, f, α)
        abs(dphi) < best[1] && (best = (abs(dphi), α))
    end
    abs(dphi) <= target && return α, phi
    # out of trials: the best point seen, evaluated again so the trial arrays match it
    α = best[2]
    phi, _ = _phi!(state, f, α)
    return α, phi
end

function _measure_scaling!(state, Lt)
    for i in eachindex(state.scaling)
        isnan(state.scaling[i]) || continue
        nv = real(dot(state.v[i], state.v[i]))
        iszero(nv) && continue
        mul!(state.temp, Lt[i], state.v[i])
        state.scaling[i] = nv / real(dot(state.d, state.temp))
    end
    return state
end

# One step along `state.d`: line search, then x, Lᵢ x and the gradient at the new point.
function _descend!(state, f, L, Lt, alpha0, eta, ls_maxit, refresh)
    R = typeof(state.alpha)
    for i in eachindex(f)
        mul!(state.v[i], L[i], state.d)
    end
    _measure_scaling!(state, Lt)
    dphi0 = zero(R)
    for i in eachindex(f)
        state.slope[i] = real(dot(state.v[i], state.gu[i]))
        dphi0 += state.slope[i] / _scale(state, i)
        if state.quadratic[i]
            state.w[i] .= state.u[i] .+ state.v[i]
            value_and_gradient!(state.gq[i], f[i], state.w[i])
            state.gq[i] .-= state.gu[i]
            state.curvature[i] = real(dot(state.v[i], state.gq[i]))
        end
    end
    α, phi = _line_search!(state, f, dphi0, alpha0, eta, ls_maxit)
    for i in eachindex(f)
        if state.quadratic[i]
            state.w[i] .= state.u[i] .+ α .* state.v[i]
            state.gw[i] .= state.gu[i] .+ α .* state.gq[i]
            state.fw[i] = state.fu[i] + α * state.slope[i] + α^2 * state.curvature[i] / 2
        end
    end
    state.x .+= α .* state.d
    state.u, state.w = state.w, state.u
    state.gu, state.gw = state.gw, state.gu
    state.fu, state.fw = state.fw, state.fu
    state.f_x, state.alpha = phi, α
    state.step = α * norm(state.d, Inf)
    state.it += 1
    if refresh > 0 && state.it % refresh == 0
        # Lᵢ x and the gradients drift from their running updates in floating point; recompute
        # them now and then
        for i in eachindex(f)
            mul!(state.u[i], L[i], state.x)
            state.fu[i] = R(value_and_gradient!(state.gu[i], f[i], state.u[i]))
        end
    end
    state.grad_prev, state.grad = state.grad, state.grad_prev
    fill!(state.grad, 0)
    for i in eachindex(f)
        mul!(state.temp, Lt[i], state.gu[i])
        state.grad .+= state.temp
    end
    return state
end

_first_alpha(state) = one(state.alpha) / max(norm(state.grad), floatmin(state.alpha))

"""
    NonlinearCGIteration(; <keyword-arguments>)

Iterator implementing the nonlinear conjugate gradient method (Polak-Ribière+, see [1, §5.2])
for the smooth problem

    minimize Σᵢ fᵢ(Lᵢ x),

where every `fᵢ` is smooth and every `Lᵢ` linear. The step along each direction comes from a line
search on the cached products `Lᵢ x` and `Lᵢ d` ([2]), so an iteration applies each `Lᵢ` and its
adjoint once, and no Lipschitz constant is needed. The direction restarts at the negative gradient
whenever it stops being a descent direction.

See also: [`NonlinearCG`](@ref), [`LimitedMemoryBFGSIteration`](@ref).

# Arguments
- `x0`: initial point.
- `f`: smooth function, or tuple of smooth functions `fᵢ`.
- `L=nothing`: linear operator, or tuple of operators `Lᵢ` (one per `fᵢ`); identity when `nothing`.
- `eta=0.1`: line search accuracy, the accepted `|φ'(α)|` relative to `|φ'(0)|`.
- `ls_maxit=10`: line search trials per iteration.
- `refresh=50`: every how many iterations `Lᵢ x` and the gradients are recomputed instead of
  updated (`0`: never).
- `adjoint_scaling=nothing`: the factors `sᵢ` with `Lᵢ' = Lᵢᴴ/sᵢ`, measured once when `nothing`.

# References
1. Nocedal, Wright, "Numerical Optimization", 2nd edition, Springer (2006).
2. Fessler, Booth, "Conjugate-gradient preconditioning methods for shift-variant PET image reconstruction", IEEE Transactions on Image Processing, vol. 8, no. 5, pp. 688-699 (1999).
"""
Base.@kwdef struct NonlinearCGIteration{Tx, Tf, TL, R, S}
    x0::Tx
    f::Tf
    L::TL = nothing
    eta::R = real(eltype(x0))(0.1)
    ls_maxit::Int = 10
    refresh::Int = 50
    adjoint_scaling::S = nothing
end

"""
    LimitedMemoryBFGSIteration(; <keyword-arguments>)

Iterator implementing the limited-memory BFGS method ([1, §7.2]) for the smooth problem

    minimize Σᵢ fᵢ(Lᵢ x),

where every `fᵢ` is smooth and every `Lᵢ` linear, with the same operator-free line search as
[`NonlinearCGIteration`](@ref): an iteration applies each `Lᵢ` and its adjoint once.

See also: [`LimitedMemoryBFGS`](@ref), [`NonlinearCGIteration`](@ref).

# Arguments
- `x0`: initial point.
- `f`: smooth function, or tuple of smooth functions `fᵢ`.
- `L=nothing`: linear operator, or tuple of operators `Lᵢ` (one per `fᵢ`); identity when `nothing`.
- `memory=5`: number of stored correction pairs.
- `eta=0.9`: line search accuracy, the accepted `|φ'(α)|` relative to `|φ'(0)|`.
- `ls_maxit=10`: line search trials per iteration.
- `refresh=50`: every how many iterations `Lᵢ x` and the gradients are recomputed instead of
  updated (`0`: never).
- `adjoint_scaling=nothing`: the factors `sᵢ` with `Lᵢ' = Lᵢᴴ/sᵢ`, measured once when `nothing`.

# References
1. Nocedal, Wright, "Numerical Optimization", 2nd edition, Springer (2006).
"""
Base.@kwdef struct LimitedMemoryBFGSIteration{Tx, Tf, TL, R, S}
    x0::Tx
    f::Tf
    L::TL = nothing
    memory::Int = 5
    eta::R = real(eltype(x0))(0.9)
    ls_maxit::Int = 10
    refresh::Int = 50
    adjoint_scaling::S = nothing
end

const SmoothDescentIteration = Union{NonlinearCGIteration, LimitedMemoryBFGSIteration}

Base.IteratorSize(::Type{<:SmoothDescentIteration}) = Base.IsInfinite()

get_assumptions(::Type{<:SmoothDescentIteration}) = AssumptionGroup(
    RepeatedOperatorTerm(:f => (is_smooth,), :L => (is_linear,)),
)

_hessian(::NonlinearCGIteration) = x -> nothing
_hessian(iter::LimitedMemoryBFGSIteration) = x -> LBFGSOperator(iter.memory, x)

function _start(iter::SmoothDescentIteration)
    f, L = _terms(iter.f, iter.L)
    Lt = map(adjoint, L)
    return f, L, Lt, SmoothDescentState(f, L, Lt, iter.x0, iter.adjoint_scaling, _hessian(iter))
end

function Base.iterate(iter::NonlinearCGIteration)
    f, L, Lt, state = _start(iter)
    _descend!(state, f, L, Lt, _first_alpha(state), iter.eta, iter.ls_maxit, iter.refresh)
    _ncg_direction!(state, true)
    return state, (f, L, Lt, state)
end

function Base.iterate(iter::NonlinearCGIteration, (f, L, Lt, state))
    # the next search starts from the step the last one accepted
    alpha0 = state.alpha
    _descend!(state, f, L, Lt, alpha0, iter.eta, iter.ls_maxit, iter.refresh)
    _ncg_direction!(state, false)
    return state, (f, L, Lt, state)
end

function _ncg_direction!(state, first::Bool)
    gg = real(dot(state.grad_prev, state.grad_prev))
    β = first || iszero(gg) ? zero(gg) :
        max(zero(gg), (real(dot(state.grad, state.grad)) - real(dot(state.grad, state.grad_prev))) / gg)
    state.d .= β .* state.d .- state.grad
    real(dot(state.grad, state.d)) >= 0 && (state.d .= .-state.grad)
    return state
end

function Base.iterate(iter::LimitedMemoryBFGSIteration)
    f, L, Lt, state = _start(iter)
    _descend!(state, f, L, Lt, _first_alpha(state), iter.eta, iter.ls_maxit, iter.refresh)
    _lbfgs_direction!(state)
    return state, (f, L, Lt, state)
end

function Base.iterate(iter::LimitedMemoryBFGSIteration, (f, L, Lt, state))
    _descend!(state, f, L, Lt, one(state.alpha), iter.eta, iter.ls_maxit, iter.refresh)
    _lbfgs_direction!(state)
    return state, (f, L, Lt, state)
end

function _lbfgs_direction!(state)
    # correction pair: s = α d (the step just taken), y = change of the gradient
    state.temp .= state.alpha .* state.d
    state.grad_prev .= state.grad .- state.grad_prev
    update!(state.hessian, state.temp, state.grad_prev)
    mul!(state.d, state.hessian, state.grad)
    state.d .*= -1
    if real(dot(state.grad, state.d)) >= 0
        reset!(state.hessian)
        state.d .= .-state.grad
    end
    return state
end

default_stopping_criterion(tol, ::SmoothDescentIteration, state::SmoothDescentState) = state.step <= tol
default_solution(::SmoothDescentIteration, state::SmoothDescentState) = state.x
default_iteration_summary(it, ::SmoothDescentIteration, state::SmoothDescentState) =
    ("" => it, "f(x)" => state.f_x, "‖∇f(x)‖" => norm(state.grad, Inf), "α" => state.alpha)

"""
    NonlinearCG(; <keyword-arguments>)

Constructs the nonlinear conjugate gradient method for the smooth problem

    minimize Σᵢ fᵢ(Lᵢ x).

The returned object has type `IterativeAlgorithm{NonlinearCGIteration}`, and can be called with
the problem's arguments to trigger its solution.

See also: [`NonlinearCGIteration`](@ref), [`LimitedMemoryBFGS`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=1_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion, on the size `‖α d‖∞` of the last step
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `NonlinearCGIteration` constructor upon call
"""
NonlinearCG(;
    maxit = 1_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(NonlinearCGIteration; maxit, stop, solution, verbose, freq, summary, display, kwargs...)

"""
    LimitedMemoryBFGS(; <keyword-arguments>)

Constructs the limited-memory BFGS method for the smooth problem

    minimize Σᵢ fᵢ(Lᵢ x).

The returned object has type `IterativeAlgorithm{LimitedMemoryBFGSIteration}`, and can be called
with the problem's arguments to trigger its solution.

See also: [`LimitedMemoryBFGSIteration`](@ref), [`NonlinearCG`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=1_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion, on the size `‖α d‖∞` of the last step
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `LimitedMemoryBFGSIteration` constructor upon call
"""
LimitedMemoryBFGS(;
    maxit = 1_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(LimitedMemoryBFGSIteration; maxit, stop, solution, verbose, freq, summary, display, kwargs...)
