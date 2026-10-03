"""
    CHAMBOLLE_POCK_DATA_STEP

The dual step of the data block of a `ChambollePock` reconstruction, in units of the inverse
curvature of the data term, per unit density-compensation weight (see
[`_ChambollePockPreconditioner`](@ref)).
"""
const CHAMBOLLE_POCK_DATA_STEP = 0.3

"""
    _ChambollePockPreconditioner

The block-diagonal preconditioning (Pock & Chambolle, ICCV 2011) a `ChambollePock`
reconstruction runs with when its step sizes are left to `reconstruct`.

Chambolle-Pock stacks every term into ``h(Kx)`` with ``K = [𝒜; D₁; …]``. One scalar dual step for
all blocks lets the largest-norm block, ``𝒜``, dictate it: the data dual then barely moves, and a
non-Cartesian ``𝒜'𝒜``, whose spectrum spans the sampling density, converges slowest of all. Two
changes fix that without touching the objective:

- the data term is written as ``½‖W^{-1/2}(c W^{1/2}𝒜x - c W^{1/2}y)/c‖²`` — the same function of
  ``x`` — with ``W`` the density-compensation weights (normalised to a largest weight of 1; all ones
  on a Cartesian grid) and ``c = ‖[D₁; …]‖ / ‖W^{1/2}𝒜‖``, so the data block of ``K`` has the norm of
  the regularization blocks and its dual step, in the units of the original data term, is
  ``σ c² W`` per sample;
- the steps are ``σ = a‖W^{1/2}𝒜‖² / ‖D‖²`` and ``τ = 0.99 / (2a‖W^{1/2}𝒜‖²)``, so the data block
  takes the per-sample dual step ``a W`` and the step budget ``τσ‖K‖² < 1`` is split evenly between
  the data and the regularization blocks.

`a` is [`CHAMBOLLE_POCK_DATA_STEP`](@ref), the best of 0.01, 0.03, 0.1, 0.3 and 1 on both 2D 8-coil cases
(anisotropic TV, default λ; NRMSE after 25 / 50 / 200 iterations):

| | radial | Cartesian |
|---|---|---|
| Vũ-Condat | 0.32 / 0.23 / 0.070 | 0.40 / 0.36 / 0.150 |
| Chambolle-Pock, preconditioned | 0.028 / 0.0165 / 0.0163 | 0.083 / 0.072 / 0.067 |

On radial data an iteration costs about three Vũ-Condat iterations — the data block applies the
NFFT and its adjoint where Vũ-Condat applies the Toeplitz normal operator — and the setup (density
compensation when the acquisition carries none, the power iteration for ``‖W^{1/2}𝒜‖``) about 0.4 s
on that case; 50 iterations reached in 1.2 s what 200 Vũ-Condat iterations (1.1 s) did not.

`steps` is filled in by [`build_model_with_variables`](@ref), which is where the regularization
operators first exist.
"""
mutable struct _ChambollePockPreconditioner{W}
    weights::W
    data_step::Float64
    steps::Union{Nothing, NamedTuple}
end

# `w` broadcast over the trailing (coil) dimensions of `y`, in `y`'s storage.
function _storage_like(y, w)
    R = real(eltype(y))
    wd = copyto!(similar(unname(y), R, size(w)), w)
    wf = similar(y, R)
    wf .= wd
    return wf
end

# The density weights of `acq`, normalised to a largest weight of 1: `nothing` on a Cartesian grid,
# where every sample has the same weight.
_density_weights(::CartesianAcquisitionInfo) = nothing
function _density_weights(acq::NonCartesianAcquisitionInfo)
    dcf = isnothing(acq.dcf) ? density_compensation(acq).dcf : acq.dcf
    w = Array(unname(dcf))
    return w ./ maximum(w)
end

"""
    _chambolle_pock_preconditioner(method, acq) -> Union{Nothing, _ChambollePockPreconditioner}

The preconditioner for `method`'s reconstruction of `acq`, or `nothing` when `method` does not run
`ChambollePock` with a least-squares data term, or sets any of its step sizes itself.
"""
function _chambolle_pock_preconditioner(method::IterativeReconstruction, acq)
    alg = method.algorithm
    alg isa ProximalAlgorithms.IterativeAlgorithm{<:ProximalAlgorithms.ChambollePockIteration} || return nothing
    method.fidelity isa L2Loss || return nothing
    any(k -> haskey(alg.kwargs, k), (:tau, :sigma, :ratio, :normL)) && return nothing
    return _ChambollePockPreconditioner(_density_weights(acq), CHAMBOLLE_POCK_DATA_STEP, nothing)
end
_chambolle_pock_preconditioner(method, acq) = nothing

# ‖W^{1/2}𝒜‖² by power iteration from `x`.
function _weighted_opnorm2(𝒜, w, x; maxit = 20)
    v = copy(x)
    v ./= norm(v)
    s = zero(real(eltype(v)))
    for _ in 1:maxit
        r = 𝒜 * v
        isnothing(w) || (r .*= w)
        v = 𝒜' * r
        s = norm(v)
        iszero(s) && break
        v ./= s
    end
    return s
end

_term_list(t::StructuredOptimization.Term) = (t,)
_term_list(ts::StructuredOptimization.TermSet) = Tuple(ts)

"""
    _preconditioned_data_term(𝒜, y, x, reg_terms, p) -> Term

The least-squares data term in the form described in [`_ChambollePockPreconditioner`](@ref), with
`p.steps` set to the step sizes that go with it; `ls(𝒜x - y)` with `p.steps = nothing` when there is
no regularization block to balance against.
"""
function _preconditioned_data_term(𝒜, y, x, reg_terms, p::_ChambollePockPreconditioner)
    R = real(eltype(y))
    terms = Iterators.flatten(map(_term_list, reg_terms))
    nD2 = sum(t -> Float64(AbstractOperators.estimate_opnorm(StructuredOptimization.operator(t)))^2, terms; init = 0.0)
    if iszero(nD2)
        p.steps = nothing
        return @term ls(𝒜 * x - y)
    end
    w = isnothing(p.weights) ? nothing : _storage_like(y, p.weights)
    nA2 = Float64(_weighted_opnorm2(𝒜, w, ~x))
    c = sqrt(nD2 / nA2)
    a = p.data_step
    p.steps = (; tau = R(0.99 / (2 * a * nA2)), sigma = R(a * nA2 / nD2), normL = R(sqrt(2 * nD2)))
    if isnothing(w)
        return StructuredOptimization.Term(1, SqrNormL2(R(1 / c^2)), R(c) * (𝒜 * x) - R(c) .* y, "ls(𝒜x - y)")
    end
    sw = R(c) .* sqrt.(w)
    return StructuredOptimization.Term(1, SqrNormL2(R(1 / c^2) ./ w), sw .* (𝒜 * x) - sw .* y, "ls(𝒜x - y)")
end
