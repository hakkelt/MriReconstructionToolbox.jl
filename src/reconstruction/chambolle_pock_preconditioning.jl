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
compensation when the acquisition carries none, `AbstractOperators.estimate_opnorm` of
``W^{1/2}𝒜``) about 0.4 s on that case; 50 iterations reached in 1.2 s what 200 Vũ-Condat
iterations (1.1 s) did not.

`steps` is filled in by [`build_model_with_variables`](@ref), which is where the regularization
operators first exist.
"""
mutable struct _ChambollePockPreconditioner{W}
    weights::W
    steps::Union{Nothing, NamedTuple}
end

# `w`, laid out like the trajectory's sample and frame axes, in `y`'s storage and reshaped to
# broadcast against `y`: the sample axes lead `y`, the frame axes end it, and the axes between them
# (coils, slabs) have size one.
function _broadcast_weights(y, w)
    nframe = _frame_dims_count(size(w), size(y))
    nsample = ndims(w) - nframe
    shift = ndims(y) - ndims(w)
    shape = ntuple(d -> d <= nsample ? size(w, d) : d > ndims(y) - nframe ? size(w, d - shift) : 1, ndims(y))
    return _to_storage_of(unname(y), reshape(w, shape))
end

# `w` expanded to the size of `y`, for the operators that need one weight per entry.
function _storage_like(y, w)
    wf = similar(y, real(eltype(y)))
    wf .= _broadcast_weights(y, w)
    return wf
end

# The density weights of `acq`, normalised to a largest weight of 1: `nothing` on a Cartesian grid,
# where every sample has the same weight. A weight of zero (a ramp's k-space centre) would take its
# sample out of the data term, so every weight is at least the smallest positive one.
_density_weights(::CartesianAcquisitionInfo) = nothing
function _density_weights(acq::NonCartesianAcquisitionInfo)
    dcf = isnothing(acq.dcf) ? density_compensation(acq).dcf : acq.dcf
    w = Array(unname(dcf))
    w = max.(w, minimum(filter(>(0), w)))
    return w ./ maximum(w)
end

# `_density_weights` for the slices of one acquisition, estimated once per trajectory: a shared
# trajectory reaches every slice as the same array, so the slices after the first reuse its
# weights. A slice that finds another computing them waits for that result.
function _density_weights_per_trajectory()
    entries = IdDict{Any, Tuple{ReentrantLock, Base.RefValue{Any}}}()
    entries_lock = ReentrantLock()
    return function (acq)
        acq isa NonCartesianAcquisitionInfo || return _density_weights(acq)
        entry_lock, weights = lock(() -> get!(() -> (ReentrantLock(), Ref{Any}()), entries, acq.trajectory), entries_lock)
        return lock(entry_lock) do
            isassigned(weights) || (weights[] = _density_weights(acq))
            weights[]
        end
    end
end

"""
    _chambolle_pock_preconditioner(method, acq) -> Union{Nothing, _ChambollePockPreconditioner}

The preconditioner for `method`'s reconstruction of `acq`, or `nothing` when `method` does not run
`ChambollePock` with a least-squares data term, or sets any of its step sizes itself.
`density_weights(acq)` gives its density weights.
"""
function _chambolle_pock_preconditioner(method::IterativeReconstruction, acq; density_weights = _density_weights)
    alg = method.algorithm
    alg isa ProximalAlgorithms.IterativeAlgorithm{<:ProximalAlgorithms.ChambollePockIteration} || return nothing
    method.fidelity isa L2Loss || return nothing
    any(k -> haskey(alg.kwargs, k), (:tau, :sigma, :ratio, :normL)) && return nothing
    return _ChambollePockPreconditioner(density_weights(acq), nothing)
end
_chambolle_pock_preconditioner(method, acq; density_weights = _density_weights) = nothing

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
    nD2 = with_serial_blas() do
        sum(t -> Float64(AbstractOperators.estimate_opnorm(StructuredOptimization.operator(t)))^2, terms; init = 0.0)
    end
    if iszero(nD2)
        p.steps = nothing
        return @term ls(𝒜 * x - y)
    end
    # `‖W^{1/2}𝒜‖` from above, since the step budget `τσ‖K‖² < 1` must hold. On a Cartesian grid
    # the value returned is the closed-form bound of the SENSE operator, and the margin only sets how
    # many power steps certify it: that bound sits 1.1% above `‖𝒜‖` on the Cartesian cine case, so
    # the default 1% ran all 100 steps (0.55 s, 6% of a 200-iteration solve) to return the same
    # value; a loose bound costs this solver rate, not safety. Otherwise twenty power steps give the
    # residual bound, which errs upwards; iterating to the default margin took up to five times as
    # many NFFT pairs for a value 1% lower.
    nA2 = with_serial_blas() do
        isnothing(p.weights) && return Float64(AbstractOperators.estimate_opnorm(𝒜; rel_margin = 0.1))^2
        W½𝒜 = DiagOp(codomain_type(𝒜), size(y), _storage_like(y, sqrt.(p.weights))) * 𝒜
        return Float64(AbstractOperators.estimate_opnorm(W½𝒜; maxit = 20))^2
    end
    c = sqrt(nD2 / nA2)
    a = CHAMBOLLE_POCK_DATA_STEP
    p.steps = (; tau = R(0.99 / (2 * a * nA2)), sigma = R(a * nA2 / nD2), normL = R(sqrt(2 * nD2)))
    sw, λ = if isnothing(p.weights)
        R(c), R(1 / c^2)
    else
        _storage_like(y, c .* sqrt.(p.weights)), _broadcast_weights(y, R.(1 / c^2 ./ p.weights))
    end
    return StructuredOptimization.Term(1, SqrNormL2(λ), sw .* (𝒜 * x) - sw .* y, "ls(𝒜x - y)")
end
