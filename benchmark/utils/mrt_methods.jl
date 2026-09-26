# How MRT reconstructs each catalog method. Shared by the harness and by the comparison suite's MRT
# rows, so the two time exactly the same call.

"""
    OUTER_ITERATIONS, CG_ITERATIONS, ADMM_RHO

The fixed effort every iterative method runs at: `OUTER_ITERATIONS` ADMM (or FISTA) iterations,
`CG_ITERATIONS` inner CG per ADMM iteration, a fixed ADMM penalty `ADMM_RHO`, and no early stop.
CG-SENSE runs `CG_ITERATIONS` iterations. Overridable with `CMP_OUTER` / `CMP_CG_ITERS`.

`ADMM_RHO` is relative to `‖𝒜‖²` on a checkout whose `reconstruct` scales a given ADMM penalty
by it (`_scale_admm_penalty`), and absolute on one that does not. The difference is large only
for radial cases (`‖𝒜‖² ≈ 2·10⁶`): there an absolute `0.05` never lets the regularizer act, and
their NRMSE from such a checkout is that of an unregularized solve, whatever `λ` says.
"""
const OUTER_ITERATIONS = parse(Int, get(ENV, "CMP_OUTER", "20"))
const CG_ITERATIONS = parse(Int, get(ENV, "CMP_CG_ITERS", "10"))
const ADMM_RHO = 5.0e-2

"""
    RADIAL_ADMM_RHO, RADIAL_LAMBDA

The ADMM penalty and λ per method for radial cases, where the Cartesian values do not carry over.
Calibrated on `shepp_logan_2d_8ch_radial` (tv, tgv, wavelet) and `torso_cine_8ch_radial` (lowrank,
llr, ttv) by NRMSE after `OUTER_ITERATIONS` iterations, over a λ grid `10^(-4:0.5:0.5)` and ρ from
0.002 to 20 relative to `‖𝒜‖²`.

NRMSE falls steadily as ρ decreases down to about 0.005. Below that the cine methods flatten
(ttv is best at 0.01, within 5% at 0.002) while the 2D ones keep improving, so `0.002` is within 5%
of each method's best and at least matches MRT's default adaptive penalty on every method. At 0.002 the NRMSE is tv 0.035, tgv 0.034, llr 0.087, lowrank 0.107 and ttv 0.067.
L1-wavelet runs FISTA, so only its λ was calibrated (NRMSE 0.279).
"""
const RADIAL_ADMM_RHO = 2.0e-3
const RADIAL_LAMBDA = Dict(
    :tv => 1.0e-3, :atv => 1.0e-3, :wavelet => 3.0e-3, :tgv => 1.0e-3, :lowrank => 3.0e-2, :llr => 3.0e-3,
    :ttv => 1.0e-3,
)

"""
    DEFAULT_LAMBDA

λ per method for Cartesian cases when no calibrated value is asked for; radial cases use
`RADIAL_LAMBDA`. The harness always uses these (through [`default_lambda`](@ref)), so a timing
and its NRMSE are comparable across checkouts regardless of later recalibration.
"""
const DEFAULT_LAMBDA = Dict(
    :tv => 0.01, :atv => 0.01, :wavelet => 0.005, :tgv => 0.003, :lowrank => 0.01, :llr => 0.01,
    :ttv => 0.01,
)

"""
    default_lambda(c::BenchCase, method) -> λ

`RADIAL_LAMBDA` for a radial case and `DEFAULT_LAMBDA` otherwise; `0.0` for an unregularized
method.
"""
default_lambda(c::BenchCase, method::Symbol) =
    get(c.trajectory === :noncartesian ? RADIAL_LAMBDA : DEFAULT_LAMBDA, penalty_of(method), 0.0)

"""
    admm_rho(c::BenchCase) -> ρ

`RADIAL_ADMM_RHO` for a radial case and `ADMM_RHO` otherwise.
"""
admm_rho(c::BenchCase) = c.trajectory === :noncartesian ? RADIAL_ADMM_RHO : ADMM_RHO

"""
    WAVELET_LEVELS

Decomposition depth of the L1-wavelet rows (`db2`), matched across toolkits.
"""
const WAVELET_LEVELS = parse(Int, get(ENV, "CMP_WAVELET_LEVELS", "3"))

"""
    mrt_regularizer(c::BenchCase, method, λ)

`:tv` is the isotropic `λ Σ ‖∇x‖₂` (per voxel), `:atv` the anisotropic `λ Σᵢ ‖Δⁱx‖₁`. They are
different problems with different optima: on `shepp_logan_3d_8ch_cartesian` the isotropic one
reaches NRMSE 0.008 where the anisotropic one reaches 0.0033.
"""
function mrt_regularizer(c::BenchCase, method::Symbol, λ::Real)
    method = penalty_of(method)
    vol = c.family === :volume
    method === :tv && return vol ? TotalVariation3D(λ) : TotalVariation2D(λ)
    method === :atv && return vol ? AnisotropicTotalVariation3D(λ) : AnisotropicTotalVariation2D(λ)
    method === :wavelet && return vol ?
        L1Wavelet3D(λ; wavelet = MriReconstructionToolbox.WT.db2, levels = WAVELET_LEVELS) :
        L1Wavelet2D(λ; wavelet = MriReconstructionToolbox.WT.db2, levels = WAVELET_LEVELS)
    method === :tgv && return TotalGeneralizedVariation2D(λ; ratio = 2.0)
    method === :lowrank && return LowRank(λ; time_dim = :time)
    method === :llr && return LocallyLowRank(λ; block_size = (8, 8), time_dim = :time)
    method === :ttv && return TemporalTotalVariation(λ; time_dim = :time)
    throw(ArgumentError("$method has no regularizer"))
end

"""
    PDHG_ITERATIONS

The iteration count of a PDHG row: `OUTER_ITERATIONS × CG_ITERATIONS`, the normal-operator
applications of the matching ADMM row. A PDHG iteration applies the operator about once where an
ADMM iteration applies it once per inner CG step, so equal iteration counts would not be equal
effort.
"""
const PDHG_ITERATIONS = OUTER_ITERATIONS * CG_ITERATIONS

"""
    default_maxit(method) -> Int

`CG_ITERATIONS` for CG-SENSE, `PDHG_ITERATIONS` for a PDHG row, `OUTER_ITERATIONS` otherwise.
"""
default_maxit(m::Symbol) =
    m === :cgsense ? CG_ITERATIONS : haskey(PDHG_METHODS, m) ? PDHG_ITERATIONS : OUTER_ITERATIONS

"""
    mrt_algorithm(method, maxit; rho = ADMM_RHO)

Fixed-ρ ADMM with a fixed inner CG and no early stop for every regularized method except
L1-wavelet, which runs FISTA (forcing a fixed-ρ ADMM on it wrecks it); CGNR for CG-SENSE;
`ChambollePock` for a PDHG row, with its default step sizes. The data term is its smooth part, so
it runs the Condat-Vũ form of the iteration.
"""
function mrt_algorithm(method::Symbol, maxit::Int; rho::Real = ADMM_RHO)
    method === :cgsense && return MriReconstructionToolbox.CGNR(; maxit, tol = 0.0)
    method === :wavelet && return MriReconstructionToolbox.FISTA(; maxit, tol = 0.0)
    haskey(PDHG_METHODS, method) && return MriReconstructionToolbox.ChambollePock(; maxit, tol = 0.0)
    return MriReconstructionToolbox.ADMM(; rho, maxit, tol = 0.0, cg_tol = 0.0, cg_maxit = CG_ITERATIONS)
end

"""
    mrt_reconstructor(c, method; λ = default_lambda(c, method), rho = admm_rho(c), maxit, acq = nothing) -> () -> image

A zero-argument closure running MRT's reconstruction of `c` by `method`, for `time_run`, with the
ADMM penalty `rho` (relative to `‖𝒜‖²`; ignored by the methods that do not run ADMM). The
acquisition is built once, outside the closure; `acq` passes one in (the comparison suite reuses
it across rows). `maxit` defaults to [`default_maxit`](@ref).
`maxit` and `reltol = 0` are set on `IterativeReconstruction` as well as on the algorithm: the
method's values win over the algorithm's, so both must agree to run the full count.
"""
function mrt_reconstructor(
        c::BenchCase, method::Symbol;
        λ::Real = default_lambda(c, method),
        rho::Real = admm_rho(c),
        maxit::Int = default_maxit(method),
        acq = nothing,
    )
    if method === :adjoint || method === :gridding
        a = something(acq, mrt_acquisition(c; dcf = method === :gridding))
        return () -> reconstruct(a, DirectReconstruction(); verbosity = Silent())
    end
    a = something(acq, mrt_acquisition(c))
    reg = method === :cgsense ? () : mrt_regularizer(c, method, λ)
    m = IterativeReconstruction(; regularization = reg, algorithm = mrt_algorithm(method, maxit; rho), maxit, reltol = 0.0)
    return () -> reconstruct(a, m; verbosity = Silent())
end
