# How MRT reconstructs each catalog method. Shared by the harness and by the comparison suite's MRT
# rows, so the two time exactly the same call.

"""
    OUTER_ITERATIONS, CG_ITERATIONS, ADMM_RHO

The fixed effort every iterative method runs at: `OUTER_ITERATIONS` ADMM (or FISTA) iterations,
`CG_ITERATIONS` inner CG per ADMM iteration, a fixed ADMM penalty `ADMM_RHO`, and no early stop.
CG-SENSE runs `CG_ITERATIONS` iterations. Overridable with `CMP_OUTER` / `CMP_CG_ITERS`.
"""
const OUTER_ITERATIONS = parse(Int, get(ENV, "CMP_OUTER", "20"))
const CG_ITERATIONS = parse(Int, get(ENV, "CMP_CG_ITERS", "10"))
const ADMM_RHO = 5.0e-2

"""
    DEFAULT_LAMBDA

λ per method when no calibrated value is asked for. The harness always uses these, so a timing
and its NRMSE are comparable across checkouts regardless of later recalibration.
"""
const DEFAULT_LAMBDA = Dict(
    :tv => 0.01, :wavelet => 0.005, :tgv => 0.003, :lowrank => 0.01, :llr => 0.01, :ttv => 0.01,
)

"""
    WAVELET_LEVELS

Decomposition depth of the L1-wavelet rows (`db2`), matched across toolkits.
"""
const WAVELET_LEVELS = parse(Int, get(ENV, "CMP_WAVELET_LEVELS", "3"))

"""
    mrt_regularizer(c::BenchCase, method, λ)
"""
function mrt_regularizer(c::BenchCase, method::Symbol, λ::Real)
    vol = c.family === :volume
    method === :tv && return vol ? TotalVariation3D(λ) : TotalVariation2D(λ)
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
    mrt_algorithm(method, maxit; rho = ADMM_RHO)

Fixed-ρ ADMM with a fixed inner CG and no early stop for every regularized method except
L1-wavelet, which runs FISTA (forcing a fixed-ρ ADMM on it wrecks it); CGNR for CG-SENSE.
"""
function mrt_algorithm(method::Symbol, maxit::Int; rho::Real = ADMM_RHO)
    method === :cgsense && return MriReconstructionToolbox.CGNR(; maxit, tol = 0.0)
    method === :wavelet && return MriReconstructionToolbox.FISTA(; maxit, tol = 0.0)
    return MriReconstructionToolbox.ADMM(; rho, maxit, tol = 0.0, cg_tol = 0.0, cg_maxit = CG_ITERATIONS)
end

"""
    mrt_reconstructor(c, method; λ = DEFAULT_LAMBDA[method], maxit, acq = nothing) -> () -> image

A zero-argument closure running MRT's reconstruction of `c` by `method`, for `time_run`. The
acquisition is built once, outside the closure; `acq` passes one in (the comparison suite reuses
it across rows). `maxit` defaults to `CG_ITERATIONS` for CG-SENSE and `OUTER_ITERATIONS` otherwise.
`maxit` and `reltol = 0` are set on `IterativeReconstruction` as well as on the algorithm: the
method's values win over the algorithm's, so both must agree to run the full count.
"""
function mrt_reconstructor(
        c::BenchCase, method::Symbol;
        λ::Real = get(DEFAULT_LAMBDA, method, 0.0),
        maxit::Int = method === :cgsense ? CG_ITERATIONS : OUTER_ITERATIONS,
        acq = nothing,
    )
    if method === :adjoint || method === :gridding
        a = something(acq, mrt_acquisition(c; dcf = method === :gridding))
        return () -> reconstruct(a, DirectReconstruction(); verbosity = Silent())
    end
    a = something(acq, mrt_acquisition(c))
    reg = method === :cgsense ? () : mrt_regularizer(c, method, λ)
    m = IterativeReconstruction(; regularization = reg, algorithm = mrt_algorithm(method, maxit), maxit, reltol = 0.0)
    return () -> reconstruct(a, m; verbosity = Silent())
end
