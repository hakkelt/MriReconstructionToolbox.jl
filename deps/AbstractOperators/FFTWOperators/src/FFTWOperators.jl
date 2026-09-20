module FFTWOperators

using AbstractOperators, FFTW, LinearAlgebra
using Base.Cartesian: @ncall
using Polyester: @batch
import LinearAlgebra: mul!
import Base: size, ndims

import AbstractOperators:
    _normalize_array_type,
    _array_wrapper_type,
    domain_type,
    codomain_type,
    fun_name,
    get_normal_op,
    allocate_in_domain,
    allocate_in_codomain,
    domain_array_type,
    codomain_array_type,
    can_be_combined,
    combine,
    is_thread_safe,
    is_AcA_diagonal,
    is_AAc_diagonal,
    diag_AcA,
    diag_AAc,
    is_orthogonal,
    is_invertible,
    is_full_row_rank,
    is_full_column_rank,
    is_symmetric,
    has_fast_opnorm,
    check,
    is_threaded,
    supports_threading,
    _resolve_threaded,
    _elementwise_threaded,
    threading_threshold,
    _copy_operator_impl

"""
	fftw_threading_threshold(kind::Symbol) -> Int

Element count at which threading an FFTW transform of this `kind` starts to pay.

PROVENANCE: measured (AMD EPYC 7352, 8 threads, OPENBLAS_NUM_THREADS=1), sweeping `mul!`
against a 1-thread plan of the same size. The `:c2c` entry is from 2026-08-15; the other two
were re-measured on 2026-09-20 **with `FFTW.ESTIMATE`**, which is what every operator here
plans with by default, and moved an octave down as a result:

| kind | first sustained win | speedup there | at n = 2^22 |
|---|---|---|---|
| `:c2c` (DFT/IDFT) | 2^13 | 1.75x | 6.99x |
| `:r2r` (DCT/IDCT) | 2^14 | 2.00x (1D), 4.10x (2D) | 6.44x |
| `:r2c` (RDFT/IRDFT) | 2^14 | 1.14x (1D), 1.87x (2D) | 6.12x |

The original 2^15 for the two real transforms was measured against `FFTW.MEASURE` plans,
where a tuned serial plan is harder to beat. Under the planner these operators actually use,
2^14 already pays on every shape tried and 2^15 left measured speedup on the table -- 4.10x
on a 128x128 DCT, refused.

Below these sizes threading an FFT is a *large* pessimisation, not a wash -- a 256-point
c2c transform measures 0.02x, and a 1024-element one 0.15x at eight threads -- which is why
the policy applies here rather than trusting FFTW's planner to sort it out.
"""
fftw_threading_threshold(kind::Symbol) = kind === :c2c ? 2^13 : 2^14

"""
	_fftw_num_threads(kind, num_threads, threaded, n) -> Int

Resolve the plan-time FFTW thread count.

`num_threads` is FFTW's own vocabulary and is an explicit **command**: given, it wins
outright, which keeps an escape hatch for callers who know what they want. `threaded` is the
package-wide keyword and follows the package-wide rule -- `false` vetoes, `true`/`nothing`
enable subject to the policy above. See `AbstractOperators._resolve_threaded`.

Past the threshold the whole pool is used. An intermediate count was measured and rejected
(AMD EPYC 7352, 8 threads, `FFTW.ESTIMATE`, `ComplexF64` in-place c2c, min of 300 reps,
2026-09-20): sweeping every thread count over a grid of transform size x batch count, capping
to two threads in the first octave above the threshold and four in the second costs more than
it saves -- 1.47x on a 2^15 DCT and 1.64x on a 256x256 one at 2^16, against at most 1.45x
bought in a single c2c cell (64x64 x4 at 2^14: 25.52 us at four threads, 37.07 at eight).
The full count is the best or within 15 % of the best on every other cell measured, and never
slower than one thread anywhere above the threshold.

What `n` genuinely cannot express is **nesting**: an 8-thread plan applied from an
already-saturated outer loop oversubscribes. Measured at `nthreads()` concurrent tasks each
applying its own plan, an 8-thread plan is 7-32 % slower than a 1-thread one (64^2x8: 213.22
vs 162.50 us; 320^2x8: 11896.72 vs 11116.78), while at half that concurrency it is still
1.4-1.9x faster. The crossover is a property of the caller, not of the transform, so it
belongs to the caller: pass `threaded = false` or an explicit `num_threads` when planning an
operator that will be applied from inside a threaded region.
"""
function _fftw_num_threads(kind::Symbol, num_threads, threaded::Bool, n::Int)
    num_threads !== nothing && return Int(num_threads)
    use = _resolve_threaded(threaded) do
        Threads.nthreads() > 1 && n >= fftw_threading_threshold(kind)
    end
    return use ? Threads.nthreads() : 1
end

"""
	_with_fftw_threads(f, num_threads)

Run the planning callable `f` with FFTW's global thread count temporarily set to
`num_threads`, restoring the previous value afterwards.

FFTW's thread count is process-global state consulted at *plan* time, so it has to be set
around planning and put back; forgetting the restore would silently change the thread count
of every plan built later in the session.
"""
function _with_fftw_threads(f, num_threads::Int)
    prev = FFTW.get_num_threads()
    FFTW.set_num_threads(num_threads)
    try
        return f()
    finally
        FFTW.set_num_threads(prev)
    end
end

include("DFT.jl")
include("RDFT.jl")
include("IRDFT.jl")
include("DCT.jl")
include("Shift.jl")
include("combination_rules.jl")

end # module FFTWOperators
