module RistrettoMKLExt

using Ristretto
using MKL

function __init__()
    # The one thing this extension can usefully do.
    #
    # KMP_BLOCKTIME cannot be fixed from Julia at all: libiomp5 reads it once, when it is
    # loaded, which has already happened by the time any Julia code runs. Assigning to ENV here
    # does nothing, and `ccall((:kmp_set_blocktime, "libiomp5.so"), Cvoid, (Cint,), 0)` returns
    # cleanly while also doing nothing. So the only available remedy is to tell the user.
    #
    # `mkl_set_dynamic(0)` is not called: it makes no measurable difference to level-3 or level-1
    # throughput. See docs/src/high-level/performance.md.
    if get(ENV, "KMP_BLOCKTIME", "") != "0"
        @warn """
        MKL is loaded but KMP_BLOCKTIME is $(get(ENV, "KMP_BLOCKTIME", "unset")), not 0.

        MKL's worker threads will spin at full CPU for 200 ms after each parallel region and
        crowd out the Julia threads running the FFT and prox steps of a reconstruction.

        This cannot be set from Julia — MKL reads it once, at load. Export it before starting
        Julia:

            export KMP_BLOCKTIME=0

        See the Performance & Threading page of the documentation.""" maxlog = 1
    end
    return nothing
end

end # module RistrettoMKLExt
