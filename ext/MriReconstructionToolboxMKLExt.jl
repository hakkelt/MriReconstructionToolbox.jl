module MriReconstructionToolboxMKLExt

using MriReconstructionToolbox
using MKL

function __init__()
    # The one thing this extension can usefully do.
    #
    # KMP_BLOCKTIME cannot be fixed from Julia at all: libiomp5 reads it once, when it is
    # loaded, which has already happened by the time any Julia code runs. Assigning to ENV here
    # does nothing, and `ccall((:kmp_set_blocktime, "libiomp5.so"), Cvoid, (Cint,), 0)` returns
    # cleanly while also doing nothing (measured: 16.2 s, against 3.8 s when the variable is
    # exported before launch). So the only available remedy is to tell the user.
    #
    # `mkl_set_dynamic(0)` used to be called here as well. It was dropped: the call works (the
    # flag does flip) but made no measurable difference to either level-3 or level-1 throughput
    # — 3000³ gemm 0.242 s vs 0.236 s, 2000x 16 MiB axpy 0.506 s vs 0.554 s, both within noise.
    # Its supposed benefit was preserving ThreadPinning placement, which was never demonstrated.
    # See docs/src/high-level/performance.md.
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

end # module MriReconstructionToolboxMKLExt
