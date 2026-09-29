module BARTBridge

using BartIO

"""
    run_bart(num_outputs::Int, cmd::String, inputs...)

Wrapper around `BartIO.bart` that standardises the BART environment. `TOOLBOX_PATH` must name
the BART build; `_setup.jl` sets it from `MRT_BENCH_BART_MKL` / `MRT_BENCH_BART_OPENBLAS`, and
sets `BART_USE_FFTW_WISDOM=0`, so no call reuses FFT plans another call measured.
"""
function run_bart(num_outputs::Int, cmd::String, inputs...)
    haskey(ENV, "TOOLBOX_PATH") || error(
        "TOOLBOX_PATH is not set: configure MRT_BENCH_BART_MKL / MRT_BENCH_BART_OPENBLAS in " *
            "benchmark/slurm/site.env"
    )
    return bart(num_outputs, cmd, inputs...)
end

export run_bart

end
