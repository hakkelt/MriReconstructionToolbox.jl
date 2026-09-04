module BARTBridge

using BartIO

"""
    run_bart(num_outputs::Int, cmd::String, inputs...; wisdom::Bool = false)

Wrapper around `BartIO.bart` that standardises the BART environment.

`BART_USE_FFTW_WISDOM` is left to the value set by the caller (`_setup.jl` sets it to `"0"` by
default, because `"1"` forces `FFTW_MEASURE` on every fresh `bart` process and the wisdom file is
never persisted, a measured ~6x penalty on short recons). Pass `wisdom = true` only for recons
whose own compute is long enough (measured > 5 s) that the one-off `FFTW_MEASURE` planning pays
for itself; `run_bart` then flips the variable on for that call and restores it afterwards.
`TOOLBOX_PATH` is only set if the caller has not already pinned it to a specific BART build.
"""
function run_bart(num_outputs::Int, cmd::String, inputs...; wisdom::Bool = false)
    if !haskey(ENV, "TOOLBOX_PATH")
        ENV["TOOLBOX_PATH"] = "/project/c_mrrecon/bart_mkl"
    end
    if wisdom
        saved = get(ENV, "BART_USE_FFTW_WISDOM", "0")
        ENV["BART_USE_FFTW_WISDOM"] = "1"
        try
            return bart(num_outputs, cmd, inputs...)
        finally
            ENV["BART_USE_FFTW_WISDOM"] = saved
        end
    end
    return bart(num_outputs, cmd, inputs...)
end

export run_bart

end
