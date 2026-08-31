module BARTBridge

using BartIO

"""
    run_bart(num_outputs::Int, cmd::String, inputs...)

Wrapper around `BartIO.bart` to standardize BART environment configurations.
Sets `BART_USE_FFTW_WISDOM=1` as requested.
"""
function run_bart(num_outputs::Int, cmd::String, inputs...)
    # Set the environment variable for FFTW wisdom
    ENV["BART_USE_FFTW_WISDOM"] = "1"
    # Ensure BART path is known if necessary
    # On this machine, BART is at /project/c_mrrecon/bart_mkl
    if !haskey(ENV, "TOOLBOX_PATH")
        ENV["TOOLBOX_PATH"] = "/project/c_mrrecon/bart_mkl"
    end
    
    return bart(num_outputs, cmd, inputs...)
end

export run_bart

end
