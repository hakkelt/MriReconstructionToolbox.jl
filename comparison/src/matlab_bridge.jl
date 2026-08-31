module MATLABBridge

using MATLAB

"""
    setup_matlab_paths()

Add the original implementations to the MATLAB path so they can be called.
Requires `module load matlab` to be active in the shell before running Julia.
"""
function setup_matlab_paths()
    base_path = joinpath(@__DIR__, "..", "original_implementations")
    
    mat"addpath(genpath($(joinpath(base_path, \"espirit-matlab-examples\"))))"
    mat"addpath(genpath($(joinpath(base_path, \"RING\"))))"
    mat"addpath(genpath($(joinpath(base_path, \"primal-dual-toolbox\"))))"
    
    # Exclude .git directories from MATLAB path just in case
end

export setup_matlab_paths

end
