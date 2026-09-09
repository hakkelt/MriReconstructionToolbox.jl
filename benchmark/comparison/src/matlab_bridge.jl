module MATLABBridge

using MATLAB

"""
    setup_matlab_paths(; require = ("espirit-matlab-examples", "RING", "primal-dual-toolbox"))

Add the reference implementations under `benchmark/comparison/original_implementations` to the
MATLAB path so they can be called. Errors if a required one is missing, naming the directory it
looked in. Requires `module load matlab` to be active in the shell before running Julia.

Not every reference is vendored: `LORAKS` (LORAKS 2.0, <https://mr.usc.edu/download/loraks2/>)
has to be downloaded and unpacked into that directory by hand before it can be required.
"""
function setup_matlab_paths(; require = ("espirit-matlab-examples", "RING", "primal-dual-toolbox"))
    base_path = joinpath(@__DIR__, "..", "original_implementations")
    for name in require
        dir = joinpath(base_path, name)
        isdir(dir) && !isempty(readdir(dir)) ||
            error("MATLAB reference `$name` is missing from $dir")
        mat"addpath(genpath($dir))"
    end
    return
end

export setup_matlab_paths

end
