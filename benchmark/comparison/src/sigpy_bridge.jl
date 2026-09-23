module SigPyBridge

using PyCall

const sigpy = PyNULL()
const np = PyNULL()

const sigpy_mri_app = PyNULL()

function __init__()
    # PyCall's interpreter is fixed when PyCall is built, so setting `ENV["PYTHON"]` here would
    # change nothing. `MRT_BENCH_SIGPY_PYTHON` only documents which interpreter was intended; a
    # mismatch means PyCall must be rebuilt (`ENV["PYTHON"] = ...; Pkg.build("PyCall")`).
    want = get(ENV, "MRT_BENCH_SIGPY_PYTHON", "")
    if !isempty(want) && (!ispath(want) || realpath(want) != realpath(PyCall.python))
        @warn "PyCall uses a different Python than MRT_BENCH_SIGPY_PYTHON; rebuild PyCall to change it" pycall = PyCall.python configured = want
    end

    copy!(sigpy, pyimport("sigpy"))
    copy!(np, pyimport("numpy"))
    return copy!(sigpy_mri_app, pyimport("sigpy.mri.app"))
end

export sigpy, np, sigpy_mri_app

end
