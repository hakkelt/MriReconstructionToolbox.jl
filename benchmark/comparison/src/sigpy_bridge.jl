module SigPyBridge

using PyCall

const sigpy = PyNULL()
const np = PyNULL()

const sigpy_mri_app = PyNULL()

function __init__()
    # The virtual environment SigPy is installed in -- where it lives on this cluster, unless
    # `MRT_BENCH_SIGPY_PYTHON` names another interpreter.
    ENV["PYTHON"] = get(
        ENV, "MRT_BENCH_SIGPY_PYTHON", "/scratch/c_mrrecon/venvs/sigpy/bin/python",
    )

    copy!(sigpy, pyimport("sigpy"))
    copy!(np, pyimport("numpy"))
    return copy!(sigpy_mri_app, pyimport("sigpy.mri.app"))
end

export sigpy, np, sigpy_mri_app

end
