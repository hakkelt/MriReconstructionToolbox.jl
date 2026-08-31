module SigPyBridge

using PyCall

const sigpy = PyNULL()
const np = PyNULL()

const sigpy_mri_app = PyNULL()

function __init__()
    # Use the requested virtual environment for SigPy
    ENV["PYTHON"] = "/scratch/c_mrrecon/venvs/sigpy/bin/python"
    
    copy!(sigpy, pyimport("sigpy"))
    copy!(np, pyimport("numpy"))
    copy!(sigpy_mri_app, pyimport("sigpy.mri.app"))
end

export sigpy, np, sigpy_mri_app

end
