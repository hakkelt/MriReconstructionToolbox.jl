module ComparisonHarness

include("bart_bridge.jl")
include("sigpy_bridge.jl")
include("matlab_bridge.jl")
include("mirt_bridge.jl")
include("phantoms.jl")

using .BARTBridge
using .SigPyBridge
using .MATLABBridge
using .MIRTBridge
using .Phantoms

export run_bart
export sigpy, np, sigpy_mri_app
export setup_matlab_paths
export MIRT
export generate_multicoil_brain

end
