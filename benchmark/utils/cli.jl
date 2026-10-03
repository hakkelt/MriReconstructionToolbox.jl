# Command-line parsing shared by the benchmark entry points (`run.jl`, `compare.jl`,
# `large_datasets/run.jl`). Included at top level, before anything else is loaded, since `run.jl`
# re-executes itself in another environment from these flags.

"""
    _arg(name, default = nothing; args = ARGS) -> String
    _list(name, default = nothing; args = ARGS) -> Vector{String}
    _flag(name; args = ARGS) -> Bool

The value of the last `--name=value` in `args` (`default` when absent), that value split at
commas, and whether the bare flag `--name` is present.
"""
_arg(name, default = nothing; args = ARGS) =
    (i = findlast(a -> startswith(a, "--$name="), args)) === nothing ? default : args[i][(length(name) + 4):end]
_list(name, default = nothing; args = ARGS) = (v = _arg(name; args); v === nothing ? default : String.(split(v, ",")))
_flag(name; args = ARGS) = "--$name" in args
