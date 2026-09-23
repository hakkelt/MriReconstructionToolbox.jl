# Environment knobs shared by the MRT harness (`benchmark/run.jl`) and the comparison suite.
#
# Every machine-specific path comes from the environment, which `load_site_env!` fills from the
# untracked `benchmark/slurm/site.env` for variables that are not already set. SLURM jobs get the
# same file through `benchmark/slurm/common.sh`, so a login-node run and a cluster run see the same
# configuration without either naming a path in a tracked file.

"""
    SITE_ENV_PATH

The untracked site file (`benchmark/slurm/site.env`, copied from `site.env.example`).
"""
const SITE_ENV_PATH = normpath(joinpath(@__DIR__, "..", "slurm", "site.env"))

"""
    load_site_env!(path = SITE_ENV_PATH) -> Vector{String}

Read `KEY=value` lines from the site file into `ENV`, skipping comments, blank lines and keys that
are already set (the caller's environment wins). Returns the keys it set. A missing file is not an
error: every consumer has its own fallback or its own error message.
"""
function load_site_env!(path::AbstractString = SITE_ENV_PATH)
    set = String[]
    isfile(path) || return set
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        m = match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", s)
        m === nothing && continue
        key, val = m.captures[1], strip(m.captures[2])
        if length(val) >= 2 && val[1] == val[end] && val[1] in ('"', '\'')
            val = val[2:(end - 1)]
        end
        haskey(ENV, key) && continue
        ENV[key] = val
        push!(set, key)
    end
    return set
end

"""
    env_flag(name, default = false) -> Bool

`true` for `1`/`true`/`yes`/`on` (any case), `false` for `0`/`false`/`no`/`off`, `default` when unset.
"""
function env_flag(name::AbstractString, default::Bool = false)
    v = lowercase(strip(get(ENV, name, "")))
    isempty(v) && return default
    v in ("1", "true", "yes", "on") && return true
    v in ("0", "false", "no", "off") && return false
    error("$name=$(ENV[name]) is not a boolean")
end

"""
    ensure_download_path!()

Point MRITestData at `MRT_BENCH_DATA_DIR` when it is set, else at MRITestData's own Scratch cache
when nothing has chosen a location yet. The package refuses to touch the disk until a location is
configured, and the location must not be a tracked preference (it is a path of one machine).
"""
function ensure_download_path!()
    MRITestData = Base.require(Base.PkgId(Base.UUID("b3f1a2c4-5d6e-4a7b-9c8d-0e1f2a3b4c5d"), "MRITestData"))
    dir = get(ENV, "MRT_BENCH_DATA_DIR", "")
    if !isempty(dir)
        current = Base.invokelatest(MRITestData.get_download_path)
        (current === nothing || normpath(string(current)) != normpath(abspath(dir))) &&
            Base.invokelatest(MRITestData.set_download_path!, dir)
    elseif Base.invokelatest(MRITestData.get_download_path) === nothing
        Base.invokelatest(MRITestData.set_download_path!, :cache)
    end
    return nothing
end
