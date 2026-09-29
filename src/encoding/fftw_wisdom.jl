# FFTW wisdom cached on disk, per machine.
#
# A `MEASURE` or `PATIENT` plan is found by timing candidate algorithms on the machine it runs
# on, which costs 0.1–1 s (`MEASURE`) or 10–200 s (`PATIENT`) per transform and is paid again in
# every new process. FFTW records what it learned as "wisdom", which any later plan of the same
# problem reuses, including an `ESTIMATE` one: a 256×256×8 `ESTIMATE` plan that runs in 9 ms per
# transform runs in 0.8 ms once the wisdom of a measured one is loaded. The wisdom is specific to
# the CPU, the FFTW build, the transform (sizes, batch, strides, in-place or not) and the number
# of FFTW threads, so the cache file is named after the CPU model and FFTW version, and a
# session only benefits from entries made with its own thread count.

const _WISDOM_LOCK = ReentrantLock()
# The file the process's wisdom was last loaded from, `""` before any load.
const _WISDOM_LOADED_FROM = Ref("")
# Set once a plan that measures has been made, so there is something worth saving.
const _WISDOM_DIRTY = Ref(false)
# Forces the planner rigor of every plan MRT makes, for `plan_fft_wisdom`.
const _FFTW_RIGOR = Base.ScopedValues.ScopedValue{Union{Nothing, UInt32}}(nothing)

"""
    _fftw_flags(fast_planning::Bool) -> UInt32

The FFTW planner flags for a plan MRT is about to make: `ESTIMATE` under `fast_planning`, else
`MEASURE`, unless [`plan_fft_wisdom`](@ref) forces a rigor. Loads the on-disk wisdom first, so an
`ESTIMATE` plan of a problem measured in an earlier session gets the measured plan.
"""
function _fftw_flags(fast_planning::Bool)
    _load_fftw_wisdom()
    flags = something(_FFTW_RIGOR[], fast_planning ? FFTW.ESTIMATE : FFTW.MEASURE)
    flags == FFTW.ESTIMATE || (_WISDOM_DIRTY[] = true)
    return flags
end

"""
    fftw_wisdom_path() -> Union{String, Nothing}

The file MRT keeps FFTW wisdom in, or `nothing` when the cache is off. It lives in the package's
scratch space (`~/.julia/scratchspaces/<uuid>/fftw_wisdom/`), else under `\$XDG_CACHE_HOME` (or
`~/.cache`) when that is not writable, and its name carries the CPU model and FFTW version, so
machines sharing a home directory keep separate files. The environment variable
`MRT_FFTW_WISDOM` overrides it: `off` disables the cache, any other value names the directory.
Only the `fftw` provider of FFTW.jl has wisdom; under another provider the cache is off.
"""
function fftw_wisdom_path()
    setting = get(ENV, "MRT_FFTW_WISDOM", "")
    lowercase(setting) in ("off", "0", "false", "no") && return nothing
    FFTW.get_provider() == "fftw" || return nothing
    dir = isempty(setting) ? _default_wisdom_dir() : setting
    isnothing(dir) && return nothing
    return joinpath(dir, "wisdom-" * _machine_key() * ".fftw")
end

# Resolved once per process: `missing` until then.
const _DEFAULT_WISDOM_DIR = Ref{Union{Missing, Nothing, String}}(missing)

function _default_wisdom_dir()::Union{Nothing, String}
    cached = _DEFAULT_WISDOM_DIR[]
    ismissing(cached) || return cached
    dir = _find_wisdom_dir()
    _DEFAULT_WISDOM_DIR[] = dir
    return dir
end

function _find_wisdom_dir()
    try
        return Scratch.get_scratch!(@__MODULE__, "fftw_wisdom")
    catch
    end
    base = get(ENV, "XDG_CACHE_HOME", "")
    isempty(base) && (base = joinpath(homedir(), ".cache"))
    dir = joinpath(base, "MriReconstructionToolbox", "fftw_wisdom")
    try
        mkpath(dir)
        return dir
    catch
        return nothing
    end
end

const _MACHINE_KEY = Ref("")

function _machine_key()
    isempty(_MACHINE_KEY[]) || return _MACHINE_KEY[]
    cpus = Sys.cpu_info()
    model = isempty(cpus) ? "unknown" : strip(cpus[1].model)
    _MACHINE_KEY[] = string(hash((model, string(Sys.ARCH), string(FFTW.version))); base = 16)
    return _MACHINE_KEY[]
end

function _load_fftw_wisdom()
    path = fftw_wisdom_path()
    something(path, "") == _WISDOM_LOADED_FROM[] && return nothing
    lock(_WISDOM_LOCK) do
        target = something(path, "")
        target == _WISDOM_LOADED_FROM[] && return nothing
        if !isnothing(path) && isfile(path)
            try
                FFTW.import_wisdom(path)
            catch err
                @debug "FFTW wisdom not loaded" path exception = err
            end
        end
        _WISDOM_LOADED_FROM[] = target
    end
    return nothing
end

"""
    _save_fftw_wisdom()

Write the process's FFTW wisdom to [`fftw_wisdom_path`](@ref), after merging in whatever another
process saved there meanwhile. The file is replaced by a rename, so a concurrent reader never sees
it half written. A no-op until a plan that measures has been made, and whenever the cache is off
or its directory is not writable.
"""
function _save_fftw_wisdom()
    _WISDOM_DIRTY[] || return nothing
    path = fftw_wisdom_path()
    isnothing(path) && return nothing
    lock(_WISDOM_LOCK) do
        try
            isfile(path) ? FFTW.import_wisdom(path) : mkpath(dirname(path))
            tmp = string(path, ".", getpid(), ".tmp")
            FFTW.export_wisdom(tmp)
            mv(tmp, path; force = true)
        catch err
            @debug "FFTW wisdom not saved" path exception = err
        end
    end
    return nothing
end

const _FFTW_RIGORS = (estimate = FFTW.ESTIMATE, measure = FFTW.MEASURE, patient = FFTW.PATIENT, exhaustive = FFTW.EXHAUSTIVE)

"""
    plan_fft_wisdom(acq::AcquisitionInfo; rigor = :patient, threaded = true) -> Union{String, Nothing}

Plan every FFT that reconstructing `acq` makes — the encoding operator's forward and adjoint
transforms and its normal operator's — with planner rigor `rigor` (`:measure`, `:patient` or
`:exhaustive`), and save what FFTW learned to the on-disk wisdom cache
([`fftw_wisdom_path`](@ref), returned). Later sessions on this machine then get these plans for
the same problem, even where MRT itself plans with `ESTIMATE`.

Wisdom holds only for the exact transform and FFTW thread count it was made with, so run this with
the thread count reconstructions will use (`julia -t N`), on an acquisition shaped like theirs (the
same image size, coil count and frame count; the data itself does not matter), and on the machine
that will run them. `:patient` takes 10–35 s per 2D transform and minutes per 3D one; `:measure`
is what MRT does on its own for a long solve.
"""
function plan_fft_wisdom(acq::AcquisitionInfo; rigor::Symbol = :patient, threaded::Bool = true)
    @argcheck haskey(_FFTW_RIGORS, rigor) "rigor must be one of $(keys(_FFTW_RIGORS))"
    Base.ScopedValues.with(_FFTW_RIGOR => _FFTW_RIGORS[rigor]) do
        𝒜 = model_encoding_operator(nothing, acq; threaded, fast_planning = false)
        AbstractOperators.get_normal_op(𝒜)
    end
    _WISDOM_DIRTY[] = true
    _save_fftw_wisdom()
    return fftw_wisdom_path()
end
