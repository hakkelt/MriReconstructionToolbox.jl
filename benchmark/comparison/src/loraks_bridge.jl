"""
    LORAKSBridge

Call the authors' MATLAB LORAKS package as a numerical **oracle** for MRT's calibrationless
`StructuredLowRank`, over `.mat` files and a `matlab -batch` subprocess.

# Why a subprocess and not `MATLAB.jl`

`MATLAB.jl` cannot be precompiled where no MATLAB installation is visible, so listing it in
`benchmark/comparison/Project.toml` takes `Pkg.instantiate` — and with it the whole comparison
suite — down on every machine without MATLAB. That is why `matlab_bridge.jl` is `include`d
inside a `try` and its one call site stayed commented out. Shelling out has neither problem:
`MAT.jl` is pure Julia, the dependency exists everywhere, and a missing MATLAB is an ordinary
`false` from [`loraks_available`](@ref) rather than a load error. It is also how BART is
already called from here.

# Licensing

The LORAKS package is **not** vendored, and must not be. Its licence permits educational,
research and non-profit use; MriReconstructionToolbox is MIT, which grants commercial use, so
the two cannot ship together. `benchmark/comparison/original_implementations/LORAKS2/` is
gitignored for that reason. Download it yourself from <http://mr.usc.edu/download/LORAKS2/>
(a form issues a personalised link) and unpack it there.

Its licence also requires that work using it cite the technical report and the formulation's
original paper; [`loraks_citation`](@ref) carries the three references, and
`compare_structured_low_rank.jl` prints them whenever the oracle actually runs.

# Oracle, not a source

Nothing here is derived from LORAKS source. This bridge writes inputs, runs the package
unmodified, and reads the result back — so MRT's own implementation can be checked against
its numbers without taking on its licence. Fixes to MRT must come from the papers, with the
oracle used only to say whether MRT agrees.
"""
module LORAKSBridge

using MAT
using LinearAlgebra

export loraks_available, loraks_recon, loraks_citation, LORAKS_DIR

"Where the unpacked LORAKS 2.1 package is expected (gitignored; see the module docstring)."
const LORAKS_DIR = abspath(joinpath(@__DIR__, "..", "original_implementations", "LORAKS2"))

"""
    loraks_citation() -> String

The citations the LORAKS licence requires from anything that uses it.
"""
loraks_citation() = """
LORAKS reference implementation:
  [1] T. H. Kim, J. P. Haldar. LORAKS Software Version 2.0: Faster Implementation and
      Enhanced Capabilities. University of Southern California, Los Angeles, CA,
      Technical Report USC-SIPI-443, May 2018.
  [2] J. P. Haldar. Low-Rank Modeling of Local k-Space Neighborhoods (LORAKS) for
      Constrained MRI. IEEE Trans. Med. Imaging 33:668-681, 2014.
  [3] J. P. Haldar, J. Zhuo. P-LORAKS: Low-Rank Modeling of Local k-Space Neighborhoods
      with Parallel Imaging Data. Magn. Reson. Med. 75:1499-1514, 2016.
"""

"""
    matlab_executable() -> Union{String, Nothing}

The `matlab` binary to drive, or `nothing` when there is none. `MRT_BENCH_MATLAB` names it
outright; otherwise `matlab` has to be on `PATH` already.

**Prefer the environment variable to `module load matlab`.** On this cluster the module prepends
MATLAB's own library directory to `LD_LIBRARY_PATH`, and the `libpcre2` it ships is not the one
Julia is built against, so a Julia started from that shell dies on the first `using` that touches
a regular expression:

    ERROR: LoadError: PCRE compilation error: unrecognised compile-time option bit(s) at offset 0

MATLAB's launcher sets up its own environment, so an absolute path needs no module at all:

    MRT_BENCH_MATLAB=/opt/software/packages/matlab/r2024b/bin/matlab julia --project=...

This is the second reason the bridge is a subprocess rather than `MATLAB.jl`: in-process, that
library conflict has no workaround.

Run the oracle on a **compute node**, not a shared login node. MATLAB opens a thread per core and
holds the whole problem in memory; on this cluster's login node it is reaped mid-run
(`ProcessSignaled(9)` after ~80 s, against the ~340 s the reconstruction actually takes).
"""
function matlab_executable()
    explicit = get(ENV, "MRT_BENCH_MATLAB", "")
    isempty(explicit) || return isfile(explicit) ? explicit : nothing
    found = Sys.which("matlab")
    return found === nothing ? nothing : String(found)
end

"""
    loraks_available() -> Bool

Whether both halves of the oracle are present: the unpacked package and a runnable MATLAB.
"""
function loraks_available()
    isdir(LORAKS_DIR) && isfile(joinpath(LORAKS_DIR, "P_LORAKS.m")) || return false
    return matlab_executable() !== nothing
end

"""
    loraks_recon(kspace_zf, mask; rank, radius = 3, ltype = "C", max_iter = 50)
        -> (; kspace, elapsed)

Reconstruct with the authors' `P_LORAKS`. `kspace_zf` is `(N1, N2, Nc)` complex with zeros at
the unsampled positions and `mask` is the `(N1, N2)` sampling pattern. Returns the filled-in
k-space (the same shape) and MATLAB's own wall time in seconds.

`radius` is LORAKS' k-space neighbourhood **radius**, so the neighbourhood is the disc
`k₁² + k₂² ≤ radius²` — not the rectangle MRT's `window` keyword describes. `radius = 3` covers
29 samples, closest to MRT's `window = (6, 6)` at 36; they are comparable, not identical, and a
difference in the last digits should not be read as a defect in either.
"""
function loraks_recon(
        kspace_zf::AbstractArray{<:Complex, 3}, mask::AbstractMatrix;
        rank::Integer, radius::Integer = 3, ltype::AbstractString = "C", max_iter::Integer = 50,
    )
    exe = matlab_executable()
    exe === nothing && error("no MATLAB on PATH; `module load matlab/r2024b` first")
    isdir(LORAKS_DIR) || error("LORAKS is not unpacked in $LORAKS_DIR — see LORAKSBridge's docstring")

    dir = mktempdir()
    try
        infile = joinpath(dir, "in.mat")
        outfile = joinpath(dir, "out.mat")
        matwrite(
            infile, Dict(
                "kData" => ComplexF64.(kspace_zf),
                "kMask" => Float64.(mask),
                "rank" => Float64(rank),
                "radius" => Float64(radius),
                "ltype" => String(ltype),
                "max_iter" => Float64(max_iter),
            )
        )
        driver = @__DIR__
        # `-batch` exits non-zero on an uncaught MATLAB error, so `run` raises on failure.
        run(
            pipeline(
                `$exe -nodisplay -nosplash -batch "addpath('$driver'); loraks_oracle('$infile','$outfile','$LORAKS_DIR')"`,
                stdout = devnull,
            )
        )
        out = matread(outfile)
        return (; kspace = ComplexF64.(out["recon"]), elapsed = Float64(out["elapsed"]))
    finally
        rm(dir; recursive = true, force = true)
    end
end

end # module
