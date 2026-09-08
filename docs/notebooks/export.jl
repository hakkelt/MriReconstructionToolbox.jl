#=
export.jl — execute and export docs/notebooks/*.ipynb to HTML.

Usage (from the repository root, or from docs/notebooks/):

    julia --project=docs/notebooks docs/notebooks/export.jl [selector] [--timeout=SECONDS]

`selector` is optional:
  - omitted, or "all"      -> export every notebook in docs/notebooks/
  - a number, e.g. "01"/"1" -> export the one notebook whose filename starts with that number
  - a name fragment, e.g. "regularization" -> export notebooks whose filename contains it

`--timeout=SECONDS` (default 600) is the PER-NOTEBOOK wall-clock budget (passed to nbconvert's
`ExecutePreprocessor.timeout`, which is a per-CELL timeout — export.jl also enforces it as a
per-notebook kill switch, so one stuck cell cannot hang the whole run past the budget).

Output: rendered HTML files under docs/notebooks/build/ (gitignored — never committed). Each
notebook is executed against a FRESH in-memory copy by nbconvert; the .ipynb source files on
disk are never touched, so they stay output-free (verified again at the end as a safety check).

Requires `python3 -m nbconvert` on PATH with the `julia-1.12` Jupyter kernel installed
(`Pkg.build("IJulia")` from the docs/notebooks environment registers it — see README.md).
=#

const NOTEBOOK_DIR = @__DIR__
const BUILD_DIR = joinpath(NOTEBOOK_DIR, "build")
const DEFAULT_TIMEOUT = 600 # seconds, per notebook

function parse_args(args)
    selector = "all"
    timeout = DEFAULT_TIMEOUT
    for a in args
        if startswith(a, "--timeout=")
            timeout = parse(Int, split(a, "=")[2])
        elseif !startswith(a, "--")
            selector = a
        end
    end
    return selector, timeout
end

function matching_notebooks(selector::AbstractString)
    all_nb = sort(filter(f -> endswith(f, ".ipynb"), readdir(NOTEBOOK_DIR)))
    selector == "all" && return all_nb
    # numeric selector: "01", "1", "9" -> match the leading NN- prefix
    if occursin(r"^\d+$", selector)
        n = lpad(selector, 2, '0')
        return filter(f -> startswith(f, n * "_"), all_nb)
    end
    return filter(f -> occursin(selector, f), all_nb)
end

function export_one(nb_file::AbstractString, timeout::Int)
    src = joinpath(NOTEBOOK_DIR, nb_file)
    mkpath(BUILD_DIR)
    cmd = `python3 -m nbconvert --to html --execute
        --ExecutePreprocessor.kernel_name=julia-1.12
        --ExecutePreprocessor.timeout=$timeout
        --output-dir=$BUILD_DIR $src`
    t0 = time()
    ok = false
    msg = ""
    proc = run(pipeline(cmd; stdout = devnull, stderr = devnull); wait = false)
    watchdog = Timer(timeout + 30) do _
        process_running(proc) && kill(proc)
    end
    try
        wait(proc)
        ok = success(proc)
        msg = ok ? "" : "nbconvert exited with code $(proc.exitcode)"
    catch e
        msg = "exception: $e"
    finally
        close(watchdog)
    end
    elapsed = time() - t0
    return (; nb_file, ok, msg, elapsed)
end

function verify_no_outputs(nb_file::AbstractString)
    # Cheap textual check: an executed-in-place notebook would have non-null "execution_count"
    # or non-empty "outputs" arrays. We only ever pass notebooks to nbconvert by path (never
    # --to notebook --execute in place), so this should always pass; kept as a safety net.
    content = read(joinpath(NOTEBOOK_DIR, nb_file), String)
    return !occursin(r"\"execution_count\":\s*\d", content)
end

function main()
    selector, timeout = parse_args(ARGS)
    notebooks = matching_notebooks(selector)
    if isempty(notebooks)
        println(stderr, "No notebook matches selector \"$selector\".")
        exit(1)
    end

    println("Exporting ", length(notebooks), " notebook(s) (timeout = $(timeout)s each): ", join(notebooks, ", "))
    results = Vector{NamedTuple}(undef, 0)
    for nb_file in notebooks
        print(rpad("  " * nb_file, 45))
        flush(stdout)
        r = export_one(nb_file, timeout)
        push!(results, r)
        status = r.ok ? "OK" : "FAIL"
        println(status, "  (", round(r.elapsed, digits = 1), "s)", r.ok ? "" : "  — " * r.msg)
    end

    println()
    println("Verifying source notebooks are still output-free...")
    dirty = String[]
    for nb_file in notebooks
        verify_no_outputs(nb_file) || push!(dirty, nb_file)
    end
    if !isempty(dirty)
        println(stderr, "WARNING: these notebooks now carry outputs on disk: ", join(dirty, ", "))
    else
        println("  all clean.")
    end

    println()
    npass = count(r -> r.ok, results)
    nfail = length(results) - npass
    println("Summary: $npass passed, $nfail failed.")
    if nfail > 0
        println("Failed: ", join((r.nb_file for r in results if !r.ok), ", "))
        exit(1)
    end
    return
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
