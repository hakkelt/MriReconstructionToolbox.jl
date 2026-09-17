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
notebook is executed against a FRESH in-memory copy by nbconvert; the .ipynb files this script
regenerates from docs/notebooks/src/*.jl (see below) are never touched, so they stay
output-free (verified again at the end as a safety check).

`docs/notebooks/*.ipynb` is not tracked in git — `docs/notebooks/src/*.jl` (jupytext percent
format) is the committed source. This script regenerates the .ipynb files from it via
`python3 -m jupytext --to ipynb` before exporting, so a clean checkout is enough to run it.

Requires `python3 -m nbconvert` and `python3 -m jupytext` on PATH, with the `julia-1.13` Jupyter
kernel installed (`Pkg.build("IJulia")` from the docs/notebooks environment registers it — see
README.md).
=#

using Downloads: Downloads

const NOTEBOOK_DIR = @__DIR__
const SRC_DIR = joinpath(NOTEBOOK_DIR, "src")
const BUILD_DIR = joinpath(NOTEBOOK_DIR, "build")
const DEFAULT_TIMEOUT = 600 # seconds, per notebook

function regenerate_ipynb!()
    scripts = sort(filter(f -> endswith(f, ".jl"), readdir(SRC_DIR)))
    isempty(scripts) && error("no notebook scripts found under $SRC_DIR")
    for script in scripts
        out = joinpath(NOTEBOOK_DIR, replace(script, r"\.jl$" => ".ipynb"))
        run(`python3 -m jupytext --to ipynb --output $out $(joinpath(SRC_DIR, script))`)
    end
    return nothing
end

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
        --ExecutePreprocessor.kernel_name=julia-1.13
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
    ok && postprocess_html!(joinpath(BUILD_DIR, replace(nb_file, r"\.ipynb$" => ".html")))
    return (; nb_file, ok, msg, elapsed)
end

# Marks a page as already patched, so a re-run of the exporter does not stack the block twice.
const POSTPROCESS_MARKER = "MRT-postprocess-marker"

# Long source lines must stay reachable on a narrow screen. The element that actually clips them
# is `div.CodeMirror`, which the template gives `overflow: hidden` (the `.jp-InputArea` rules are
# clipping too, but they are only the outer box); the same goes for wide text output and wide
# tables.
const OVERFLOW_CSS = """
<style type="text/css">
/* $POSTPROCESS_MARKER: appended by docs/notebooks/export.jl -- see postprocess_html!. */
.jp-InputArea, .jp-InputArea-editor { overflow-x: auto !important; }
.jp-CodeMirrorEditor .CodeMirror, div.CodeMirror { overflow-x: auto !important; overflow-y: visible !important; }
.jp-InputArea-editor .highlight, .jp-InputArea-editor .highlight pre { overflow-x: auto; }
.jp-OutputArea-output { overflow-x: auto; }
.jp-RenderedHTMLCommon table { display: block; width: fit-content; max-width: 100%; overflow-x: auto; }
</style>"""

# MathJax 3 with SVG output, embedded in the page rather than loaded from a CDN. Two separate
# things have to hold for a sandboxed viewer to render an equation at all:
#
#  - no runtime fetches. The combined MathJax 2 build the template pulls in needs its config file
#    and web fonts (`@font-face`) at run time, both of which a content-security policy blocks. The
#    SVG build draws every glyph from paths inside a single script file: no stylesheet, no font
#    files, no XHR.
#  - no cross-origin script either. A viewer that blocks the CDN request leaves the page with raw
#    LaTeX and no visible error, which is exactly what a `<script src=...>` loader looks like when
#    it fails. Since the SVG build *is* one file, the whole renderer can be inlined, and then
#    rendering depends on nothing outside the page.
#
# The file is cached under `cache/` (gitignored) so only the first export downloads it; a compute
# node with no outbound network reuses that copy.
const MATHJAX_VERSION = "3.2.2"
const MATHJAX_URL = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/$MATHJAX_VERSION/es5/tex-mml-svg.js"
const MATHJAX_CACHE = joinpath(NOTEBOOK_DIR, "cache", "mathjax-$MATHJAX_VERSION-tex-mml-svg.js")

const MATHJAX_CONFIG = """
<script type="text/javascript">
window.MathJax = {
  tex: {
    inlineMath: [['\$', '\$'], ['\\\\(', '\\\\)']],
    displayMath: [['\$\$', '\$\$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams'
  },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] },
  svg: { fontCache: 'local' }
};
</script>"""

# Matches whichever loader a page already carries, so an older export can be re-patched in place
# (see `postprocess_html!`) without executing the notebook again.
const MATHJAX_LOADER_RE = r"<script id=\"MathJax-script\".*?</script>"s

"""
    mathjax_bundle() -> String or nothing

The MathJax SVG build as a string, downloading it into [`MATHJAX_CACHE`](@ref) once. Returns
`nothing` when it cannot be embedded, in which case the page falls back to the CDN loader.
"""
function mathjax_bundle()
    if !isfile(MATHJAX_CACHE)
        mkpath(dirname(MATHJAX_CACHE))
        try
            Downloads.download(MATHJAX_URL, MATHJAX_CACHE)
        catch err
            @warn "could not fetch the MathJax bundle; falling back to the CDN loader, which a " *
                "sandboxed viewer may block" exception = err
            return nothing
        end
    end
    js = read(MATHJAX_CACHE, String)
    # An inlined script ends at the first `</script` in its text, and a `<script` can flip the
    # parser into a state where that is no longer true. The 3.2.2 bundle contains neither; refuse
    # to inline anything that does rather than emit a page that stops parsing halfway.
    if occursin("</script", js) || occursin("<script", js)
        @warn "the MathJax bundle contains a script tag and cannot be inlined; using the CDN loader"
        return nothing
    end
    return js
end

"""
    mathjax_loader() -> String

The `<script>` element that brings in the renderer: the bundle itself when it can be embedded,
the CDN loader otherwise. Built once and reused — the bundle is ~2 MB, and every page gets it.
"""
function mathjax_loader()
    if !isassigned(MATHJAX_LOADER)
        js = mathjax_bundle()
        MATHJAX_LOADER[] = if js === nothing
            "<script id=\"MathJax-script\" async src=\"$MATHJAX_URL\"></script>"
        else
            "<script id=\"MathJax-script\" type=\"text/javascript\">\n$js\n</script>"
        end
    end
    return MATHJAX_LOADER[]
end

const MATHJAX_LOADER = Ref{String}()

"""
    mathjax_html() -> String

The configuration block plus the renderer itself.
"""
mathjax_html() = MATHJAX_CONFIG * "\n" * mathjax_loader()

"""
    postprocess_html!(html_file)

Patch the two defects nbconvert's bundled HTML template bakes into every exported page.

1. Its MathJax 2 setup never renders in a sandboxed viewer: the URL it hardcodes 404s
   (`mathjax/<ver>/latest.js` — cdnjs never served an unversioned `latest.js` alias), and even
   with that corrected the combined build needs runtime fetches a content-security policy blocks.
   The whole block is replaced by [`mathjax_html`](@ref), a MathJax 3 SVG build embedded in the
   page.
2. `overflow: hidden` on the code editor clips long source lines instead of letting the reader
   scroll to them. Overridden by [`OVERFLOW_CSS`](@ref).

Both are appended to the end of the document head, so they win on source order over the
template's own rules. Idempotent, and re-patchable: a page already carrying
[`POSTPROCESS_MARKER`](@ref) keeps its CSS but has its MathJax loader replaced, so a page exported
before the renderer was embedded can be upgraded without executing the notebook again.
"""
function postprocess_html!(html_file::AbstractString)
    isfile(html_file) || return nothing
    content = read(html_file, String)
    # Substituted through a function, never a replacement string: the bundle is full of `\\1`-like
    # sequences that `replace` would otherwise read as capture-group references.
    fixed = if occursin(POSTPROCESS_MARKER, content)
        replace(content, MATHJAX_LOADER_RE => _ -> mathjax_loader(); count = 1)
    else
        # The template writes the loader and its `text/x-mathjax-config` block between these
        # comments.
        stripped = replace(content, r"<!-- Load mathjax -->.*?<!-- End of mathjax configuration -->"s => "")
        replace(
            stripped, "</head>" => _ -> OVERFLOW_CSS * "\n" * mathjax_html() * "\n</head>";
            count = 1
        )
    end
    fixed == content || write(html_file, fixed)
    return nothing
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
    # Skip when a caller (e.g. a SLURM array with one task per notebook) already regenerated
    # every .ipynb once up front -- concurrent tasks calling regenerate_ipynb! at the same time
    # would race on the same output files.
    get(ENV, "MRT_SKIP_REGEN", "") == "1" || regenerate_ipynb!()
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

if abspath(PROGRAM_FILE) == @__FILE__()
    main()
end
