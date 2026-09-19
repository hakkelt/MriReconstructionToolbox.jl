#!/usr/bin/env julia

# Tooling for the vendored dependency stack declared in `deps/vendor.toml`.
#
#     julia tools/vendor.jl check [pkg...]        compare the manifest with GitHub and git
#     julia tools/vendor.jl rebuild [pkg...]      rebuild each fork's `integration` branch
#     julia tools/vendor.jl sync [pkg...]         project `integration` into `deps/`
#     julia tools/vendor.jl status [--json]       the state of the whole stack
#
# With no package names, every package in the manifest is processed. Only the standard library is
# used, so the script runs with a bare `julia` and no project environment. `check` and `status`
# read GitHub through the `gh` CLI; if `gh` is missing or unauthenticated, they degrade to what
# git alone can tell and say so.

module VendorTool

using Dates
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const MANIFEST = joinpath(ROOT, "deps", "vendor.toml")

struct Branch
    name::String
    parent::String
    pr::String
end

struct Package
    name::String
    fork::String
    upstream::String
    base::String
    prefix::String
    path::String
    stack::Vector{Branch}
end

# ---------------------------------------------------------------------------------------------
# manifest

function load_manifest()
    raw = TOML.parsefile(MANIFEST)
    packages = Package[]
    for (name, entry) in sort(collect(raw); by = first)
        stack = [
            Branch(b["branch"], get(b, "parent", ""), get(b, "pr", "")) for
            b in get(entry, "stack", Dict{String,Any}[])
        ]
        push!(
            packages,
            Package(
                name,
                entry["fork"],
                entry["upstream"],
                entry["base"],
                entry["prefix"],
                entry["path"],
                stack,
            ),
        )
    end
    return packages
end

select(packages, names) =
    isempty(names) ? packages : filter(p -> p.name in names, packages)

"""
Validate that the stack is a forest whose parents are declared before their children, so that
merging the branches in file order always merges a parent before the branch stacked on it.
"""
function validate(pkg::Package)
    problems = String[]
    seen = Set{String}()
    for b in pkg.stack
        b.name in seen && push!(problems, "$(pkg.name): branch `$(b.name)` declared twice")
        if !isempty(b.parent) && !(b.parent in seen)
            push!(
                problems,
                "$(pkg.name): `$(b.name)` is stacked on `$(b.parent)`, which is not declared " *
                "before it",
            )
        end
        push!(seen, b.name)
    end
    return problems
end

# ---------------------------------------------------------------------------------------------
# shelling out

struct CommandFailure <: Exception
    cmd::Cmd
    output::String
end

Base.showerror(io::IO, e::CommandFailure) = print(io, "failed: ", e.cmd, "\n", e.output)

"""
Run `cmd`, returning its trimmed output. Returns `nothing` instead of throwing when
`nothing_on_error` is set, which is how the optional `gh` queries stay optional.
"""
function capture(cmd::Cmd; nothing_on_error::Bool = false)
    out = IOBuffer()
    try
        run(pipeline(cmd; stdout = out, stderr = out))
    catch
        nothing_on_error && return nothing
        throw(CommandFailure(cmd, String(take!(out))))
    end
    return strip(String(take!(out)))
end

git(pkg::Package, args...) = capture(`git -C $(pkg.path) $(collect(args))`)
git_or_nothing(pkg::Package, args...) =
    capture(`git -C $(pkg.path) $(collect(args))`; nothing_on_error = true)
here(args...) = capture(`git -C $ROOT $(collect(args))`)

const GH_AVAILABLE = Ref{Union{Nothing,Bool}}(nothing)

function gh_available()
    if GH_AVAILABLE[] === nothing
        GH_AVAILABLE[] =
            Sys.which("gh") !== nothing &&
            capture(`gh auth status`; nothing_on_error = true) !== nothing
    end
    return GH_AVAILABLE[]::Bool
end

"""
State of one pull request as GitHub sees it: `(state, base, url)`, or `nothing` when the PR is
unknown or `gh` is unavailable. `ref` is the `owner/repo#number` form used in the manifest.
"""
function pr_state(ref::AbstractString)
    gh_available() || return nothing
    (isempty(ref) || !occursin('#', ref)) && return nothing
    repo, number = split(ref, '#')
    out = capture(
        `gh pr view $number --repo $repo --json state,baseRefName,url,isDraft`;
        nothing_on_error = true,
    )
    out === nothing && return nothing
    # `gh` returns JSON; pick out the few fields we need without a JSON dependency.
    state = match(r"\"state\":\"([^\"]*)\"", out)
    base = match(r"\"baseRefName\":\"([^\"]*)\"", out)
    url = match(r"\"url\":\"([^\"]*)\"", out)
    draft = occursin("\"isDraft\":true", out)
    (state === nothing || base === nothing) && return nothing
    return (
        state = state.captures[1],
        base = base.captures[1],
        url = url === nothing ? "" : url.captures[1],
        draft = draft,
    )
end

# ---------------------------------------------------------------------------------------------
# check

"""
Report every way the stack disagrees with itself: a branch whose PR is based on something other
than its manifest parent, a branch with no PR at all, a branch whose PR has already merged (its
code should come from `base` instead), and a branch that does not exist locally.
"""
function check(packages)
    findings = 0
    for pkg in packages
        println("== ", pkg.name)
        for problem in validate(pkg)
            println("  manifest: ", problem)
            findings += 1
        end
        for b in pkg.stack
            if git_or_nothing(pkg, "rev-parse", "--verify", "--quiet", "origin/$(b.name)") ===
               nothing
                println("  $(b.name): no `origin/$(b.name)` in $(pkg.path)")
                findings += 1
            end
            if isempty(b.pr)
                println("  $(b.name): vendored but no PR anywhere -- untracked work")
                findings += 1
                continue
            end
            info = pr_state(b.pr)
            info === nothing && continue
            if info.state == "MERGED"
                println(
                    "  $(b.name): $(b.pr) is merged -- drop the entry and take the code from " *
                    "$(pkg.base)",
                )
                findings += 1
            elseif info.state == "CLOSED"
                println("  $(b.name): $(b.pr) is closed, but the branch is still vendored")
                findings += 1
            end
            # The PR base is the claim GitHub makes about the stack; the manifest parent is the
            # claim this repository makes. A mismatch means the PR will conflict when its
            # intended parent lands.
            intended = isempty(b.parent) ? last(split(pkg.base, '/')) : b.parent
            if info.state == "OPEN" && info.base != intended
                println(
                    "  $(b.name): $(b.pr) is based on `$(info.base)`, the stack wants " *
                    "`$(intended)`",
                )
                findings += 1
            end
        end
    end
    gh_available() ||
        println("\nnote: `gh` unavailable, so PR state and base were not checked")
    println("\n", findings == 0 ? "stack is consistent" : "$findings finding(s)")
    return findings
end

# ---------------------------------------------------------------------------------------------
# rebuild

"""
Rebuild `integration` for each package: reset it to `base`, then merge every manifest branch in
file order. Because a parent is always declared before the branches stacked on it, the merge
order is the order the PRs are meant to land in, and the result is one ref meaning "everything I
have" for that package.
"""
function rebuild(packages; push::Bool = true)
    for pkg in packages
        problems = validate(pkg)
        isempty(problems) || error(join(problems, "\n"))
        println("== ", pkg.name, " (", pkg.path, ")")
        dirty = git(pkg, "status", "--porcelain")
        isempty(dirty) || error("$(pkg.path) has uncommitted changes; refusing to rebuild")
        git(pkg, "fetch", "--prune", "origin")
        git(pkg, "fetch", "--prune", "upstream")
        git(pkg, "checkout", "-B", "integration", pkg.base)
        for b in pkg.stack
            println("  merge ", b.name)
            git(pkg, "merge", "--no-ff", "-m", "integration: $(b.name)", "origin/$(b.name)")
        end
        if push
            git(pkg, "push", "--force-with-lease", "origin", "integration")
        end
        println("  ", git(pkg, "rev-parse", "--short", "HEAD"))
    end
end

# ---------------------------------------------------------------------------------------------
# sync

"""
Project each package's `integration` branch into `deps/` with `git subtree pull --squash`, so the
vendored copy carries the revision it came from in its commit message and drift becomes
computable instead of guessed at. MRT-local adaptations that no upstream would take -- relative
imports, an `ext/` wired in by hand, because an included submodule loads neither -- live in
`deps/patches/<package>.patch` and are re-applied afterwards. A patch that stops applying has
been upstreamed and can be deleted.
"""
function sync(packages)
    for pkg in packages
        println("== ", pkg.name)
        # `deps/` is generated output: a hand edit here would be silently squashed away.
        dirty = here("status", "--porcelain", "--", pkg.prefix)
        isempty(dirty) || error(
            "uncommitted changes under $(pkg.prefix); commit or discard them -- `deps/` is " *
            "generated, fix bugs on the owning branch instead",
        )
        here(
            "subtree",
            "pull",
            "--prefix",
            pkg.prefix,
            pkg.fork,
            "integration",
            "--squash",
            "-m",
            "chore($(pkg.name)): re-vendor integration",
        )
        patch = joinpath(ROOT, "deps", "patches", "$(pkg.name).patch")
        if isfile(patch)
            println("  apply ", relpath(patch, ROOT))
            here("apply", "--3way", patch)
        end
    end
end

# ---------------------------------------------------------------------------------------------
# status

json_escape(s) = replace(string(s), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n")
json(s::AbstractString) = "\"$(json_escape(s))\""
json(b::Bool) = b ? "true" : "false"
json(n::Integer) = string(n)
json(v::Vector) = "[" * join(json.(v), ",") * "]"
json(d::Vector{<:Pair}) = "{" * join(["$(json(k)):$(json(v))" for (k, v) in d], ",") * "}"

"""
The state of the whole stack, as the text summary a terminal wants or as the JSON the dashboard
reads. Regenerate the dashboard's data with

    julia tools/vendor.jl status --json > docs/src/assets/vendor-status.json
"""
function status(packages; as_json::Bool = false)
    entries = Vector{Pair{String,Any}}[]
    for pkg in packages
        for b in pkg.stack
            info = pr_state(b.pr)
            intended = isempty(b.parent) ? last(split(pkg.base, '/')) : b.parent
            state = if isempty(b.pr)
                "untracked"
            elseif info === nothing
                "unknown"
            elseif info.state == "MERGED"
                "merged"
            elseif info.state == "CLOSED"
                "closed"
            elseif info.base != intended
                "mis-based"
            elseif occursin("hakkelt/", b.pr)
                "fork"
            else
                "upstream"
            end
            push!(
                entries,
                [
                    "package" => pkg.name,
                    "branch" => b.name,
                    "parent" => b.parent,
                    "pr" => b.pr,
                    "url" => info === nothing ? "" : info.url,
                    "base_on_github" => info === nothing ? "" : info.base,
                    "intended_base" => String(intended),
                    "draft" => info === nothing ? false : info.draft,
                    "state" => state,
                ],
            )
        end
    end
    if as_json
        head = [
            "generated" => string(today()),
            "mrt_commit" => here("rev-parse", "--short", "HEAD"),
            "gh" => gh_available(),
        ]
        println("{", join(["$(json(k)):$(json(v))" for (k, v) in head], ","), ",\"branches\":[")
        println(join(["  " * json(e) for e in entries], ",\n"))
        println("]}")
    else
        width = maximum(length(e[2][2]) for e in entries; init = 0)
        current = ""
        for e in entries
            pkg, branch, state = e[1][2], e[2][2], e[9][2]
            pkg == current || (println("== ", pkg); current = pkg)
            println("  ", rpad(branch, width), "  ", rpad(state, 10), "  ", e[4][2])
        end
        counts = Dict{String,Int}()
        for e in entries
            counts[e[9][2]] = get(counts, e[9][2], 0) + 1
        end
        println("\n", join(["$v $k" for (k, v) in sort(collect(counts))], " · "))
    end
end

# ---------------------------------------------------------------------------------------------

const USAGE = """
usage: julia tools/vendor.jl <command> [package...] [options]

  check                 compare the manifest with GitHub and the local checkouts
  rebuild [--no-push]   rebuild each fork's `integration` branch from the manifest
  sync                  project `integration` into deps/ via `git subtree pull --squash`
  status [--json]       the state of the whole stack
"""

function main(args)
    isempty(args) && (print(USAGE); return 1)
    command = first(args)
    rest = collect(args[2:end])
    flags = filter(startswith("-"), rest)
    names = filter(!startswith("-"), rest)
    packages = select(load_manifest(), names)
    isempty(packages) && error("no package in $MANIFEST matches $(join(names, ", "))")
    if command == "check"
        return check(packages) == 0 ? 0 : 1
    elseif command == "rebuild"
        rebuild(packages; push = !("--no-push" in flags))
    elseif command == "sync"
        sync(packages)
    elseif command == "status"
        status(packages; as_json = "--json" in flags)
    else
        print(USAGE)
        return 1
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(VendorTool.main(ARGS))
end
