# Execution policy: where a kernel runs, and with which loop construct.
#
# There are two independent questions, and keeping them apart is the whole design:
#
#   *May* this operator thread?      A permission the caller sets at construction
#                                    (`threaded = false` is a hard veto), carried in a
#                                    type parameter so the branch folds away.
#   *Should* it, for this input?     A policy question that cannot be answered at
#                                    construction, because the size and the storage type
#                                    only arrive with `x`.
#
# `should_thread(f, x)` combines the two; `is_threaded(f, x)` reports the combined answer,
# never the request. The veto direction matters: block-parallel calculus rules
# (`SeparableSum` and friends) rebuild their children with `threaded = false` before
# spawning, and rely on that being obeyed unconditionally.
#
# The numbers below are measured, not guessed -- see `benchmark/threading_sweep.jl`, whose
# RESULTS block records the machine, thread count and date, and the crossover for every
# cost class.

export is_threaded, supports_threading

# --------------------------------------------------------------------------- cost classes
#
# A cost class says how expensive one element is, which decides two things at once: the
# size at which threading starts to pay, and which loop construct wins. Declaring the class
# once per operator keeps those two from drifting apart.

abstract type CostClass end

"""Elementwise `exp`/`log` work. SIMD-vectorisable through SLEEFPirates, so these use
`@turbo`/`@tturbo`; measured 3-7x over a plain `@simd` loop even below the threading
threshold, and 4-10x over `@batch` above it."""
struct Transcendental <: CostClass end

"""Elementwise arithmetic: soft-thresholding, clamping, scaling."""
struct Arithmetic <: CostClass end

"""Dominated by memory traffic: copies, and the fused write-and-reduce shape that most of
`src/functions/` has."""
struct MemoryBound <: CostClass end

"""A loop whose body is a whole child `prox!` call."""
struct BlockParallel <: CostClass end

# PROVENANCE: benchmark/threading_sweep.jl, 2026-09-16, znver2 (taskset 0-15),
# julia 1.13.0, -t 8. Re-run that file before editing these.
const THRESHOLD_ELEMENTWISE_TRANSCENDENTAL = 2^10
const THRESHOLD_ELEMENTWISE_ARITHMETIC     = 2^15
const THRESHOLD_MEMORY_BOUND               = 2^18
const THRESHOLD_BLOCK_PARALLEL             = 2^16

# Block *count* matters on its own, independently of the work per block: a two-block loop
# does not amortise the spawn even when each block is large.
const MIN_BLOCKS_FOR_PARALLEL = 4

threshold(::Transcendental) = THRESHOLD_ELEMENTWISE_TRANSCENDENTAL
threshold(::Arithmetic)     = THRESHOLD_ELEMENTWISE_ARITHMETIC
threshold(::MemoryBound)    = THRESHOLD_MEMORY_BOUND
threshold(::BlockParallel)  = THRESHOLD_BLOCK_PARALLEL

# ------------------------------------------------------------------------------- traits

"""
    cost_class(::Type) -> CostClass

The cost class of an operator's elementwise kernel. Operators declare it with
[`@threadable`](@ref); the conservative default keeps an undeclared operator on the serial
path at every size.
"""
cost_class(::Type) = MemoryBound()
cost_class(f) = cost_class(typeof(f))

"""
    supports_threading(f) -> Bool

Whether `f` has a threaded kernel at all. This is a property of the operator *type*, not of
the `threaded` keyword: it distinguishes "threading was declined" from "there is nothing to
decline", which is what a caller needs in order to interpret `is_threaded(f, x) == false`.
"""
supports_threading(::Type) = false
supports_threading(f) = supports_threading(typeof(f))

"""
    permits_threading(f) -> Bool

Whether `f` was *allowed* to thread, i.e. the `threaded` keyword it was built with. Says
nothing about whether it will: see [`is_threaded`](@ref).
"""
permits_threading(::Type) = false
permits_threading(f) = permits_threading(typeof(f))

"""
    threading_threshold(f) -> Int

Smallest input length at which threading `f` pays off.
"""
threading_threshold(T::Type) = threshold(cost_class(T))
threading_threshold(f) = threading_threshold(typeof(f))

"""
    unthreaded(f) -> f

Rebuild `f` with `threaded = false`, holding every field fixed. `Th` is always the trailing
type parameter that `@threadable` installs, so this reconstructs the type generically instead
of needing a method per operator.

A block-parallel calculus rule (`SeparableSum`, `SlicedSeparableSum`) must call this on each
child before handing it to a spawned task or an inner `@batch` iteration: `Polyester`'s pool
is a single all-or-nothing resource (see `NestedThreadingPolyesterExt`), so a child that still
*tries* to thread pays a full `@batch` launch while running on a pool the parent has already
claimed. Measured on 8 blocks of 2^17 elements: children left threaded inside a spawned block
region cost 1957us against 574us serial (worse than not parallelising the blocks at all);
rebuilding them unthreaded first turns the block parallelism into a real win.
"""
@inline function unthreaded(f::F) where F
    (supports_threading(F) && permits_threading(F)) || return f
    W = Base.typename(F).wrapper
    ps = Base.unwrap_unionall(F).parameters
    G = W{ps[1:(end - 1)]..., false}
    return G((getfield(f, name) for name in fieldnames(F))...)
end

"""
    @threadable NormL1{<:Any, <:Any, Th} Arithmetic

Declare that an operator has a threaded kernel, that its `threaded` keyword is carried in
the type parameter named `Th`, and which cost class its kernel belongs to. The three trait
methods are generated together so they cannot drift apart.
"""
macro threadable(sig, class)
    quote
        $(esc(:(ProximalOperators.supports_threading(::Type{<:$sig}) where {Th} = true)))
        $(esc(:(ProximalOperators.permits_threading(::Type{<:$sig}) where {Th} = Th)))
        $(esc(:(ProximalOperators.cost_class(::Type{<:$sig}) where {Th} = $class())))
        $(esc(:(Base.show(io::IO, f::$sig) where {Th} = ProximalOperators.show_operator(io, f))))
    end
end

# ------------------------------------------------------------------------------ storage
#
# Conservative by default: an array type we know nothing about never gets Julia-level
# threading. The GPU extension adds one method for `AbstractGPUArray`, and that single line
# is what keeps device storage off the threaded path everywhere, without any kernel in the
# package having to know that GPUs exist.

is_cpu_storage(::Type{<:AbstractArray}) = false
is_cpu_storage(::Type{<:Array}) = true

# Wrappers answer for what they wrap. This is not a convenience: a wrapper that fell through
# to the conservative `false` would make a host `Symmetric` look like device storage, and an
# operator that reacts to that by copying to the host would copy, re-wrap, and ask again --
# which is an infinite recursion, not a slow path. Every wrapper the package actually passes
# around is listed for that reason.
is_cpu_storage(::Type{<:Base.ReshapedArray{<:Any, <:Any, P}}) where {P} = is_cpu_storage(P)
is_cpu_storage(::Type{<:SubArray{<:Any, <:Any, P}}) where {P} = is_cpu_storage(P)
is_cpu_storage(::Type{<:Base.ReinterpretArray{<:Any, <:Any, <:Any, P}}) where {P} = is_cpu_storage(P)
is_cpu_storage(::Type{<:Symmetric{<:Any, P}}) where {P} = is_cpu_storage(P)
is_cpu_storage(::Type{<:Hermitian{<:Any, P}}) where {P} = is_cpu_storage(P)
is_cpu_storage(::Type{<:Transpose{<:Any, P}}) where {P} = is_cpu_storage(P)
is_cpu_storage(::Type{<:Adjoint{<:Any, P}}) where {P} = is_cpu_storage(P)
is_cpu_storage(::Type{<:Diagonal{<:Any, P}}) where {P} = is_cpu_storage(P)
is_cpu_storage(x::AbstractArray) = is_cpu_storage(typeof(x))
is_cpu_storage(x::Number) = true
is_cpu_storage(xs::Tuple) = all(is_cpu_storage, xs)

# ------------------------------------------------------------------------------ decision

# `false` vetoes outright; `true` only permits, and lets the policy decline.
@inline _resolve_threaded(policy::F, threaded::Bool) where {F} = threaded ? policy()::Bool : false

"""
    should_thread(f, x) -> Bool

Whether the kernel of `f` should run threaded for input `x`: `f` must permit it, more than
one Julia thread must exist, `x` must live in CPU memory, and it must be long enough to
amortise the spawn.
"""
@inline function should_thread(f::F, x) where {F}
    _resolve_threaded(permits_threading(F)) do
        Threads.nthreads() > 1 &&
            is_cpu_storage(x) &&
            _worklength(x) >= threading_threshold(F)
    end
end

@inline _worklength(x::AbstractArray) = length(x)
@inline _worklength(xs::Tuple) = sum(_worklength, xs; init = 0)
@inline _worklength(::Number) = 1

"""
    is_threaded(f, x) -> Bool

Whether `prox!`/`gradient!` on `f` will actually thread for an input like `x`. This is the
honest reporter: it answers `false` when threading was vetoed, when the input is too small,
when it lives on a device, and when Julia was started single-threaded.

Unlike the operator's `threaded` keyword, this needs an input, because these operators are
size-agnostic at construction.
"""
@inline is_threaded(f, x) = should_thread(f, x)

# --------------------------------------------------------------------------- strategies
#
# Which loop construct to emit. Kept separate from `should_thread` so that the choice of
# *kernel* and the choice of *threaded or not* stay independently testable.

abstract type Strategy end
struct Simd   <: Strategy end   # @inbounds @simd
struct Batch  <: Strategy end   # Polyester.@batch, under a NestedThreading budget
struct Turbo  <: Strategy end   # LoopVectorization.@turbo
struct TTurbo <: Strategy end   # LoopVectorization.@tturbo

is_threaded_strategy(::Strategy) = false
is_threaded_strategy(::Batch) = true
is_threaded_strategy(::TTurbo) = true

"""
    execution_strategy(f, x) -> Strategy

The loop construct to use for `f` on `x`. Transcendental kernels over real arrays go to
LoopVectorization, which vectorises `exp`/`log`; everything else uses `@simd` or Polyester.
Complex arrays never go to LoopVectorization: `check_args` rejects them and silently falls
back to a scalar loop, so dispatching there would only cost a warning.
"""
@inline execution_strategy(f::F, x) where {F} =
    _strategy(cost_class(F), should_thread(f, x), eltype(x))

# Only the element types LoopVectorization actually vectorises are routed to it. `Real` is
# too wide a gate: `BigFloat` gains nothing, and `Bool` arithmetic sends VectorizationBase
# into an infinite recursion rather than falling back cleanly.
const TurboEltype = Union{Float32, Float64}

@inline _strategy(::Transcendental, threaded::Bool, ::Type{<:TurboEltype}) =
    threaded ? TTurbo() : Turbo()
@inline _strategy(::CostClass, threaded::Bool, ::Type) = threaded ? Batch() : Simd()

# ------------------------------------------------------------------------- loop macros
#
# The loop body is spliced in literally rather than passed as a closure, because
# LoopVectorization works on the loop *expression*: `@turbo for i in ...; y[i] = h(x[i]);
# end` cannot see through an opaque `h` and falls back to a scalar loop. That is also why
# the transcendental operators write their loops out instead of calling the helpers in
# `kernels.jl`.
#
# Each macro emits one copy of the body per strategy it can select. The strategy is a
# singleton known from the operator's type, so inference drops the untaken branches and
# nothing is paid at run time.

# Accumulators need one more piece of care than they look like they do. The threaded branch
# compiles to a closure over the loop's variables, and a variable that is both captured by a
# closure and assigned outside it gets boxed for the whole enclosing scope -- so a shared
# `acc` would make the *serial* branch allocate once per iteration (measured: 32 bytes per
# element, 2 MiB for a 2^16 input). Each branch therefore gets its own shadowed copy, and
# the result is assigned back afterwards, which leaves the caller's variable untouched by
# any closure.

# Pull the accumulator names out of a `reduction = ((+, acc), (max, m))` option.
function _reduction_vars(opts)
    vars = Symbol[]
    for opt in opts
        (opt isa Expr && opt.head === :(=) && opt.args[1] === :reduction) || continue
        spec = opt.args[2]
        (spec isa Expr && spec.head === :tuple) || continue
        for pair in spec.args
            if pair isa Expr && pair.head === :tuple && length(pair.args) == 2 &&
               pair.args[2] isa Symbol
                push!(vars, pair.args[2])
            end
        end
    end
    return vars
end

# `Polyester.@batch reduction` infers its accumulator as `Any`, and because `should_thread`
# is a run-time question the strategy is a small `Union`, so that `Any` would propagate out
# of every `prox!` in the package and undo the `@inferred` checks the test suite already
# makes. The accumulator's type is fixed before the branch and reasserted after it, which
# costs nothing at run time and keeps the union-split result concrete.
function _accum_types(vars)
    tvars = [gensym(string("acctype_", v)) for v in vars]
    pre = Expr(:block, (:(local $t = typeof($v)) for (t, v) in zip(tvars, vars))...)
    post = Expr(:block, (:($v = $v::$t) for (t, v) in zip(tvars, vars))...)
    return pre, post
end

# Wrap `body` so that every accumulator it touches is a fresh local, copied in from the
# caller's binding and copied back out on the way.
function _isolate(body, vars)
    isempty(vars) && return body
    shadow = Expr(:block, (Expr(:(=), v, v) for v in vars)...)
    out = length(vars) == 1 ? vars[1] : Expr(:tuple, vars...)
    inner = Expr(:let, shadow, Expr(:block, body, out))
    return Expr(:(=), out, inner)
end

# `minbatch` is what keeps `@batch` from splitting a loop into slivers. The default suits an
# elementwise loop over array entries; a loop whose iterations are whole slices passes its
# own, so it is only supplied when the caller did not.
_has_minbatch(opts) = any(o -> o isa Expr && o.head === :(=) && o.args[1] === :minbatch, opts)

function _check_loop(loop, name)
    (loop isa Expr && loop.head === :for) ||
        error("ProximalOperators: `@$name` expects a `for` loop, got `$loop`")
    return loop
end

# `NestedThreading.@budgeted` decides whether to exclude the `:polyester` pool from the
# budget scope by reading the *name* of the macro it wraps, and it reads that name off a
# `@batch` / `Polyester.@batch` head. A `GlobalRef` head is opaque to that check, so the
# scope would restrict the very pool the loop is about to run on: measured at 2^20 elements,
# a `@budgeted GlobalRef(Polyester, Symbol("@batch"))` loop took 2145 us against 796 us for
# the same loop written as `Polyester.@batch`, i.e. worse than the 673 us serial path. The
# head is therefore emitted as a qualified `.` expression, with the module reached through
# this package's own binding so that it resolves wherever the macro is expanded.
const _BATCH_HEAD = Expr(:., GlobalRef(@__MODULE__, :Polyester), QuoteNode(Symbol("@batch")))

_batch_call(source, opts, loop) = Expr(:macrocall, _BATCH_HEAD, source, opts..., loop)

# Polyester's `@batch` insists on a bare `for` -- it rejects `@inbounds for ...` outright --
# so the elision goes around the loop *body* instead of around the loop.
#
# `@fastmath` goes around the body for the same reason `@simd` goes around the serial
# branch's loop: without it, a division- or branch-heavy body (`SqrHingeLoss`'s prox kernel,
# say) keeps strict IEEE ordering per thread, which serializes the reduction and stalls the
# pipeline on the division latency -- measured at 2038us for 2^20 elements over 4 threads,
# slower than the 574us serial `@simd` path it should be beating. `@simd` grants the serial
# branch that same reassociation freedom already, so giving the threaded branch `@fastmath`
# is matching a relaxation the package already accepts elsewhere, not adding a new one:
# with it, the same body drops to 295us. Every `@elementwise_loop`/`@transcendental_loop`
# reduction in the package already tolerates this (their tests pass under `@simd`), so this
# is not a new numerical contract, only extending the existing one to the threaded branch.
function _inbounds_body(loop::Expr)
    fastmath = Expr(:macrocall, GlobalRef(Base, Symbol("@fastmath")), nothing, copy(loop.args[2]))
    body = Expr(:macrocall, GlobalRef(Base, Symbol("@inbounds")), nothing, fastmath)
    return Expr(:for, copy(loop.args[1]), body)
end

"""
    @elementwise_loop strategy for i in eachindex(x, y) ... end

Run an elementwise loop serially (`@inbounds @simd`) or threaded (`Polyester.@batch` under
a NestedThreading budget), according to `strategy`. For a loop that accumulates, declare
the accumulators with `@elementwise_loop strategy reduction = ((+, acc),) for ...`.
"""
macro elementwise_loop(strategy, args...)
    loop = _check_loop(last(args), "elementwise_loop")
    opts = args[1:(end - 1)]
    vars = _reduction_vars(opts)
    serial = _isolate(:(@inbounds @simd $(copy(loop))), vars)
    batch = _batch_call(__source__,
                        (opts..., (_has_minbatch(opts) ? () : (:(minbatch = 4096),))...),
                        _inbounds_body(loop))
    batch = Expr(:macrocall, GlobalRef(NestedThreading, Symbol("@budgeted")), __source__, batch)
    batch = _isolate(batch, vars)
    pre, post = _accum_types(vars)
    quote
        local strat = $(esc(strategy))
        $(esc(pre))
        if strat isa $Batch
            $(esc(batch))
        else
            $(esc(serial))
        end
        $(esc(post))
        nothing
    end
end

"""
    @transcendental_loop strategy for i in eachindex(x, y) ... end

As [`@elementwise_loop`](@ref), but adds the `@turbo`/`@tturbo` variants that
`execution_strategy` selects for real-valued `exp`/`log` kernels. The `@simd`/`@batch`
branches remain, because complex inputs are not vectorisable by LoopVectorization.
"""
macro transcendental_loop(strategy, args...)
    loop = _check_loop(last(args), "transcendental_loop")
    opts = args[1:(end - 1)]
    vars = _reduction_vars(opts)
    serial = _isolate(:(@inbounds @simd $(copy(loop))), vars)
    batch = _batch_call(__source__,
                        (opts..., (_has_minbatch(opts) ? () : (:(minbatch = 1024),))...),
                        _inbounds_body(loop))
    batch = Expr(:macrocall, GlobalRef(NestedThreading, Symbol("@budgeted")), __source__, batch)
    batch = _isolate(batch, vars)
    turbo = _isolate(Expr(:macrocall, GlobalRef(LoopVectorization, Symbol("@turbo")), __source__,
                 :(warn_check_args = false), copy(loop)), vars)
    tturbo = _isolate(Expr(:macrocall, GlobalRef(LoopVectorization, Symbol("@tturbo")), __source__,
                  :(warn_check_args = false), copy(loop)), vars)
    pre, post = _accum_types(vars)
    quote
        local strat = $(esc(strategy))
        $(esc(pre))
        if strat isa $Turbo
            $(esc(turbo))
        elseif strat isa $TTurbo
            $(esc(tturbo))
        elseif strat isa $Batch
            $(esc(batch))
        else
            $(esc(serial))
        end
        $(esc(post))
        nothing
    end
end

# ---------------------------------------------------------------------------- BLAS bridge
#
# BLAS is not a Julia thread pool: it keeps its own live global thread count and applies its
# own size heuristics, so neither `Threads.nthreads() > 1` nor an element threshold of ours
# is the right question to ask about it. The only thing to decide is whether a `threaded =
# false` operator is allowed to let BLAS thread underneath it -- and it is not, because that
# is exactly the oversubscription a block-parallel parent vetoes it to prevent.
#
# The permitted branch deliberately opens no scope at all: entering one costs several
# microseconds, which is not negligible against the microsecond-scale `mul!`s these
# operators issue.

@inline _with_blas_threading(f::F, threaded::Bool) where {F} =
    threaded ? f() : NestedThreading.with_restricted_threads(f)

@inline _with_blas_threading(f::F, op, x) where {F} =
    _with_blas_threading(f, permits_threading(op))

# ------------------------------------------------------------------------------- display
#
# Adding `Th` makes the default `show` actively unhelpful -- `NormL1{Float64, Nothing,
# true}(3.5, nothing)` tells a reader nothing they wanted to know -- so operators that gain
# the parameter print as the call that would rebuild them instead. `@threadable` installs
# this, so a converted operator cannot forget it.

function show_operator(io::IO, f::F) where {F}
    print(io, nameof(F), "(")
    written = false
    for name in fieldnames(F)
        name === :buf && continue
        written && print(io, ", ")
        show(io, getfield(f, name))
        written = true
    end
    if !permits_threading(F)
        written && print(io, "; ")
        print(io, "threaded = false")
    end
    print(io, ")")
    is_preallocated(f) && print(io, " [preallocated]")
    return nothing
end

# ------------------------------------------------------------------ block loops
#
# Calculus rules whose loop body is a whole child `prox!` are a different decision from an
# elementwise kernel: what matters is the number of blocks as much as their size. A
# two-block loop does not amortise the spawn however large the blocks are, which is why
# `MIN_BLOCKS_FOR_PARALLEL` is a separate condition and not folded into the threshold.
#
# The budget scope alone is not enough for a child that threads through `Polyester`: that
# pool is claimed all-or-nothing (see `NestedThreadingPolyesterExt`), not shared out in
# degrees the way a BLAS thread count is, so a still-threaded child pays a full `@batch`
# launch against a pool with nothing free. `unthreaded` (above) is how these loops veto that:
# each child is rebuilt with `threaded = false` before it is proxed, so it runs its own
# kernel via `@simd` instead of contending for the block loop's pool. A child that calls BLAS
# still benefits from the budget scope on its own, uncounted terms.

"""
    should_thread_blocks(f, nblocks, work) -> Bool

Whether a loop over `nblocks` independent child operators, `work` elements in total, should
run in parallel.
"""
@inline function should_thread_blocks(f::F, nblocks::Integer, work::Integer) where {F}
    _resolve_threaded(permits_threading(F)) do
        Threads.nthreads() > 1 &&
            nblocks >= MIN_BLOCKS_FOR_PARALLEL &&
            work >= THRESHOLD_BLOCK_PARALLEL
    end
end

@inline block_strategy(threaded::Bool) = threaded ? Batch() : Simd()
