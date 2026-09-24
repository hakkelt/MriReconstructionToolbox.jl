export OperatorPool, with_operator_pool, recycle!

"""
	OperatorPool()

Storage that lets many operators of the same shape be built one after another without
allocating their working memory each time.

Inside [`with_operator_pool`](@ref), the intermediate buffers a [`Compose`](@ref) allocates
between its operators are taken from the arrays [`recycle!`](@ref) returned to the pool, when
one of the right type and size is there.

A pool may be shared by several tasks at once: taking and returning buffers is locked, and a
buffer taken by one task is not handed to another until it is recycled again. The results of
operators built with a pool are identical to those of operators built without one.
"""
struct OperatorPool
    buffers::Vector{Array}
    lock::Threads.SpinLock
end
OperatorPool() = OperatorPool(Array[], Threads.SpinLock())

const _POOL_KEY = :AbstractOperators_OperatorPool

"""
	with_operator_pool(f, pool::OperatorPool)

Call `f()` with `pool` active for operator construction in the current task. The previous pool,
if any, is restored afterwards. Tasks spawned inside `f` do not inherit the pool.
"""
function with_operator_pool(f, pool::OperatorPool)
    tls = task_local_storage()
    prev = get(tls, _POOL_KEY, nothing)
    tls[_POOL_KEY] = pool
    try
        return f()
    finally
        if prev === nothing
            delete!(tls, _POOL_KEY)
        else
            tls[_POOL_KEY] = prev
        end
    end
end

_active_pool() = get(task_local_storage(), _POOL_KEY, nothing)::Union{Nothing, OperatorPool}

"""
	_pooled_codomain_buffer(L::AbstractOperator)

An array for the codomain of `L`: one of the active pool's recycled arrays of the same type and
size if there is one, otherwise `allocate_in_codomain(L)`.
"""
function _pooled_codomain_buffer(L::AbstractOperator)
    pool = _active_pool()
    if pool !== nothing && codomain_array_type(L) <: Array
        T = codomain_type(L)
        sz = size(L, 1)
        buf = @lock pool.lock begin
            i = findfirst(b -> eltype(b) === T && size(b) == sz, pool.buffers)
            i === nothing ? nothing : popat!(pool.buffers, i)
        end
        buf === nothing || return buf
    end
    return allocate_in_codomain(L)
end

"""
	recycle!(pool::OperatorPool, L::AbstractOperator)
	recycle!(L::AbstractOperator)

Return the intermediate buffers of every [`Compose`](@ref) inside `L` to `pool` (the active
pool, in the second form; without one it does nothing), so that the next operator built under
the pool reuses them. `L`, and every operator that shares buffers with it, must not be applied
again afterwards.
"""
function recycle!(pool::OperatorPool, L::AbstractOperator)
    found = Base.IdSet{Array}()
    _collect_buffers!(found, L)
    @lock pool.lock append!(pool.buffers, found)
    return pool
end

function recycle!(L::AbstractOperator)
    pool = _active_pool()
    pool === nothing || recycle!(pool, L)
    return nothing
end

function _collect_buffers!(found, L::Compose)
    for b in L.buf
        b isa Array && push!(found, b)
    end
    foreach(A -> _collect_buffers!(found, A), L.A)
    return found
end

function _collect_buffers!(found, L)
    for name in fieldnames(typeof(L))
        isdefined(L, name) || continue
        _collect_field_buffers!(found, getfield(L, name))
    end
    return found
end

_collect_field_buffers!(found, v::AbstractOperator) = _collect_buffers!(found, v)
function _collect_field_buffers!(found, v::Union{Tuple, AbstractVector{<:AbstractOperator}})
    for e in v
        e isa AbstractOperator && _collect_buffers!(found, e)
    end
    return found
end
_collect_field_buffers!(found, _) = found
