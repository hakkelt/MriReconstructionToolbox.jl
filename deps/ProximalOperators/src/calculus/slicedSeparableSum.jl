# Separable sum, using slices of an array as variables

export SlicedSeparableSum

"""
    SlicedSeparableSum((f_1, ..., f_k), (J_1, ..., J_k); threaded=true)

Return the function
```math
g(x) = \\sum_{i=1}^k f_i(x_{J_i}).
```

    SlicedSeparableSum(f, (J_1, ..., J_k))

Analogous to the previous one, but apply the same function `f` to all slices
of the variable `x`:
```math
g(x) = \\sum_{i=1}^k f(x_{J_i}).
```

With `threaded = true` (the default) the slices are proxed in parallel when there are
enough of them to be worth it; see [`is_threaded`](@ref). Each slice is independent, so
this is the calculus rule that benefits most.
"""
struct SlicedSeparableSum{S <: Tuple, T <: AbstractArray, N, Th}
  fs::S    # Tuple, where each element is a Vector with elements of the same type; the functions to prox on
  # Example: S = Tuple{Array{ProximalOperators.NormL1{Float64},1}, Array{ProximalOperators.NormL2{Float64},1}}
  idxs::T  # Vector, where each element is a Vector containing the indices to prox on
  # Example: T = Array{Array{Tuple{Colon,UnitRange{Int64}},1},1}
end

function SlicedSeparableSum(fs::Tuple, idxs::Tuple; threaded::Bool=true)
    ftypes = DataType[]
    fsarr = Array{Any,1}[]
    indarr = Array{eltype(idxs),1}[]
    for (i,f) in enumerate(fs)
        t = typeof(f)
        fi = findfirst(isequal(t), ftypes)
        if fi === nothing
            push!(ftypes, t)
            push!(fsarr, Any[f])
            push!(indarr, eltype(idxs)[idxs[i]])
        else
            push!(fsarr[fi], f)
            push!(indarr[fi], idxs[i])
        end
    end
    fsnew = ((Array{typeof(fs[1]),1}(fs) for fs in fsarr)...,)
    @assert typeof(fsnew) == Tuple{(Array{ft,1} for ft in ftypes)...}
    SlicedSeparableSum{typeof(fsnew),typeof(indarr),length(fsnew),threaded}(fsnew, indarr)
end

# Constructor for the case where the same function is applied to all slices
SlicedSeparableSum(f::F, idxs::T; threaded::Bool=true) where {F, T <: Tuple} =
SlicedSeparableSum(Tuple(f for k in eachindex(idxs)), idxs; threaded)

@threadable SlicedSeparableSum{<:Any, <:Any, <:Any, Th} BlockParallel

# The per-type loops live in ordinary functions rather than inside the generated body: the
# elements of one group all share a type, so the loop is type-stable and `@batch` can take
# it, and keeping it out of the `@generated` expression keeps macro expansion predictable.
# `minbatch = 1` stops Polyester from subdividing a single slice.
#
# When the outer loop over slices threads, each slice's own function is proxed through
# `unthreaded`: `Polyester`'s pool is claimed all-or-nothing, so a slice that still permits
# threading would spend a `@batch` launch trying to claim a pool this loop already holds
# (see `unthreaded` in `src/utilities/execution.jl`). When the outer loop stays serial there
# is nothing to veto, so the slice keeps whatever permission it was built with.

function _call_slices(strategy::Strategy, fs, idxs, x)
    v = 0.0
    @elementwise_loop strategy reduction = ((+, v),) minbatch = 1 for k in eachindex(fs)
        fk = strategy isa Batch ? unthreaded(fs[k]) : fs[k]
        idx = idxs[k]
        v += fk(idx isa Tuple ? view(x, idx...) : view(x, idx))
    end
    return v
end

function _prox_slices!(strategy::Strategy, y, fs, idxs, x, gamma)
    v = 0.0
    @elementwise_loop strategy reduction = ((+, v),) minbatch = 1 for k in eachindex(fs)
        fk = strategy isa Batch ? unthreaded(fs[k]) : fs[k]
        idx = idxs[k]
        v += if idx isa Tuple
            prox!(view(y, idx...), fk, view(x, idx...), gamma)
        else
            prox!(view(y, idx), fk, view(x, idx), gamma)
        end
    end
    return v
end

# how many slices there are in total, known from the type alone for the block-count test
nslices(f::SlicedSeparableSum) = sum(length, f.fs)

# Unroll the loop over the different types of functions to evaluate
@generated function (f::SlicedSeparableSum{A, B, N})(x) where {A, B, N}
    ex = :(v = 0.0)
    for i = 1:N # For each function type
        ex = quote $ex;
            v += _call_slices(strategy, f.fs[$i], f.idxs[$i], x)
        end
    end
    quote
        strategy = block_strategy(should_thread_blocks(f, nslices(f), length(x)))
        $ex
        return v
    end
end

# Unroll the loop over the different types of functions to prox on
@generated function prox!(y, f::SlicedSeparableSum{A, B, N}, x, gamma) where {A, B, N}
    ex = :(v = 0.0)
    for i = 1:N # For each function type
        ex = quote $ex;
            v += _prox_slices!(strategy, y, f.fs[$i], f.idxs[$i], x, gamma)
        end
    end
    quote
        strategy = block_strategy(should_thread_blocks(f, nslices(f), length(x)))
        $ex
        return v
    end
end

component_types(::Type{<:SlicedSeparableSum{S}}) where {S} = Tuple(A.parameters[1] for A in fieldtypes(S))

# no scratch space of its own: slices are proxed in place through views, so
# only the component functions need preallocating, each for its own slice
function preallocate(f::SlicedSeparableSum{S, T, N, Th}, x) where {S, T, N, Th}
    fs = map(f.fs, (f.idxs...,)) do fs_group, idxs_group
        [preallocate(fi, idx isa Tuple ? view(x, idx...) : view(x, idx))
         for (fi, idx) in zip(fs_group, idxs_group)]
    end
    return SlicedSeparableSum{typeof(fs), T, N, Th}(fs, f.idxs)
end

@generated is_proximable(::Type{T}) where T <: SlicedSeparableSum = return all(is_proximable, component_types(T)) ? :(true) : :(false)
@generated is_convex(::Type{T}) where T <: SlicedSeparableSum = return all(is_convex, component_types(T)) ? :(true) : :(false)
@generated is_set_indicator(::Type{T}) where T <: SlicedSeparableSum = return all(is_set_indicator, component_types(T)) ? :(true) : :(false)
@generated is_singleton_indicator(::Type{T}) where T <: SlicedSeparableSum = return all(is_singleton_indicator, component_types(T)) ? :(true) : :(false)
@generated is_cone_indicator(::Type{T}) where T <: SlicedSeparableSum = return all(is_cone_indicator, component_types(T)) ? :(true) : :(false)
@generated is_affine_indicator(::Type{T}) where T <: SlicedSeparableSum = return all(is_affine_indicator, component_types(T)) ? :(true) : :(false)
@generated is_smooth(::Type{T}) where T <: SlicedSeparableSum = return all(is_smooth, component_types(T)) ? :(true) : :(false)
@generated is_generalized_quadratic(::Type{T}) where T <: SlicedSeparableSum = return all(is_generalized_quadratic, component_types(T)) ? :(true) : :(false)
@generated is_strongly_convex(::Type{T}) where T <: SlicedSeparableSum = return all(is_strongly_convex, component_types(T)) ? :(true) : :(false)

function prox_naive(f::SlicedSeparableSum, x, gamma)
    fy = zero(eltype(x))
    y = similar(x)
    for t in eachindex(f.fs)
        for k in eachindex(f.fs[t])
            y[f.idxs[t][k]...], fy1 = prox_naive(f.fs[t][k], x[f.idxs[t][k]...], gamma)
            fy += fy1
        end
    end
    return y, fy
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{T}) where T <: SlicedSeparableSum = _combine_tiers(map(device_tier, component_types(T))...)
