# L2,1 norm/Sum of norms of columns or rows (times a constant)

export NormL21

"""
    NormL21(λ=1, dim=1; threaded=true)

Return the "sum of ``L_2`` norm" function
```math
f(X) = λ⋅∑_i\\|x_i\\|
```
for a nonnegative `λ`, where ``x_i`` is the ``i``-th column of ``X`` if `dim == 1`, and the ``i``-th row of ``X`` if `dim == 2`.
In words, it is the sum of the Euclidean norms of the columns or rows.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct NormL21{R, I, Th}
    lambda::R
    dim::I
    function NormL21{R,I,Th}(lambda::R, dim::I) where {R, I, Th}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda, dim)
        end
    end
end

is_convex(f::Type{<:NormL21}) = true

# The loop here is over *slices*, not elements, so it is classed as block-parallel: the
# per-iteration work is a whole column (or row) reduction, and `minbatch = 1` keeps `@batch`
# from subdividing a slice.
@threadable NormL21{<:Any, <:Any, Th} BlockParallel

NormL21(lambda::R=1, dim::I=1; threaded::Bool=true) where {R, I} =
    NormL21{R, I, threaded}(lambda, dim)

# On a device the nested slice loops become a `dims` reduction plus a broadcast: two kernel
# launches over the whole array instead of one launch per column, and no scalar indexing.
function _normL21_call_device(f, X)
    R = real(eltype(X))
    nrm = sqrt.(sum(abs2, X; dims = f.dim))
    return f.lambda * sum(nrm)
end

function _normL21_prox_device!(Y, f, X, gamma)
    R = real(eltype(X))
    gl = gamma * f.lambda
    nrm = sqrt.(sum(abs2, X; dims = f.dim))
    scal = max.(R(1) .- gl ./ nrm, R(0))
    Y .= scal .* X
    return f.lambda * sum(scal .* nrm)
end

function (f::NormL21)(X)
    is_cpu_storage(typeof(X)) || return _normL21_call_device(f, X)
    R = real(eltype(X))
    n21X = R(0)
    strategy = execution_strategy(f, X)
    if f.dim == 1
        @elementwise_loop strategy reduction = ((+, n21X),) minbatch = 1 for j in axes(X, 2)
            nslice = R(0)
            for i in axes(X, 1)
                nslice += abs(X[i, j])^2
            end
            n21X += sqrt(nslice)
        end
    elseif f.dim == 2
        @elementwise_loop strategy reduction = ((+, n21X),) minbatch = 1 for i in axes(X, 1)
            nslice = R(0)
            for j in axes(X, 2)
                nslice += abs(X[i, j])^2
            end
            n21X += sqrt(nslice)
        end
    end
    return f.lambda * n21X
end

function prox!(Y, f::NormL21, X, gamma)
    is_cpu_storage(typeof(X)) || return _normL21_prox_device!(Y, f, X, gamma)
    R = real(eltype(X))
    gl = gamma * f.lambda
    n21X = R(0)
    strategy = execution_strategy(f, X)
    if f.dim == 1
        @elementwise_loop strategy reduction = ((+, n21X),) minbatch = 1 for j in axes(X, 2)
            nslice = R(0)
            for i in axes(X, 1)
                nslice += abs(X[i, j])^2
            end
            nslice = sqrt(nslice)
            scal = 1 - gl / nslice
            scal = scal <= 0 ? R(0) : scal
            for i in axes(X, 1)
                Y[i, j] = scal * X[i, j]
            end
            n21X += scal * nslice
        end
    elseif f.dim == 2
        @elementwise_loop strategy reduction = ((+, n21X),) minbatch = 1 for i in axes(X, 1)
            nslice = R(0)
            for j in axes(X, 2)
                nslice += abs(X[i, j])^2
            end
            nslice = sqrt(nslice)
            scal = 1-gl/nslice
            scal = scal <= 0 ? R(0) : scal
            for j in axes(X, 2)
                Y[i, j] = scal * X[i, j]
            end
            n21X += scal * nslice
        end
    end
    return f.lambda * n21X
end

function prox_naive(f::NormL21, X, gamma)
    Y = max.(0, 1 .- f.lambda * gamma ./ sqrt.(sum(abs.(X).^2, dims=f.dim))) .* X
    return Y, f.lambda * sum(sqrt.(sum(abs.(Y).^2, dims=f.dim)))
end
