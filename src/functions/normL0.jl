# L0 pseudo-norm (times a constant)

export NormL0

"""
    NormL0(λ=1; threaded=true)

Return the ``L_0`` pseudo-norm function
```math
f(x) = λ\\cdot\\mathrm{nnz}(x)
```
for a nonnegative parameter `λ`.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct NormL0{R, Th}
    lambda::R
    function NormL0{R, Th}(lambda::R) where {R, Th}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

@threadable NormL0{<:Any, Th} Arithmetic

NormL0(lambda::R=1; threaded::Bool=true) where R = NormL0{R, threaded}(lambda)

(f::NormL0)(x) = f.lambda * real(eltype(x))(count(!iszero, x))

function prox!(y, f::NormL0, x, gamma)
    R = real(eltype(x))
    gl = gamma * f.lambda
    thresh = sqrt(2 * gl)
    # counting the survivors off `y` rather than off a separate flag keeps this in the
    # single-reduction shape the shared helper expresses, and so gives it a device path
    countnzy = map_reduce_prox!(
        f, y, xi -> (abs(xi) > thresh) * xi, yi -> iszero(yi) ? R(0) : R(1), x
    )
    return f.lambda * countnzy
end

function prox_naive(f::NormL0, x, gamma)
    over = abs.(x) .> sqrt(2 * gamma * f.lambda)
    y = x.*over
    return y, f.lambda * real(eltype(x))(count(!iszero, y))
end
