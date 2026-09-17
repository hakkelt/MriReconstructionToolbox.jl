"""
    normdiff2(x, y)

`norm(x - y, 2)^2` without forming `x - y`.

Written as a two-array `mapreduce` rather than as an indexed loop, so that it runs as a
device reduction on GPU storage as well as a fused pass on the host. (`zip` would not do:
an iterator of tuples falls back to a scalar `foldl`, which is exactly the thing that cannot
run on a device.) The three calculus rules that use it -- `Regularize`, `MoreauEnvelope` and
`PointwiseMinimum` -- are otherwise entirely storage-agnostic, and an indexed loop here was
the only thing keeping them off the device.
"""
function normdiff2(x::AbstractArray{C}, y::AbstractArray{C}) where {
    R <: Real, C <: Union{R, Complex{R}}
}
    return mapreduce((a, b) -> abs2(a - b), +, x, y; init = R(0))
end

"""
Fast (non-allocating) version of norm(x-y,2)
"""
normdiff(x, y) = sqrt(normdiff2(x, y))
