module RecursiveArrayToolsExt
using RecursiveArrayTools
using ProximalOperators
using ProximalOperators: preallocate
import ProximalCore: prox, prox!, gradient, gradient!

# An `ArrayPartition` is unwrapped to its tuple of blocks before being handed to
# the wrapped function, so preallocation follows the same route.
ProximalOperators.preallocate(g::SeparableSum, xs::ArrayPartition) = preallocate(g, xs.x)

(f::PrecomposedSlicedSeparableSum)(x::ArrayPartition) = f(x.x)
prox!(y::ArrayPartition, f::PrecomposedSlicedSeparableSum, x::ArrayPartition, gamma) = prox!(y.x, f, x.x, gamma)

(g::SeparableSum)(xs::ArrayPartition) = g(xs.x)
prox!(ys::ArrayPartition, g::SeparableSum, xs::ArrayPartition, gamma::Number) = prox!(ys.x, g, xs.x, gamma)
prox!(ys::ArrayPartition, g::SeparableSum, xs::ArrayPartition, gammas::Tuple) = prox!(ys.x, g, xs.x, gammas)
function prox(g::SeparableSum, xs::ArrayPartition, gamma=1)
    y, fy = prox(g, xs.x, gamma)
    return ArrayPartition(y), fy
end
gradient!(grads::ArrayPartition, g::SeparableSum, xs::ArrayPartition) = gradient!(grads.x, g, xs.x)
function gradient(g::SeparableSum, xs::ArrayPartition)
    grad, f_val = gradient(g, xs.x)
    return ArrayPartition(grad), f_val
end

end # module RecursiveArrayToolsExt
