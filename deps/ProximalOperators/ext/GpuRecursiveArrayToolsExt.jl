module GpuRecursiveArrayToolsExt

using GPUArrays: AbstractGPUArray
using LinearAlgebra: LinearAlgebra
using RecursiveArrayTools: ArrayPartition

# RecursiveArrayTools has no `dot`, `axpy!`, `axpby!` or `norm` for an `ArrayPartition`, so the
# generic methods iterate it element by element, which a device array refuses. A sum of terms keeps
# its dual (and a multi-variable problem its iterates) in one, and the solvers' inner products and
# updates run on it. Each is taken block by block here.
const DevicePartition = ArrayPartition{<:Any, <:Tuple{Vararg{AbstractGPUArray}}}

LinearAlgebra.dot(x::DevicePartition, y::DevicePartition) = sum(map(LinearAlgebra.dot, x.x, y.x))

function LinearAlgebra.axpy!(α::Number, x::DevicePartition, y::DevicePartition)
    foreach((xi, yi) -> LinearAlgebra.axpy!(α, xi, yi), x.x, y.x)
    return y
end

function LinearAlgebra.axpby!(α::Number, x::DevicePartition, β::Number, y::DevicePartition)
    foreach((xi, yi) -> LinearAlgebra.axpby!(α, xi, β, yi), x.x, y.x)
    return y
end

LinearAlgebra.norm(x::DevicePartition) = sqrt(sum(xi -> LinearAlgebra.norm(xi)^2, x.x))

end # module
