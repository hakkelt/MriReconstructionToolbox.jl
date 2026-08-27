# CPU/GPU mutual rejection via check overrides.
# These ensure that mixing CPU and GPU arrays in mul! gives a clear error message.
#
# GPU-GPU and GPU-ArrayPartition overloads use `invoke` to call the base
# AbstractOperators.check directly, bypassing these specialised methods.
# Without `invoke`, the GPU-specific methods would re-dispatch to themselves
# (infinite recursion) because AbstractGPUArray <: AbstractArray.


# GPU-GPU: delegate to the base check (handles ArrayPartition too)
function check(y::AbstractGPUArray, A, b::AbstractGPUArray)
    invoke(AbstractOperators.check, Tuple{Any, Any, Any}, y, A, b)
    return nothing
end

# ArrayPartition containers may hold GPU arrays — delegate to base check
function check(y::AbstractGPUArray, A, b::ArrayPartition)
    invoke(AbstractOperators.check, Tuple{Any, Any, Any}, y, A, b)
    return nothing
end

function check(y::ArrayPartition, A, b::AbstractGPUArray)
    invoke(AbstractOperators.check, Tuple{Any, Any, Any}, y, A, b)
    return nothing
end

# GPU output + CPU input: error (plain arrays only, not ArrayPartition)
function check(y::AbstractGPUArray, A, b::AbstractArray)
    throw(
        ArgumentError(
            "Cannot use CPU input $(typeof(b)) with GPU output $(typeof(y)). " *
                "Ensure both arrays have the same storage type.",
        ),
    )
end

# CPU output + GPU input: error (plain arrays only, not ArrayPartition)
function check(y::AbstractArray, A, b::AbstractGPUArray)
    throw(
        ArgumentError(
            "Cannot use GPU input $(typeof(b)) with CPU output $(typeof(y)). " *
                "Ensure both arrays have the same storage type.",
        ),
    )
end
