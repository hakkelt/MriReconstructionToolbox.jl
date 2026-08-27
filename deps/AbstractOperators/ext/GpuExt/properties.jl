array_type_display_string(::Type{<:AbstractGPUArray}) = "ᵍᵖᵘ"

_should_thread(::AbstractGPUArray) = false
_should_thread(::Type{<:AbstractGPUArray}) = false
