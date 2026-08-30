function check_x₀_components_size(x₀, components, image_size)
    x₀ isa Union{Tuple, NamedTuple} ||
        throw(ArgumentError("x₀ for image decomposition must be `nothing`, a Tuple, or a NamedTuple of per-component arrays."))
    if x₀ isa Tuple
        @argcheck length(x₀) == length(components) "x₀ tuple must have one entry per component ($(length(components))), got $(length(x₀))."
    else
        # A key that matches no component is silently dropped by `get_component_x0s`, which then
        # warm-starts that component from zero instead -- so a typo would cost a whole initial guess
        # without any indication. Reject it here instead.
        component_names = map(c -> c.name, components)
        unknown = filter(k -> k ∉ component_names, keys(x₀))
        unknown_str = join(unknown, ", ")
        known_str = join(component_names, ", ")
        @argcheck isempty(unknown) "x₀ names ($unknown_str) do not match any component ($known_str)."
    end
    for x in values(x₀)
        @argcheck size(x) == image_size "Size of x₀ ($(size(x))) must match the image size ($image_size)"
    end
    return nothing
end

# Component initialisation: the first component gets the direct-recon estimate x̂
# (standard L+S/RPCA warm start), the rest start at zero. `x₀` (nothing / Tuple /
# NamedTuple of per-component arrays) overrides this default per component.
#
# Shape and name validation lives in `check_x₀_components_size`, which every entry point runs on the
# caller's `x₀` before the problem is (possibly) sliced; these methods only select and copy.
function get_component_x0s(components, x̂, ::Nothing)
    n = length(components)
    return ntuple(i -> i == 1 ? copy(unname(x̂)) : zero(unname(x̂)), n)
end

get_component_x0s(components, x̂, x₀::Tuple) = map(unname, x₀)

function get_component_x0s(components, x̂, x₀::NamedTuple)
    return ntuple(length(components)) do i
        name = components[i].name
        if haskey(x₀, name)
            unname(x₀[name])
        elseif i == 1
            copy(unname(x̂))
        else
            zero(unname(x̂))
        end
    end
end
