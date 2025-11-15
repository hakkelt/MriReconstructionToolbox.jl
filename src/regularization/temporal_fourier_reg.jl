"""
    TemporalFourier(λ; time_dim=nothing)

Create a temporal Fourier regularization term with parameter `λ`. The regularization term is given by `λ‖𝓕ₜ{x}‖₁`,
where `𝓕ₜ` is the discrete Fourier transform along the temporal dimension specified by `time_dim`. If `time_dim`
is not provided, it will be inferred as the dimension named `:time` if `x` is a `NamedDimsArray`.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array of the same size as `x`.
- `time_dim`: (optional) Dimension along which to apply the Fourier transform. Can be an `Integer` (1-based index)
or a `Symbol` (dimension name). If not provided, it will be inferred as the dimension named `:time` if `x` is a `NamedDimsArray`.
"""
struct TemporalFourier{T,D} <: Regularization
    λ::T
    time_dim::D
    function TemporalFourier(λ::T; time_dim::D=nothing) where {T,D}
        @argcheck isnothing(time_dim) || time_dim isa Integer || time_dim isa Symbol "time_dim must be an Integer or Symbol"
        if time_dim isa Integer
            @argcheck time_dim > 0 "time_dim must be positive"
        end
        return new{T,D}(λ, time_dim)
    end
end

function get_operator(reg::TemporalFourier, x::AbstractArray; threaded::Bool=true)
    time_dim = get_time_dim(reg.time_dim, x)
    F = DFT(unname(x), time_dim; threaded)
    if x isa NamedDimsArray
        transformed_dimnames = dimnames(x)
        transformed_dimnames[time_dim] = :frequency
        F = NamedDimsOp{transformed_dimnames}(F)
    end
    return F
end

function get_affected_dims(reg::TemporalFourier, ::AcquisitionInfo, image_dims)
    return (get_time_dim(reg.time_dim, image_dims),)
end

function materialize(reg::TemporalFourier, x::Variable{T}; threaded::Bool) where {T}
    R = real(T)
    λ = R.(reg.λ)
    op = get_operator(reg, ~x; threaded)
    repr = if reg.λ isa AbstractArray
        "‖Γ .* 𝓕ₜ$(get_name(x))‖₁"
    else
        @sprintf "%g ⋅ ‖𝓕ₜ%s‖₁" λ get_name(x)
    end
    return StructuredOptimization.Term(1, NormL1(λ), op * x, repr)
end
