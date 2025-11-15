"""
    LowRank(λ; time_dim=nothing)

Create a low-rank regularization term with parameter `λ`. The regularization term is given by `λ‖𝓧‖_*`,
where `𝓧` is the Casorati matrix formed by unfolding the input variable
`x` along the temporal dimension specified by `time_dim`. If `time_dim`
is not provided, it will be inferred as the dimension named `:time` if `x` is a `NamedDimsArray`.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array of the same size as `x`.
- `time_dim`: (optional) Dimension along which to form the Casorati matrix. Can be an `Integer` (1-based index)
or a `Symbol` (dimension name). If not provided, it will be inferred as the dimension named `:time` if `x` is a `NamedDimsArray`.

# Notes:
- The nuclear norm `‖𝓧‖_*` is the sum of the singular values of the matrix `𝓧`.
- The Casorati matrix is formed by reshaping `x` such that the specified `time_dim` becomes the second dimension,
and all other dimensions before it are reshaped into columns (first dimension), while dimensions after it are treated as batch dimensions.
"""
struct LowRank{T,D} <: Regularization
    λ::T
    time_dim::D
    function LowRank(λ::T; time_dim::D=nothing) where {T,D}
        _check_lowrank_dims(time_dim)
        return new{T,D}(λ, time_dim)
    end
end

"""
    RankLimit(max_rank; time_dim=nothing)

Create a rank limit regularization term that constrains the rank of the Casorati matrix formed by unfolding
the input variable `x` along the temporal dimension specified by `time_dim`. The rank of the Casorati matrix
will be limited to `max_rank`. If `time_dim` is not provided, it will be inferred as the dimension named `:time`
if `x` is a `NamedDimsArray`.

# Arguments
- `max_rank`: Maximum allowed rank for the Casorati matrix.
- `time_dim`: (optional) Dimension along which to form the Casorati matrix. Can be an `Integer` (1-based index)
or a `Symbol` (dimension name). If not provided, it will be inferred as the dimension named `:time` if `x` is
a `NamedDimsArray`.

# Notes:
- The nuclear norm `‖𝓧‖_*` is the sum of the singular values of the matrix `𝓧`.
- The Casorati matrix is formed by reshaping `x` such that the specified `time_dim` becomes the second dimension,
and all other dimensions before it are reshaped into columns (first dimension), while dimensions after it are treated as batch dimensions.
"""
struct RankLimit{D} <: Regularization
    max_rank::Int
    time_dim::D
    function RankLimit(max_rank::Int; time_dim::D=nothing) where {D}
        @argcheck max_rank > 0 "max_rank must be positive"
        _check_lowrank_dims(time_dim)
        return new{D}(max_rank, time_dim)
    end
end

function _check_lowrank_dims(time_dim)
    @argcheck isnothing(time_dim) || time_dim isa Integer || time_dim isa Symbol "time_dim must be an Integer or Symbol"
    if time_dim isa Integer
        @argcheck time_dim > 0 "time_dim must be positive"
    end
end

function get_operator(reg::Union{LowRank,RankLimit}, x::AbstractArray; threaded::Bool=true)
    affected_dims = get_affected_dims(reg, nothing, x)
    batch_dims = length(affected_dims)+1:ndims(x)
    @argcheck length(batch_dims) >= 0 "Input variable must have at least as many dimensions as affected_dims"
    if isempty(batch_dims)
        ℰ = Eye(unname(x))
    else
        x_view = @view x[fill(:, length(affected_dims))..., (ones(Int, length(batch_dims))...)]
        ℰ = Eye(unname(x_view))
    end
    ℛ = reshape(ℰ, :, size(x, affected_dims[end])) # Collapse all but time_dim
    if !isempty(batch_dims)
        ℛ = BatchOp(ℛ, size(x)[length(affected_dims)+1:end]; threaded)
    end
    if x isa NamedDimsArray
        transformed_dimnames = (:space, dimnames(x, affected_dims[end]))
        if !isempty(batch_dims)
            transformed_dimnames = (transformed_dimnames..., batch_dims...)
        end
        ℛ = NamedDimsOp{dimnames(x),transformed_dimnames}(ℛ)
    end
end

function materialize(reg::LowRank, x::Variable{T}; threaded::Bool) where {T}
    R = real(T)
    λ = R.(reg.λ)
    op = get_operator(reg, ~x; threaded)
    repr = if reg.λ isa AbstractArray
        "λ ⋅ ‖Γ .* (ℛ * $(get_name(x)))‖_*"
    else
        @sprintf "%g ⋅ ‖ℛ * %s‖_*" λ get_name(x)
    end
    return StructuredOptimization.Term(1, NuclearNorm(λ), op * x, repr)
end

function materialize(reg::RankLimit, x::Variable{T}; threaded::Bool) where {T}
    op = get_operator(reg, ~x; threaded)
    repr = @sprintf "rank(ℛ * %s) ≤ %d" get_name(x) reg.max_rank
    return StructuredOptimization.Term(1, IndBallRank(reg.max_rank), op * x, repr)
end

function get_affected_dims(reg::Union{LowRank,RankLimit}, ::Union{Nothing,AcquisitionInfo}, image_dims)
    time_dim = get_time_dim(reg.time_dim, image_dims)
    if time_dim isa Symbol
        time_dim_idx = findfirst(==(time_dim), image_dims)
        return image_dims[1:time_dim_idx]
    else
        return 1:time_dim
    end
end
