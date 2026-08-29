abstract type Regularization end

"""
	calculate(reg, x; threaded=true)

Evaluate the value of a regularization term `reg` at a given point `x`. This function is useful for testing and debugging.
"""
function calculate(reg, x; threaded = true)
    reg_bound = bind_dimensions(reg, dims_of(x))
    x_var = Variable(unname(x))
    t = materialize(reg_bound, x_var; threaded)
    f = StructuredOptimization.extract_functions(t)
    op = StructuredOptimization.extract_affines((x_var,), t)
    x_val = ~x_var
    y = op isa Eye ? x_val : op * x_val
    return f(y)
end

"""
	materialize(reg, x; threaded)

Create a `StructuredOptimization.Term` corresponding to the regularization `reg` applied to the variable `x`.
The `threaded` argument indicates whether to use multi-threading for operations that support it.

# Example
```juliajulia
julia> using MriReconstructionToolbox, StructuredOptimization

julia> x = Variable(8, 8);

julia> reg = L1Image(0.1);

julia> term = materialize(reg, x; threaded=false)
Term{Float64}(1, NormL1{Float64}(0.1), (Variable(Float64, (8, 8), :x)), "0.1 ⋅ ‖x‖₁")
```
"""
function materialize(reg::Regularization, ::Variable; threaded::Bool)
    throw(ArgumentError("materialize not implemented for $(typeof(reg))"))
end

"""
	materialize_with_auxiliaries(reg, x; threaded)

Create the term(s) for `reg` together with the tuple of *auxiliary variables* they introduce.

Most regularizations are a function of the image alone and introduce nothing, so the default simply wraps
[`materialize`](@ref). A few — [`TotalGeneralizedVariation2D`](@ref) is the example — are defined as the
minimum of a joint objective over an extra field, and that field has to become a variable of the optimization
problem: it is solved for alongside the image and discarded afterwards.

Auxiliary variables are initialized to zero and are not part of the reconstructed image, so callers must
recover the image from the image variable itself rather than from the solver's variable ordering.

Returns `(terms, auxiliary_variables::Tuple)`.
"""
function materialize_with_auxiliaries(reg::Regularization, x::Variable; threaded::Bool)
    return materialize(reg, x; threaded), ()
end

"""
	dims_of(x)

The dimension identifiers of `x`: its names when it is a `NamedDimsArray`, its indices otherwise. This is
what a `time_dim`/`dim` argument is resolved against, so that a named dimension stays a `Symbol`.
"""
dims_of(x::AbstractArray) = 1:ndims(x)
dims_of(x::NamedDimsArray) = dimnames(x)

"""
	_check_dim_spec(dim, name; allow_nothing=true)

Validate a dimension a regularization is parameterized by: an `Integer` index, a dimension name
(`Symbol`), or -- when `allow_nothing` -- `nothing` to have it inferred from the image's dimension names.
`name` only names the argument in the error message.
"""
function _check_dim_spec(dim, name::AbstractString; allow_nothing::Bool = true)
    @argcheck (allow_nothing && isnothing(dim)) || dim isa Integer || dim isa Symbol "$name must be an Integer or Symbol"
    if dim isa Integer
        @argcheck dim > 0 "$name must be positive"
    end
    return nothing
end

"""
	materialize_all(regs, x; threaded)

Materialize every regularization in `regs` against the variable `x`, returning
`(term_list::Tuple, auxiliary_variables::Tuple)`: one entry of `term_list` per regularization, and the
concatenated auxiliary variables of all of them (see [`materialize_with_auxiliaries`](@ref)).

The terms are returned as a list rather than already summed because callers differ in what they do with
them: `build_model_with_variables` needs to know whether any auxiliary variable exists *before* it picks
the form of the data term the regularization terms are added to.
"""
function materialize_all(regs, x::Variable; threaded::Bool)
    term_list = ()
    auxiliaries = ()
    for reg in regs
        @argcheck reg isa Regularization "All regularization terms must be of type Regularization."
        reg_terms, reg_auxiliaries = materialize_with_auxiliaries(reg, x; threaded)
        term_list = (term_list..., reg_terms)
        auxiliaries = (auxiliaries..., reg_auxiliaries...)
    end
    return term_list, auxiliaries
end

"""
	get_operator(reg, x; threaded=true)

Get the linear operator associated with the regularization `reg` for an input variable `x`.
The `threaded` argument indicates whether to use multi-threading for operations that support it.

# Example
```julia
julia> using MriReconstructionToolbox

julia> x = Variable(8, 8);

julia> reg = L1Wavelet2D(0.2, wavelet=WT.db2, levels=2);

julia> op = get_operator(reg, ~x; threaded=false)
WaveletOp{Float64,WT.Daubechies{2}}(Float64, (8, 8), WT.Daubechies{2}(), 2)
```
"""
function get_operator(reg::Regularization, x::AbstractArray; threaded::Bool = true)
    throw(ArgumentError("get_operator not implemented for $(typeof(reg))"))
end

"""
	identity_operator(x)

The identity operator on the domain of `x`, wrapped in a [`NamedDimsOp`](@ref) when `x` carries
dimension names. This is what `get_operator` returns for every regularization that acts on the image
itself rather than on a transform of it.
"""
identity_operator(x::AbstractArray) = Eye(x)
identity_operator(x::NamedDimsArray) = NamedDimsOp{dimnames(x), dimnames(x)}(Eye(parent(x)))

"""
	_collapse_direction_axes(op, x, n_components)

Collapse the trailing direction axes of `op`'s codomain into a single one, giving the
`(length(x), n_components)` matrix shape the mixed ℓ₂,₁ norm is taken over: one row per voxel, one column
per direction. A `NamedDimsOp` has to be unwrapped and rewrapped rather than reshaped in place (a
`Reshape` around it collides with its codomain names), and the collapsed codomain no longer maps 1:1 to
the original names, so it gets an anonymous one.
"""
function _collapse_direction_axes(op, x::AbstractArray, n_components::Int)
    inner = op isa NamedDimsOp ? parent(op) : op
    collapsed = reshape(inner, length(x), n_components)
    return op isa NamedDimsOp ? NamedDimsOp{dimnames(x), (:_, :direction)}(collapsed) : collapsed
end

"""
	get_affected_dims(reg, acq_info, image_dims)

Get the dimensions in the image domain that are affected by the regularization `reg`.
This is used to determine which dimensions can be used for problem decomposition during reconstruction.

`acq_info` is the acquisition the term will be applied to, or `nothing` when it is not available. Terms
implement the `::Nothing` method; the `::AcquisitionInfo` one falls back to it, so only a term whose
affected dimensions genuinely depend on the acquisition needs to define both.
"""
function get_affected_dims(reg::Regularization, ::AcquisitionInfo, image_dims)
    return get_affected_dims(reg, nothing, image_dims)
end

function get_affected_dims(::R, ::Nothing, image_dims) where {R <: Regularization}
    throw(ArgumentError("get_affected_dims not implemented for $(R.name.wrapper)"))
end

"""
	scale_regularization(reg, factor)

Return a copy of `reg` with its `λ` adjusted so that, when the image variable itself is scaled by `factor`
(e.g. because problem decomposition uses one common data-scaling factor for all slices instead of a per-slice
one), the regularization term has the same relative strength as it would with an unscaled variable and the
original `λ`.

For a term `λ^k ⋅ h(x)` with `h` homogeneous of degree `p` in `x`, this requires `λ_eff = λ ⋅ factor^((2-p)/k)`.
L1-type terms (`k=1, p=1`, e.g. `L1Image`, `L1Wavelet2D/3D`, `TotalVariation2D/3D`, `TemporalFourier`, `LowRank`)
therefore scale `λ` linearly with `factor`. Quadratic penalties (`Tikhonov`, `k=2, p=2`) and rank constraints
(`RankLimit`, which has no `λ`) are already scale-consistent and need no correction; the default falls back to
returning `reg` unchanged.
"""
scale_regularization(reg::Regularization, factor::Real) = reg

"""
	bind_dimensions(reg, image_dims)

Return a copy of `reg` with any symbol-based dimension parameters (such as `time_dim=:time` or `dim=:echo`,
or `nothing` for inferred dimensions) resolved to 1-based integer dimension indices against `image_dims`.
"""
bind_dimensions(reg::Regularization, image_dims) = reg
bind_dimensions(reg::Regularization, ::Nothing) = reg
bind_dimensions(regs::Tuple, image_dims) = map(r -> bind_dimensions(r, image_dims), regs)
bind_dimensions(regs::Tuple, ::Nothing) = regs
