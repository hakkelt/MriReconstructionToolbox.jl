abstract type Regularization end

"""
	calculate(reg, x; threaded=true)

Evaluate the value of a regularization term `reg` at a given point `x`. This function is useful for testing and debugging.
"""
function calculate(reg, x; threaded = true)
    x_var = Variable(x)
    t = materialize(reg, x_var; threaded)
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
	get_affected_dims(reg, x)

Get the dimensions in the image domain that are affected by the regularization `reg`.
This is used to determine which dimensions can be used for problem decomposition during reconstruction.
"""
function get_affected_dims(::R, acq_info::AcquisitionInfo, image_dims) where {R <: Regularization}
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
