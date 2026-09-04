using TestItems

@testsnippet RegTestSetup begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets
end

@testsnippet ProxOf begin
    const SO = MriReconstructionToolbox.StructuredOptimization
    const PC = MriReconstructionToolbox.ProximalCore

    function functions_of(reg, x)
        term = MriReconstructionToolbox.materialize(reg, Variable(x); threaded = false)
        return SO.extract_functions(term)
    end

    function prox_of(reg, x, γ = 1.0)
        y = similar(x)
        value = PC.prox!(y, functions_of(reg, x), x, γ)
        return y, value
    end
end

@testsnippet TestHelpers begin
    using LinearAlgebra: norm

    relative_error(z, truth) = norm(z .- truth) / norm(truth)
    test_type_stable(::Type{T}, value) where {T} = (Test.@test typeof(value) == T; value)
end

@testsnippet FiniteDiff begin
    # Forward difference at the first index along dimension `d`, backward difference elsewhere --
    # matches the boundary convention `get_operator` uses for the (Second)TotalVariation family.
    function manual_gradient(x::AbstractArray, ndims_spatial::Int)
        step(d) = CartesianIndex(ntuple(k -> k == d ? 1 : 0, ndims(x)))
        manual = zeros(eltype(x), size(x)..., ndims_spatial)
        for d in 1:ndims_spatial, idx in CartesianIndices(x)
            manual[idx, d] = if idx[d] == first(axes(x, d))
                x[idx + step(d)] - x[idx]
            else
                x[idx] - x[idx - step(d)]
            end
        end
        return manual
    end
end

@testsnippet SyntheticCoils begin
    # A smooth, complex-valued coil pattern: a Gaussian blob offset around a ring per coil, with a
    # linear phase ramp, normalized so the coils combine to unit magnitude (root-sum-of-squares).
    function synthetic_sensitivities(::Type{T}, Nx, Ny, Nc; phase_scale = 0.5) where {T}
        X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
        Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
        sens = zeros(T, Nx, Ny, Nc)
        for c in 1:Nc
            a = (c - 1) * 2π / Nc
            sens[:, :, c] = exp.(-((X .- cos(a) / 2) .^ 2 .+ (Y .- sin(a) / 2) .^ 2)) .*
                cis.(phase_scale .* (X .* cos(a) .+ Y .* sin(a)))
        end
        sens ./= sqrt.(sum(abs2, sens; dims = 3)) .+ 1.0e-8
        return sens
    end
end

@testsnippet ModelEval begin
    function eval_term(terms)
        vars = StructuredOptimization.extract_variables(terms)
        @assert length(vars) == 1
        xvar = vars[1]
        f = StructuredOptimization.extract_functions(terms)
        op = StructuredOptimization.extract_operators((xvar,), terms)
        xval = ~xvar
        return f(op * xval)
    end
end
