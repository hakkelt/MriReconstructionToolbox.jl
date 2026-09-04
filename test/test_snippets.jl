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
