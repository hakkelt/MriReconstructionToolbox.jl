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

    function prox_of(reg, x, γ = 1.0)
        term = MriReconstructionToolbox.materialize(reg, Variable(x); threaded = false)
        y = similar(x)
        value = PC.prox!(y, SO.extract_functions(term), x, γ)
        return y, value
    end
end
