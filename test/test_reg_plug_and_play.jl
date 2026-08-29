using TestItems

@testitem "PlugAndPlay regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    const SO = MriReconstructionToolbox.StructuredOptimization
    const PC = MriReconstructionToolbox.ProximalCore

    # A soft-thresholding "denoiser": with σ = strength√γ it removes exactly γλ when strength = √λ, so the
    # plug-and-play prox must coincide with the prox of `L1Image(λ)`. This is the reference the whole
    # machinery is checked against, since no closed form exists for a real denoiser.
    soft_threshold(image, σ) = sign.(image) .* max.(abs.(image) .- σ^2, 0)

    function prox_of(reg, x, γ = 1.0)
        term = MriReconstructionToolbox.materialize(reg, Variable(x); threaded = false)
        y = similar(x)
        value = PC.prox!(y, SO.extract_functions(term), x, γ)
        return y, value
    end

    @testset "Constructor" begin
        reg = PlugAndPlay(soft_threshold; strength = 0.3)
        @test reg.strength == 0.3
        @test reg.complex_handling == :split
        @test reg.spatial_dims === nothing
        @test_throws ArgumentError PlugAndPlay(soft_threshold; strength = -1)
        @test_throws ArgumentError PlugAndPlay(soft_threshold; complex_handling = :real)
        @test_throws ArgumentError PlugAndPlay(soft_threshold; spatial_dims = 0)
    end

    @testset "get_operator" begin
        x = randn(8, 8)
        @test get_operator(PlugAndPlay(soft_threshold), x; threaded = false) isa Eye
        named = NamedDimsArray{(:x, :y)}(randn(8, 8))
        @test get_operator(PlugAndPlay(soft_threshold), named; threaded = false) isa
            MriReconstructionToolbox.NamedDimsOp
    end

    @testset "a soft-threshold denoiser reproduces L1Image" for λ in (0.05, 0.4), γ in (0.3, 1.0)
        x = randn(8, 8, 3)
        pnp, _ = prox_of(PlugAndPlay(soft_threshold; strength = sqrt(λ)), x, γ)
        l1, _ = prox_of(L1Image(λ), x, γ)
        @test pnp ≈ l1
    end

    @testset "the value function is unavailable" begin
        x = randn(8, 8)
        @test isnan(MriReconstructionToolbox.calculate(PlugAndPlay(soft_threshold; strength = 0.2), x))
        _, value = prox_of(PlugAndPlay(soft_threshold; strength = 0.2), x)
        @test isnan(value)
    end

    @testset "complex handling" begin
        x = randn(ComplexF64, 8, 8)
        σ = sqrt(0.1)

        split, _ = prox_of(PlugAndPlay(soft_threshold; strength = σ, complex_handling = :split), x)
        @test split ≈ complex.(soft_threshold(real(x), σ), soft_threshold(imag(x), σ))

        magnitude, _ = prox_of(PlugAndPlay(soft_threshold; strength = σ, complex_handling = :magnitude), x)
        @test magnitude ≈ soft_threshold(abs.(x), σ) .* cis.(angle.(x))
        # the phase must survive untouched wherever the magnitude did
        kept = abs.(magnitude) .> 0
        @test angle.(magnitude[kept]) ≈ angle.(x[kept])

        native, _ = prox_of(PlugAndPlay(soft_threshold; strength = σ, complex_handling = :native), x)
        @test native ≈ soft_threshold(x, σ)
    end

    @testset "the denoiser is applied per batch slice" begin
        # A denoiser that reports the size it was handed, so the slicing itself can be checked.
        seen = Tuple{Vararg{Int}}[]
        recording(image, σ) = (push!(seen, size(image)); image)
        x = randn(6, 6, 4, 2)
        prox_of(PlugAndPlay(recording; spatial_dims = 2), x)
        @test length(seen) == 8
        @test all(==((6, 6)), seen)

        empty!(seen)
        prox_of(PlugAndPlay(recording; spatial_dims = 3), x)
        @test length(seen) == 2
        @test all(==((6, 6, 4)), seen)
    end

    @testset "get_affected_dims follows the denoiser's window" begin
        @test MriReconstructionToolbox.get_affected_dims(
            PlugAndPlay(soft_threshold), nothing, (:x, :y, :slice)
        ) == (:x, :y)
        @test MriReconstructionToolbox.get_affected_dims(
            PlugAndPlay(soft_threshold; spatial_dims = 3), nothing, (:x, :y, :z, :time)
        ) == (:x, :y, :z)
    end

    @testset "scale_regularization scales the noise level" begin
        reg = MriReconstructionToolbox.scale_regularization(
            PlugAndPlay(soft_threshold; strength = 0.2, complex_handling = :magnitude), 2.5
        )
        @test reg.strength ≈ 0.5
        @test reg.complex_handling == :magnitude
    end
end
