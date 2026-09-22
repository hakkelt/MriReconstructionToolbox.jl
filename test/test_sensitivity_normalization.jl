@testitem "Sensitivity normalization makes the coil sum of squares one" tags = [
    :preprocessing, :acquisition,
] begin
    using LinearAlgebra, Random

    Random.seed!(11)
    nx, ny, nc = 12, 10, 4
    maps = randn(ComplexF32, nx, ny, nc)
    maps[1:3, :, :] .= 0   # a region outside the object, where the sum of squares is zero

    normalized = normalize_sensitivity_maps(maps)
    total = dropdims(sum(abs2, normalized; dims = 3), dims = 3)
    @test eltype(normalized) === ComplexF32
    @test all(isapprox(1), total[4:end, :])
    @test all(iszero, normalized[1:3, :, :])
    # The scaling is a real, positive, per-voxel factor, so what the maps encode — the relative
    # magnitude and phase between coils — is unchanged.
    ratio = normalized[4, 1, :] ./ maps[4, 1, :]
    @test all(r -> isapprox(r, ratio[1]), ratio)
    @test real(ratio[1]) > 0
    # Real up to the round-off of forming the ratio itself; the factor applied was real.
    @test all(r -> abs(imag(r)) < 1.0e-6 * abs(real(r)), ratio)
    # Non-mutating: the input still has its own arbitrary scale.
    @test !all(isapprox(1), dropdims(sum(abs2, maps; dims = 3), dims = 3)[4:end, :])

    # A voxel far below the peak keeps its zeros rather than being divided by noise.
    faint = copy(maps)
    faint[5, 5, :] .*= 1.0f-4
    @test all(iszero, normalize_sensitivity_maps(faint; threshold = 1.0e-3)[5, 5, :])
    @test !all(iszero, normalize_sensitivity_maps(faint; threshold = 0)[5, 5, :])

    # The coil axis is found by name as well as by position.
    named = NamedDimsArray{(:x, :y, :coil)}(copy(maps))
    @test dimnames(normalize_sensitivity_maps(named)) == (:x, :y, :coil)
    @test unname(normalize_sensitivity_maps(named)) ≈ normalized
end

@testitem "Normalized sensitivity maps make the encoding operator a contraction" tags = [
    :preprocessing, :acquisition, :encoding,
] begin
    using LinearAlgebra, Random
    using MriReconstructionToolbox: get_encoding_operator
    import MriReconstructionToolbox.AbstractOperators as AbstractOperators

    Random.seed!(3)
    nx, ny, nc = 16, 16, 4
    maps = NamedDimsArray{(:x, :y, :coil)}(randn(ComplexF32, nx, ny, nc))
    ksp = NamedDimsArray{(:kx, :ky, :coil)}(randn(ComplexF32, nx, ny, nc))
    info = AcquisitionInfo(ksp; sensitivity_maps = maps, image_size = (nx, ny))

    normalized = normalize_sensitivity_maps(info)
    @test info.sensitivity_maps === maps   # the input is left untouched

    # Fully sampled Cartesian SENSE over normalized maps has ‖𝒜‖ = 1 exactly, which is what lets
    # the step size come from a closed-form bound instead of a power iteration.
    E = get_encoding_operator(normalized; threaded = false)
    @test AbstractOperators.estimate_opnorm(E; maxit = 500) ≈ 1 atol = 1.0e-5
    @test AbstractOperators.estimate_opnorm(
        get_encoding_operator(info; threaded = false); maxit = 500
    ) > 1.5

    # An acquisition without maps is returned as it is, not rejected.
    plain = AcquisitionInfo(ksp[:, :, 1]; image_size = (nx, ny))
    @test normalize_sensitivity_maps(plain) === plain
end
