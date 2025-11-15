using Test
using MriReconstructionToolbox
using LinearAlgebra

# Helpers: simple phantoms and coil maps
function phantom2d(nx::Int, ny::Int)
    x = range(-1, 1; length=nx)
    y = range(-1, 1; length=ny)
    X = repeat(collect(x), 1, ny)
    Y = repeat(collect(y)', nx, 1)
    p = zeros(Float32, nx, ny)
    # Large ellipse
    p .+= ifelse.((X.^2 ./ 0.9^2 .+ Y.^2 ./ 0.8^2) .<= 1, 1.0f0, 0.0f0)
    # Smaller high-intensity circle
    p .+= ifelse.(((X .- 0.3).^2 .+ (Y .+ 0.2).^2) .<= 0.15^2, 0.8f0, 0.0f0)
    # Negative ellipse (hole)
    p .-= ifelse.(((X .+ 0.2).^2 ./ 0.25^2 .+ (Y .- 0.1).^2 ./ 0.15^2) .<= 1, 0.3f0, 0.0f0)
    return complex.(p)
end

function phantom3d(nx::Int, ny::Int, nz::Int)
    base = phantom2d(nx, ny)
    slabs = [Float32(0.9 + 0.2 * (k - 1) / max(1, nz - 1)) .* base for k in 1:nz]
    vol = cat(slabs...; dims=3)
    return complex.(vol)
end

function coil_smaps2d(nx::Int, ny::Int, nc::Int)
    x = range(-1, 1; length=nx)
    y = range(-1, 1; length=ny)
    X = repeat(collect(x), 1, ny)
    Y = repeat(collect(y)', nx, 1)
    smaps = Array{ComplexF32}(undef, nx, ny, nc)
    centers = [(cos(2π*(i-1)/nc), sin(2π*(i-1)/nc)) for i in 1:nc]
    for i in 1:nc
        cx, cy = centers[i]
        σ = 0.6f0
        mag = @. exp(-((X - cx)^2 + (Y - cy)^2) / (2σ^2))
        phase = @. exp(im * (0.5f0 * X + 0.3f0 * Y))
        smaps[:, :, i] = ComplexF32.(mag .* phase)
    end
    denom = sqrt.(sum(abs2, smaps; dims=3) .+ eps(Float32))
    smaps ./= denom
    return smaps
end

function coil_smaps3d(nx::Int, ny::Int, nz::Int, nc::Int)
    sm2d = coil_smaps2d(nx, ny, nc)
    sm3d = repeat(sm2d, 1, 1, 1, nz)
    permutedims(sm3d, (1, 2, 4, 3))  # (x,y,z,coil)
end

objective(𝒜, y, x, regs...) = 0.5 * sum(abs2, 𝒜 * x .- y) + sum(calculate(r, x; threaded=true) for r in regs)

function random_mask(nx, ny)
    mask = falses(nx, ny);
    counter = 0
    while counter < nx * ny ÷ 4
        i = 0
        while i < 1 || i > nx
            i = Int(randn() / 3 * nx ÷ 2 + nx ÷ 2 + 1)
        end
        j = 0
        while j < 1 || j > ny
            j = Int(randn() / 3 * ny ÷ 2 + ny ÷ 2 + 1)
        end
        if !mask[i, j]
            mask[i, j] = true
            counter += 1
        end
    end
    return mask
end

#=@testset "reconstruct: direct fully-sampled 2D single-coil" begin
    nx, ny = 32, 32
    x = ComplexF32.(phantom2d(nx, ny))
    E = get_encoding_operator(zeros(ComplexF32, nx, ny), false)
    y = E * x
    x̂ = reconstruct(y, false)
    @test x̂ ≈ E' * y atol=1e-5 rtol=1e-5
end

@testset "reconstruct: direct fully-sampled 3D parallel" begin
    nx, ny, nz, nc = 24, 24, 12, 4
    x = ComplexF32.(phantom3d(nx, ny, nz))
    smaps = coil_smaps3d(nx, ny, nz, nc)
    E = get_encoding_operator(zeros(ComplexF32, nx, ny, nz, nc), true; sensitivity_maps=smaps)
    y = E * x
    x̂ = reconstruct(y, true; sensitivity_maps=smaps)
    @test x̂ ≈ E' * y atol=1e-5 rtol=1e-5
end

@testset "reconstruct: iterative 2D subsampled (Tikhonov / CG)" begin
    nx, ny = 32, 32
    x = ComplexF32.(phantom2d(nx, ny))
    mask = falses(nx, ny); mask[:, 1:2:ny] .= true
    E = get_encoding_operator(zeros(ComplexF32, count(mask)), false; img_size=(nx, ny), subsampling=mask)
    y = E * x
    reg = Tikhonov(0.01f0)
    alg = (CG(; maxit=10), FISTA(; maxit=10), ADMM(; maxit=10))
    x̂ = reconstruct(y, false; img_size=(nx, ny), subsampling=mask, regularization=reg, algorithm=alg)
    x0 = E' * y
    @test objective(E, y, x̂, reg) <= objective(E, y, x0, reg) + 1e-5
end

@testset "reconstruct: iterative 2D subsampled (L1Wavelet / FISTA)" begin
    nx, ny = 32, 32;
    x = ComplexF32.(phantom2d(nx, ny));
    mask = random_mask(nx, ny);
    E = get_encoding_operator(zeros(ComplexF32, count(mask)), false; img_size=(nx, ny), subsampling=mask)
    y = E * x;
    reg = L1Wavelet2D(0.01)
    alg = (CG(; maxit=10), FISTA(; maxit=20), ADMM(; maxit=10));
    x̂ = reconstruct(y, false; img_size=(nx, ny), subsampling=mask, regularization=reg, algorithm=alg)
    x0 = E' * y
    @test objective(E, y, x̂, reg) <= objective(E, y, x0, reg) + 1e-5
end

@testset "reconstruct: iterative 3D subsampled (L1Wavelet+TV / ADMM)" begin
    nx, ny, nz = 24, 24, 12
    x = ComplexF32.(phantom3d(nx, ny, nz))
    mask = falses(nx, ny, nz); mask[:, 1:2:ny, :] .= true
    E = get_encoding_operator(zeros(ComplexF32, count(mask)), true; img_size=(nx, ny, nz), subsampling=mask)
    y = E * x
    regs = (L1Wavelet3D(0.01f0), TotalVariation3D(0.01f0))
    alg = (CG(; maxit=10), FISTA(; maxit=15), ADMM(; maxit=20))
    x̂ = reconstruct(y, true; img_size=(nx, ny, nz), subsampling=mask, regularization=regs, algorithm=alg)
    x0 = E' * y
    @test objective(E, y, x̂, regs...) <= objective(E, y, x0, regs...) + 1e-5
end

@testset "reconstruct: iterative 2D subsampled parallel (L1Wavelet)" begin
    nx, ny, nc = 32, 32, 4
    x = ComplexF32.(phantom2d(nx, ny))
    smaps = coil_smaps2d(nx, ny, nc)
    mask = falses(nx, ny); mask[:, 1:3:ny] .= true
    E = get_encoding_operator(zeros(ComplexF32, count(mask), nc), false; sensitivity_maps=smaps, image_size=(nx, ny), subsampling=mask)
    y = E * x
    reg = L1Wavelet2D(0.02f0)
    alg = (CG(; maxit=10), FISTA(; maxit=20), ADMM(; maxit=10))
    x̂ = reconstruct(y, false; sensitivity_maps=smaps, img_size=(nx, ny), subsampling=mask, regularization=(reg,), algorithm=alg)
    x0 = E' * y
    @test objective(E, y, x̂, reg) <= objective(E, y, x0, reg) + 1e-5
end=#