@testset "Accuracy Wrappers" begin

if !Sys.iswindows()

include("../Wrappers/DUCC0.jl")

@testset "DUCC0 Wrapper in multiple dimensions" begin
  for (u,N) in enumerate([(255,), (31,33), (11,12,14)]) # can only do D=1:3
    
    eps = 1e-7
      
    D = length(N)
    @info "Testing in $D dimensions"

    J = prod(N)
    k = rand(Float64,D,J) .- 0.5
    p = Ducc0NufftPlan(k, N) #; m, σ, precompute = pre, fftflags = FFTW.ESTIMATE)
    pNDFT = NDFTPlan(k, N)

    fHat = rand(Float64,J) + rand(Float64,J)*im
    f = adjoint(pNDFT) * fHat
    fApprox = adjoint(p) * fHat

    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error adjoint nfft "  e
    @test e < eps

    gHat = pNDFT * f
    gHatApprox = p * f
    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error nfft "  e
    @test e < eps

  end
end

end

include("../Wrappers/FINUFFT.jl")

@testset "FINUFFT Wrapper in multiple dimensions" begin
  for (u,N) in enumerate([(255,), (31,33), (11,12,14)]) # can only do D=1:3
    
    eps = 1e-7
      
    D = length(N)
    @info "Testing in $D dimensions"

    J = prod(N)
    k = rand(Float64,D,J) .- 0.5
    p = FINUFFTPlan(k, N)
    pNDFT = NDFTPlan(k, N)

    fHat = rand(Float64,J) + rand(Float64,J)*im
    f = adjoint(pNDFT) * fHat
    fApprox = adjoint(p) * fHat

    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error adjoint nfft "  e
    @test e < eps

    gHat = pNDFT * f
    gHatApprox = p * f
    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error nfft "  e
    @test e < eps

  end
end

@testset "FINNUFFT Wrapper in multiple dimensions" begin
  for (u,N) in enumerate([(40,1), (40,2), (40,3)])
    
    eps = 1e-7
    D = N[2]

    J =N[1]
    k = rand(Float64,D,J) .- 0.5
    y = (rand(Float64,D,N[1]) .- 0.5) .* 10

    p = FINNUFFTPlan(k, y) 
    pNNDFT = NNDFTPlan(k, y)

    fHat = rand(Float64,J) + rand(Float64,J)*im
    f = adjoint(pNNDFT) * fHat
    fApprox = adjoint(p) * fHat

    e = norm(f[:] - fApprox[:]) / norm(f[:])
    @debug "error adjoint nnfft "  e
    @test e < eps

    gHat = pNNDFT * f
    gHatApprox = p * f

    e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
    @debug "error nnfft "  e
    @test e < eps

  end
end


end