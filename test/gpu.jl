#=
# uncomment this block for stand-alone testing
using JLArrays: JLArray
arrayTypes = [JLArray]
=#

import FFTW # ESTIMATE
using LinearAlgebra: norm
using NFFT: plan_nfft, NDFTPlan, size_in, size_out
import NFFT # LINEAR, FULL, TENSOR, POLYNOMIAL
using NFFTTools: sdc
using Test: @test, @testset, @test_throws

m = 5
σ = 2.0

@testset "GPU NFFT Plans" begin
  for arrayType in arrayTypes

    @testset "GPU_NFFT in multiple dimensions" begin
      for (u, N) in enumerate([(256,), (32, 32), (12, 12, 12)])
        eps = [1e-7, 1e-3, 1e-6, 1e-4]
        for (l, window) in enumerate([:kaiser_bessel, :gauss, :kaiser_bessel_rev, :spline])
          D = length(N)
          @info "Testing $arrayType in $D dimensions using $window window"

          J = prod(N)
          k = rand(Float64, D, J) .- 0.5
          p = plan_nfft(Array, k, N; m, σ, window, precompute=NFFT.FULL,
            fftflags=FFTW.ESTIMATE)
          p_d = plan_nfft(arrayType, k, N; m, σ, window, precompute=NFFT.FULL)
          pNDFT = NDFTPlan(k, N)

          fHat = rand(Float64, J) + rand(Float64, J) * im
          f = adjoint(pNDFT) * fHat
          fHat_d = arrayType(fHat)
          fApprox_d = adjoint(p_d) * fHat_d
          fApprox = Array(fApprox_d)
          e = norm(f[:] - fApprox[:]) / norm(f[:])
          @debug "error adjoint nfft " e
          @test e < eps[l]

          gHat = pNDFT * f
          gHatApprox = Array(p_d * arrayType(f))
          e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
          @debug "error nfft " e
          @test e < eps[l]
        end
      end
    end

    @testset "GPU_NFFT batched over transforms and frames" begin
      N = (24, 20)
      J, batch, frames = 300, 3, 2
      k = rand(Float64, 2, J * frames) .- 0.5
      p_d = plan_nfft(arrayType, k, N; m, σ, batch, frames)
      @test size_in(p_d) == (N..., batch, frames)
      @test size_out(p_d) == (J, batch, frames)

      f = randn(ComplexF64, N..., batch, frames)
      fHat = randn(ComplexF64, J, batch, frames)
      g = Array(p_d * arrayType(f))
      h = Array(adjoint(p_d) * arrayType(fHat))
      for t in 1:frames
        p = plan_nfft(Array, k[:, (t - 1) * J .+ (1:J)], N; m, σ, fftflags=FFTW.ESTIMATE)
        for b in 1:batch
          @test g[:, b, t] ≈ p * f[:, :, b, t] rtol = 1e-10
          @test h[:, :, b, t] ≈ adjoint(p) * fHat[:, b, t] rtol = 1e-10
        end
      end
      @test_throws ArgumentError plan_nfft(arrayType, k[:, 1:(end - 1)], N; frames)
      @test_throws DimensionMismatch p_d * arrayType(f[:, :, 1:1, :])
    end

    @testset "GPU_NFFT Sampling Density" begin

      N = (9, 8) # 9×8 grid of equally spaced sampling points
      # create a 10x10 grid of unit spaced sampling points
      T = Float32
      fun = N -> @. T((0:(N-1)) / N - 0.5) # 1D grid
      nodes = hcat([[x; y] for x in fun(N[1]), y in fun(N[2])]...)

      # approximate the density weights
      p = plan_nfft(arrayType, nodes, N; m=5, σ=2.0)
      weights = sdc(p, iters=5)
      weights = Vector(weights)
      @test all(≈(1/prod(N)), weights)
    end
  end
end
