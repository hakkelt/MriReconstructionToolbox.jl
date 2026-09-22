using GPUEnv

GPUEnv.activate(; include_jlarrays = true, persist = true)

using ProximalAlgorithms: PANOCplus, FastForwardBackward, ADMM

# Generic-array coverage: exercises the same small end-to-end problems the rest of the
# suite runs on CPU, but with data on every GPUArrays-compatible backend GPUEnv finds on
# this host (JLArrays always, plus real backends such as CUDA). Each check compares a
# GPU-array solve against the CPU-array solve of the same problem, and confirms the
# result never silently falls back to a CPU array.
for backend in gpu_backends(; include_jlarrays = true)
    @testset "GPU backend: $(backend.name)" begin
        A, b = randn(6, 5), randn(6)
        Ag, bg = to_gpu(backend, A), to_gpu(backend, b)

        @testset "ls + norm(x,1): $alg" for (alg, alg_kwargs) in (
                (PANOCplus, (tol = 1.0e-8,)),
                (FastForwardBackward, (tol = 1.0e-8,)),
                (ADMM, (maxit = 2000, rho = 1.0)),
            )
            x_cpu = Variable(5)
            solve(problem(ls(A * x_cpu - b) + 0.05 * norm(x_cpu, 1)), alg(; alg_kwargs...))

            x_gpu = Variable(gpu_zeros(backend, Float64, 5))
            solve(problem(ls(Ag * x_gpu - bg) + 0.05 * norm(x_gpu, 1)), alg(; alg_kwargs...))

            @test typeof(~x_gpu) == typeof(gpu_zeros(backend, Float64, 5))
            @test Array(~x_gpu) ≈ ~x_cpu rtol = 1.0e-3
        end

        @testset "hingeloss with a GPU label vector" begin
            y = sign.(randn(5))
            yg = to_gpu(backend, y)

            x_cpu, x_gpu = Variable(5), Variable(gpu_zeros(backend, Float64, 5))
            t_cpu, t_gpu = hingeloss(x_cpu, y), hingeloss(x_gpu, yg)

            v = randn(5)
            vg = to_gpu(backend, v)
            @test t_gpu.f(vg) ≈ t_cpu.f(v) rtol = 1.0e-8
        end

        @testset "bare Variable round trip" begin
            xg = Variable(to_gpu(backend, zeros(5)))
            @test typeof(~xg) == typeof(to_gpu(backend, zeros(5)))
        end
    end
end
