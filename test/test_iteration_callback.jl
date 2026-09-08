using TestItems

@testitem "on_iteration fires once per iteration" tags = [:reconstruction, :minimizer] setup = [IterationCallbackSetup] begin
    acq, _ = square_acquisition()

    calls = NamedTuple[]
    method = IterativeReconstruction(
        L1Image(0.01); algorithm = FISTA(), maxit = 7, tol = 0,
        on_iteration = info -> push!(calls, info),
    )
    x̂ = reconstruct(acq, method; verbosity = Silent())

    @test length(calls) == 7
    @test [c.iteration for c in calls] == 1:7
    # The last iterate the callback sees is the image `reconstruct` returns, in the same units.
    @test calls[end].x ≈ x̂
    @test issorted([c.elapsed_ns for c in calls])
    @test all(c -> !haskey(c, :slice), calls)
end

@testitem "on_iteration stops early with the solve" tags = [:reconstruction, :minimizer] setup = [IterationCallbackSetup] begin
    acq, _ = square_acquisition()

    counter = Ref(0)
    method = IterativeReconstruction(
        L1Image(0.01); algorithm = FISTA(), maxit = 50, tol = 1.0e-1,
        on_iteration = _ -> (counter[] += 1),
    )
    reconstruct(acq, method; verbosity = Silent())

    @test 0 < counter[] < 50
end

@testitem "on_iteration works for every default algorithm" tags = [:reconstruction, :minimizer] setup = [IterationCallbackSetup] begin
    acq, x_true = square_acquisition()

    # `DouglasRachford` cannot parse a least-squares fidelity plus an L1 term, so it gets the
    # prox-able fidelity it is actually chosen for; everything else runs the same problem.
    cases = (
        (CG(), (), L2Loss(), (:residual_norm,)),
        (CGNR(), (), L2Loss(), (:residual_norm,)),
        (
            FISTA(), (L1Image(0.01),), L2Loss(),
            (:objective, :smooth_value, :nonsmooth_value, :stepsize, :fixed_point_residual),
        ),
        (
            ISTA(), (L1Image(0.01),), L2Loss(),
            (:objective, :smooth_value, :nonsmooth_value, :stepsize, :fixed_point_residual),
        ),
        (ADMM(), (L1Image(0.01),), L2Loss(), (:primal_residual, :dual_residual, :iterate_change)),
        (
            DouglasRachford(), (L1Image(0.01),), HardConsistency(),
            (:objective, :smooth_value, :nonsmooth_value, :fixed_point_residual),
        ),
    )

    @testset "$(nameof(typeof(algorithm)))" for (algorithm, regs, fidelity, metric_keys) in cases
        trace = IterationTrace(x -> Float64(sum(abs2, x)))
        method = IterativeReconstruction(;
            regularization = regs, algorithm, fidelity, maxit = 6, tol = 0,
            on_iteration = trace,
        )
        x̂ = reconstruct(acq, method; verbosity = Silent())

        @test !isempty(trace)
        @test trace.iterations == 1:length(trace)
        @test trace.values[end] ≈ Float64(sum(abs2, x̂))
        @test keys(trace.metrics[1]) == metric_keys
        @test all(isfinite, values(trace.metrics[1]))
    end
end

@testitem "on_iteration under task splitting tags every slice" tags = [:reconstruction, :minimizer, :integration] setup = [IterationCallbackSetup] begin
    acq = multislice_acquisition()

    @testset "$(nameof(typeof(executor)))" for executor in
        (MriReconstructionToolbox.SequentialExecutor(), MriReconstructionToolbox.MultiThreadingExecutor())
        trace = IterationTrace(x -> Float64(sum(abs2, x)))
        method = IterativeReconstruction(
            L2Image(0.01); algorithm = FISTA(), maxit = 5, tol = 0, on_iteration = trace,
        )
        reconstruct(acq, method; task_executor = executor, verbosity = Silent())

        # 4 slabs x 5 iterations, each entry naming the slab it came from. Order is up to the
        # executor, so only the per-slice counts are asserted.
        @test length(trace) == 20
        @test length(trace.slices) == 20
        @test length(unique(trace.slices)) == 4
        @test all(s -> count(==(s), trace.slices) == 5, unique(trace.slices))
    end
end

@testitem "on_iteration on the component path yields a DecomposedImage" tags = [:reconstruction, :minimizer, :components] setup = [IterationCallbackSetup] begin
    acq, _ = square_acquisition()

    trace = IterationTrace()
    method = IterativeReconstruction(
        Component(:sparse, L1Image(0.01)), Component(:smooth, L2Image(0.01));
        algorithm = FISTA(), maxit = 4, tol = 0, on_iteration = trace,
    )
    img = reconstruct(acq, method; verbosity = Silent())

    @test length(trace) == 4
    @test all(v -> v isa DecomposedImage, trace.values)
    @test propertynames(trace.values[end]) == propertynames(img)
    @test total_image(trace.values[end]) ≈ total_image(img)
    @test trace.values[end].sparse ≈ img.sparse
end

@testitem "on_iteration leaves Verbose output intact" tags = [:reconstruction, :minimizer] setup = [IterationCallbackSetup] begin
    acq, _ = square_acquisition()

    lines = String[]
    trace = IterationTrace()
    method = IterativeReconstruction(
        L1Image(0.01); algorithm = FISTA(), maxit = 6, tol = 0, on_iteration = trace,
    )
    reconstruct(
        acq, method;
        verbosity = Verbose(; printfunc = (args...) -> push!(lines, string(args...)), freq = 2),
    )

    @test length(trace) == 6
    @test !isempty(lines)
    # The solver's own periodic table is still printed, at the requested frequency.
    @test count(l -> occursin("|", l), lines) >= 3
end

@testitem "on_iteration belongs to the method, not the config" tags = [:reconstruction, :minimizer] setup = [IterationCallbackSetup] begin
    acq, _ = square_acquisition()

    err = try
        reconstruct(acq, IterativeReconstruction(L1Image(0.01)); on_iteration = _ -> nothing)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("IterativeReconstruction", err.msg)
end

@testitem "IterationTrace collects, narrows and resets" tags = [:reconstruction, :minimizer] setup = [IterationCallbackSetup] begin
    trace = IterationTrace(x -> sum(x))

    @test isempty(trace)
    @test occursin("0 iterations", sprint(show, trace))

    for k in 1:3
        trace((; iteration = k, x = fill(Float64(k), 2), elapsed_ns = UInt64(k * 1000), objective = 1.0 / k))
    end

    @test length(trace) == 3
    @test trace.iterations == [1, 2, 3]
    # The reduction returns a scalar, so the collected column is a concrete vector, ready to plot.
    @test trace.values isa Vector{Float64}
    @test trace.values == [2.0, 4.0, 6.0]
    @test trace.times ≈ [1.0e-6, 2.0e-6, 3.0e-6]
    @test [m.objective for m in trace.metrics] == [1.0, 0.5, 1 / 3]
    @test isempty(trace.slices)

    empty!(trace)
    @test isempty(trace)
    @test isempty(trace.metrics)
end

@testitem "IterationTrace is safe to call from several tasks" tags = [:reconstruction, :minimizer] setup = [IterationCallbackSetup] begin
    trace = IterationTrace(x -> Float64(sum(x)))
    n = 200

    @sync for t in 1:4
        Threads.@spawn for k in 1:n
            trace((; iteration = k, x = [1.0], elapsed_ns = UInt64(k), slice = "[:, :, $t]"))
        end
    end

    @test length(trace) == 4n
    @test length(trace.slices) == 4n
    @test all(t -> count(==("[:, :, $t]"), trace.slices) == n, 1:4)
    @test trace.values == fill(1.0, 4n)
end
