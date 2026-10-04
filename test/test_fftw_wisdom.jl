using TestItems

@testitem "FFTW wisdom cache: location and opt-out" tags = [:encoding] begin
    using MriReconstructionToolbox: fftw_wisdom_path
    import FFTW

    dir = mktempdir()
    withenv("MRT_FFTW_WISDOM" => dir) do
        path = fftw_wisdom_path()
        @test dirname(path) == dir
        # One file per CPU model and FFTW build, so machines sharing a home keep their own.
        @test startswith(basename(path), "wisdom-") && endswith(path, ".fftw")
    end
    for off in ("off", "0", "false", "OFF")
        withenv("MRT_FFTW_WISDOM" => off) do
            @test isnothing(fftw_wisdom_path())
        end
    end
end

@testitem "FFTW wisdom cache: measured plans are saved and reloaded" tags = [:encoding] begin
    using MriReconstructionToolbox: fftw_wisdom_path, get_fourier_operator, _save_fftw_wisdom, _load_fftw_wisdom
    import FFTW

    dir = mktempdir()
    withenv("MRT_FFTW_WISDOM" => dir) do
        ksp = zeros(ComplexF32, 48, 40, 3)
        get_fourier_operator(ksp, false; fast_planning = false, threaded = false)
        _save_fftw_wisdom()
        path = fftw_wisdom_path()
        @test isfile(path) && filesize(path) > 0
        saved = read(path, String)

        # A fresh process state: the saved wisdom is read back on the next plan.
        FFTW.forget_wisdom()
        MriReconstructionToolbox._WISDOM_LOADED_FROM[] = ""
        _load_fftw_wisdom()
        exported = tempname()
        FFTW.export_wisdom(exported)
        @test sort(readlines(exported)) == sort(split(saved, '\n'; keepempty = false))
    end
end

@testitem "FFTW wisdom cache: an unreadable file is replaced" tags = [:encoding] begin
    using MriReconstructionToolbox: fftw_wisdom_path, get_fourier_operator, _save_fftw_wisdom, _WISDOM_DIRTY
    import FFTW

    dir = mktempdir()
    withenv("MRT_FFTW_WISDOM" => dir) do
        path = fftw_wisdom_path()
        write(path, "not wisdom")
        get_fourier_operator(zeros(ComplexF32, 36, 30, 2), false; fast_planning = false, threaded = false)
        @test _WISDOM_DIRTY[]
        _save_fftw_wisdom()
        @test !_WISDOM_DIRTY[]
        FFTW.import_wisdom(path)
        @test readdir(dir) == [basename(path)]
    end
end

@testitem "FFT planning: fft_planning setting and the MEASURE score" tags = [:encoding, :reconstruction] begin
    using MriReconstructionToolbox: NonCartesianAcquisitionInfo, _fast_planning, _measure_score,
        _applications_per_iteration, DEFAULT_ALGORITHMS

    @test ReconstructionConfig().fft_planning === :auto
    @test_throws ArgumentError ReconstructionConfig(; fft_planning = :patient)

    cart = AcquisitionInfo(zeros(ComplexF32, 128, 128, 8); is3D = false)
    traj = radial_trajectory(256, 128)
    radial = NonCartesianAcquisitionInfo(zeros(ComplexF32, 256, 128, 8); trajectory = traj, image_size = (128, 128))
    cgsense = IterativeReconstruction(; algorithm = CGNR(), maxit = 10)
    admm(maxit) = IterativeReconstruction(TotalVariation2D(1.0e-3); algorithm = ADMM(; cg_maxit = 10), maxit)
    config(mode) = ReconstructionConfig(; fft_planning = mode, threaded = false)

    # A forced mode wins over the score either way.
    @test _fast_planning(admm(20), radial, config(:estimate))
    @test !_fast_planning(cgsense, cart, config(:measure))
    @test !_fast_planning(DirectReconstruction(), cart, config(:measure))

    # ADMM applies the operator once per inner CG step and once for the right-hand side; a tuple of
    # candidates counts as its cheapest member.
    @test _applications_per_iteration(ADMM(; cg_maxit = 10)) == 11
    @test _applications_per_iteration(FISTA()) == 1
    @test _applications_per_iteration(DEFAULT_ALGORITHMS) == 1

    # Measured on one thread: a short Cartesian solve and a direct reconstruction plan with
    # ESTIMATE, a radial ADMM solve (whose oversampled grid ESTIMATE plans badly) with MEASURE.
    @test _fast_planning(cgsense, cart, config(:auto))
    @test _fast_planning(DirectReconstruction(), radial, config(:auto))
    @test !_fast_planning(admm(20), radial, config(:auto))

    # More work raises the score, more FFTW threads lower it, and so does a smaller grid.
    score(m, a; threaded = false) = _measure_score(m, a; threaded)
    @test score(admm(40), cart) > score(admm(20), cart) > score(cgsense, cart)
    @test score(admm(20), radial) > score(admm(20), cart)
    if Threads.nthreads() > 1
        @test score(admm(20), radial; threaded = true) < score(admm(20), radial)
    end
end

@testitem "plan_fft_wisdom fills the cache for an acquisition" tags = [:encoding] begin
    import FFTW

    dir = mktempdir()
    withenv("MRT_FFTW_WISDOM" => dir) do
        acq = simulate_acquisition(
            rand(ComplexF32, 32, 32), AcquisitionInfo(is3D = false, sensitivity_maps = coil_sensitivities(32, 32, 2))
        )
        path = plan_fft_wisdom(acq; rigor = :measure, threaded = false)
        @test path == MriReconstructionToolbox.fftw_wisdom_path()
        @test isfile(path) && filesize(path) > 0
        @test_throws ArgumentError plan_fft_wisdom(acq; rigor = :fast)
    end
end

@testitem "FFTW wisdom cache: device plans leave it alone" tags = [:encoding, :gpu] setup = [GpuEnvSetup, GpuHelpers] begin
    using MriReconstructionToolbox: get_fourier_operator, _WISDOM_DIRTY

    # A device FFT is not FFTW's, so a measured device plan has nothing to add to the host wisdom
    # and must not make the next `reconstruct` rewrite the file.
    for backend in fft_backends()
        _WISDOM_DIRTY[] = false
        get_fourier_operator(to_device(backend, zeros(ComplexF32, 48, 40, 3)), false; fast_planning = false)
        @test !_WISDOM_DIRTY[]
    end
end
