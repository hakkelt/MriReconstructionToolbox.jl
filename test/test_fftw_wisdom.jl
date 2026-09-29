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
