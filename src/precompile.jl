# A precompile workload: the three reconstructions most sessions start with, at a size that
# takes milliseconds, so the methods a first `reconstruct` compiles are stored in the package
# image. Plans use `ESTIMATE` and the wisdom cache is off, so precompilation neither measures
# transforms nor reads or writes the user's wisdom file.

PrecompileTools.@setup_workload begin
    n, nc, nt = 16, 2, 3
    img = ComplexF32.([hypot(i - n / 2, j - n / 3) < n / 4 for i in 1:n, j in 1:n])
    PrecompileTools.@compile_workload begin
        withenv("RISTRETTO_FFTW_WISDOM" => "off") do
            Base.ScopedValues.with(_FFTW_RIGOR => FFTW.ESTIMATE) do
                smaps = coil_sensitivities(n, n, nc)
                pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.2), (n, n))

                acq = AcquisitionInfo(is3D = false, image_size = (n, n), subsampling = pattern, sensitivity_maps = smaps)
                data = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
                reconstruct(data; verbosity = Silent())
                reconstruct(data, IterativeReconstruction(L1Wavelet2D(0.005f0); maxit = 2); verbosity = Silent())

                traj = Float32.(radial_trajectory(2n, 8))
                radial = simulate_acquisition(img, AcquisitionInfo(; trajectory = traj, image_size = (n, n), sensitivity_maps = smaps); inverse_crime_check = false, keep_sensitivity_maps = true)
                reconstruct(radial, IterativeReconstruction(TotalVariation2D(0.001f0); maxit = 2); verbosity = Silent())

                cine = NamedDimsArray{(:x, :y, :time)}(stack(circshift(img, (0, t)) for t in 1:nt))
                acq_cine = AcquisitionInfo(
                    is3D = false, image_size = (n, n), subsampling = pattern,
                    sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(smaps),
                )
                data_cine = simulate_acquisition(cine, acq_cine; inverse_crime_check = false, keep_sensitivity_maps = true)
                reconstruct(data_cine, IterativeReconstruction(LowRank(0.01f0; time_dim = :time); maxit = 2); verbosity = Silent())
            end
        end
    end
end
