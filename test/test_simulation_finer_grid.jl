@testsetup module FinerGridSetup
using Ristretto, NamedDims
export blob, rel

# An off-centre Gaussian on an `n`-voxel grid over the field of view [-1, 1]^D, sampled at the
# voxel centres `-1 + (i - 1/2) 2/n`. Its spectrum is negligible beyond a 64-voxel grid's band,
# so finer grids must give the same data once restricted to that band.
function blob(dims::Int...; σ = 0.12, centre = (0.21, -0.17, 0.09))
    coords(n) = [-1 + (i - 0.5) * 2 / n for i in 1:n]
    axes = coords.(dims)
    return [
        ComplexF64(exp(-sum(((axes[d][I[d]] - centre[d]) / σ)^2 for d in 1:length(dims)) / 2))
            for I in CartesianIndices(dims)
    ]
end
rel(a, b) = sqrt(sum(abs2, unname(a) .- unname(b)) / sum(abs2, unname(b)))
end
@testitem "simulate_acquisition: a finer phantom gives the data of the reconstruction grid" tags = [:simulation] setup = [FinerGridSetup] begin
    using Ristretto, NamedDims
    # Even and odd reconstruction and phantom grids; centred and FFT-ordered k-space; image origin
    # at the first or the centre voxel, together and apart.
    for n in (64, 63), (sK, sI) in (((), ()), ((1, 2), (1, 2)), ((1,), ()), ((), (2,))), fine in (101, 128, 77)
        acq = AcquisitionInfo(is3D = false, image_size = (n, n), shifted_kspace_dims = sK, shifted_image_dims = sI)
        coarse = simulate_acquisition(blob(n, n), acq; inverse_crime_check = false).kspace_data
        finer = simulate_acquisition(blob(fine, fine), acq; inverse_crime_check = false).kspace_data
        @test size(finer) == size(coarse)
        @test rel(finer, coarse) < 1.0e-6
    end
    # 3D, odd and even sizes
    acq3 = AcquisitionInfo(is3D = true, image_size = (32, 31, 30))
    coarse = simulate_acquisition(blob(32, 31, 30), acq3; inverse_crime_check = false).kspace_data
    finer = simulate_acquisition(blob(51, 50, 47), acq3; inverse_crime_check = false).kspace_data
    @test rel(finer, coarse) < 1.0e-6
    # A constant object has the same zero-frequency sample on every grid.
    acq = AcquisitionInfo(is3D = false, image_size = (40, 40))
    dc(k) = k[21, 21]
    @test dc(simulate_acquisition(ones(ComplexF64, 63, 63), acq).kspace_data) ≈ 40^2
end

@testitem "simulate_acquisition: finer phantom with coils, subsampling and batch axes" tags = [:simulation] setup = [FinerGridSetup] begin
    using Ristretto, NamedDims
    n, fine, nc = 48, 77, 3
    maps_fine = ComplexF64.(coil_sensitivities(fine, fine, nc))
    pattern = create_sampling_pattern(UniformRandomSampling(2.0, 0.1), (n, n))
    acq = AcquisitionInfo(is3D = false, image_size = (n, n), subsampling = pattern, sensitivity_maps = maps_fine)
    img = stack(blob(fine, fine; centre = (c, 0.1)) for c in (-0.2, 0.0, 0.3))
    data = simulate_acquisition(img, acq; inverse_crime_check = false)
    # The same data from the coil images resampled to the reconstruction grid.
    coil_images = Ristretto._resample_sensitivity_maps(reshape(maps_fine, fine, fine, nc, 1) .* reshape(img, fine, fine, 1, 3), 2, (n, n))
    reference = simulate_acquisition(coil_images, AcquisitionInfo(is3D = false, image_size = (n, n), subsampling = pattern); inverse_crime_check = false)
    @test size(data.kspace_data) == size(reference.kspace_data)
    @test rel(data.kspace_data, reference.kspace_data) < 1.0e-10
    @test isnothing(data.sensitivity_maps)
    kept = simulate_acquisition(img, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
    @test size(kept.sensitivity_maps) == (n, n, nc)
    @test kept.sensitivity_maps ≈ Ristretto._resample_sensitivity_maps(maps_fine, 2, (n, n))
    @test kept.kspace_data == data.kspace_data
end

@testitem "simulate_acquisition: radial data from a finer phantom" tags = [:simulation, :nfft] setup = [FinerGridSetup] begin
    using Ristretto, NamedDims
    n = 64
    traj = Float64.(radial_trajectory(2n, 32))
    acq = AcquisitionInfo(; trajectory = traj, image_size = (n, n))
    coarse = simulate_acquisition(blob(n, n), acq; inverse_crime_check = false).kspace_data
    for fine in (101, 128)
        finer = simulate_acquisition(blob(fine, fine), acq; inverse_crime_check = false).kspace_data
        @test size(finer) == size(coarse)
        # The two differ by the accuracy of the NFFT, not by the grid.
        @test rel(finer, coarse) < 1.0e-3
    end
end

@testitem "simulate_acquisition: inverse-crime checks and size errors" tags = [:simulation] setup = [FinerGridSetup] begin
    using Ristretto, NamedDims
    n = 32
    acq = AcquisitionInfo(is3D = false, image_size = (n, n))
    @test_logs (:warn, r"inverse crime") simulate_acquisition(blob(n, n), acq)
    @test_logs (:warn, r"integer multiple") simulate_acquisition(blob(2n, 2n), acq)
    @test_logs simulate_acquisition(blob(51, 51), acq)
    @test_logs simulate_acquisition(blob(n, n), acq; inverse_crime_check = false)
    @test_throws ArgumentError simulate_acquisition(blob(n - 1, n), acq; inverse_crime_check = false)
    maps = ComplexF64.(coil_sensitivities(n, n, 2))
    acq_maps = AcquisitionInfo(is3D = false, image_size = (n, n), sensitivity_maps = maps)
    @test_throws ArgumentError simulate_acquisition(blob(51, 51), acq_maps)
    @test simulate_acquisition(blob(n, n), acq_maps; inverse_crime_check = false, keep_sensitivity_maps = true).sensitivity_maps == maps
end
