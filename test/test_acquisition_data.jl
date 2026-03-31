using TestItems

@testitem "NonCartesianAcquisitionInfo" tags = [:acquisition, :nfft] begin
    using MriReconstructionToolbox
    using NamedDims

    @testset "Basic 2D construction" begin
        trajectory = randn(Float32, 2, 64)
        ksp = randn(ComplexF32, 64)
        info = AcquisitionInfo(ksp; trajectory, image_size=(8, 8))
        @test info isa NonCartesianAcquisitionInfo
        @test info.is3D == false
        @test info.image_size == (8, 8)
        @test info.trajectory === trajectory
        @test isnothing(info.dcf)
        @test isnothing(info.sensitivity_maps)
    end

    @testset "3D construction" begin
        trajectory = randn(Float32, 3, 128)
        ksp = randn(ComplexF32, 128)
        info = AcquisitionInfo(ksp; trajectory, image_size=(8, 8, 8))
        @test info isa NonCartesianAcquisitionInfo
        @test info.is3D == true
        @test info.image_size == (8, 8, 8)
    end

    @testset "With DCF" begin
        trajectory = randn(Float32, 2, 64)
        dcf = rand(Float32, 64)
        ksp = randn(ComplexF32, 64)
        info = AcquisitionInfo(ksp; trajectory, dcf, image_size=(8, 8))
        @test info.dcf === dcf
    end

    @testset "With sensitivity maps" begin
        trajectory = randn(Float32, 2, 64)
        smaps = randn(ComplexF32, 8, 8, 4)
        ksp_coil = NamedDimsArray(randn(ComplexF32, 64, 4), (:sample, :coil))
        traj_named = NamedDimsArray(trajectory, (:coord, :sample))
        info = AcquisitionInfo(ksp_coil; trajectory=traj_named, sensitivity_maps=smaps, image_size=(8, 8))
        @test info.sensitivity_maps === smaps
    end

    @testset "NamedDims trajectory" begin
        trajectory = NamedDimsArray(randn(Float32, 2, 64), (:dim, :sample))
        ksp = NamedDimsArray(randn(ComplexF32, 64), (:sample,))
        info = AcquisitionInfo(ksp; trajectory, image_size=(8, 8))
        @test info isa NonCartesianAcquisitionInfo
    end

    @testset "Validation errors" begin
        # Bad trajectory ndims
        @test_throws ArgumentError AcquisitionInfo(nothing; trajectory=randn(Float32, 3), image_size=(8, 8))
        # Wrong coordinate dimension
        @test_throws ArgumentError AcquisitionInfo(nothing; trajectory=randn(Float32, 4, 64), image_size=(8, 8, 8))
        # Mismatched image_size
        @test_throws ArgumentError AcquisitionInfo(nothing; trajectory=randn(Float32, 2, 64), image_size=(8, 8, 8))
        # DCF type mismatch
        @test_throws ArgumentError AcquisitionInfo(
            randn(ComplexF32, 64);
            trajectory=randn(Float32, 2, 64),
            dcf=rand(Float64, 64),
            image_size=(8, 8),
        )
    end

    @testset "Copy constructor" begin
        trajectory = randn(Float32, 2, 64)
        ksp = randn(ComplexF32, 64)
        info = AcquisitionInfo(ksp; trajectory, image_size=(8, 8))

        new_ksp = randn(ComplexF32, 64)
        info2 = AcquisitionInfo(info; kspace_data=new_ksp)
        @test info2.kspace_data === new_ksp
        @test info2.trajectory === trajectory
        @test info2 isa NonCartesianAcquisitionInfo
    end
end

@testitem "CartesianAcquisitionInfo copy constructor" tags = [:acquisition] begin
    using MriReconstructionToolbox

    ksp = randn(ComplexF32, 8, 8)
    info = AcquisitionInfo(ksp; image_size=(8, 8))
    @test info isa CartesianAcquisitionInfo

    new_ksp = randn(ComplexF32, 8, 8)
    info2 = AcquisitionInfo(info; kspace_data=new_ksp)
    @test info2.kspace_data === new_ksp
    @test info2.image_size == (8, 8)
    @test info2 isa CartesianAcquisitionInfo
end

@testitem "Dimension utilities" tags = [:acquisition] begin
    using MriReconstructionToolbox
    using NamedDims
    import MriReconstructionToolbox: get_image_size, get_time_dim,
        get_fourier_kspace_dims, get_fourier_image_dims,
        get_nonfourier_image_dims, get_nonfourier_kspace_dims, get_image_dims

    @testset "get_image_size - 2D Cartesian" begin
        ksp = randn(ComplexF32, 8, 8)
        info = AcquisitionInfo(ksp; image_size=(8, 8))
        @test get_image_size(info) == (8, 8)
    end

    @testset "get_image_size - 2D Cartesian with batch dims" begin
        ksp = randn(ComplexF32, 8, 8, 5)
        info = AcquisitionInfo(ksp; image_size=(8, 8))
        @test get_image_size(info) == (8, 8, 5)
    end

    @testset "get_image_size - 2D+coil with batch dims" begin
        smaps = randn(ComplexF32, 8, 8, 4)
        ksp = randn(ComplexF32, 8, 8, 4, 5)
        info = AcquisitionInfo(ksp; image_size=(8, 8), sensitivity_maps=smaps)
        @test get_image_size(info) == (8, 8, 5)
    end

    @testset "get_time_dim" begin
        @test get_time_dim(3, 1:4) == 3
        @test get_time_dim(:time, (:x, :y, :time)) == 3
        @test_throws ArgumentError get_time_dim(nothing, 1:4)  # no NamedDims, no time_dim
        @test_throws ArgumentError get_time_dim(nothing, (:x, :y, :z))  # no :time dimension
    end

    @testset "get_fourier_kspace_dims - 2D" begin
        ksp = randn(ComplexF32, 8, 8)
        info = AcquisitionInfo(ksp; image_size=(8, 8))
        @test get_fourier_kspace_dims(info) == 1:2
    end

    @testset "get_fourier_kspace_dims - 3D" begin
        ksp = randn(ComplexF32, 8, 8, 8)
        info = AcquisitionInfo(ksp; image_size=(8, 8, 8), is3D=true)
        @test get_fourier_kspace_dims(info) == 1:3
    end

    @testset "get_fourier_kspace_dims - NamedDims" begin
        ksp = NamedDimsArray(randn(ComplexF32, 8, 8), (:kx, :ky))
        info = AcquisitionInfo(ksp; image_size=(8, 8))
        @test get_fourier_kspace_dims(info) == (:kx, :ky)
    end

    @testset "get_fourier_image_dims" begin
        ksp = randn(ComplexF32, 8, 8)
        info = AcquisitionInfo(ksp; image_size=(8, 8))
        @test get_fourier_image_dims(info) == 1:2

        ksp3d = randn(ComplexF32, 8, 8, 8)
        info3d = AcquisitionInfo(ksp3d; image_size=(8, 8, 8), is3D=true)
        @test get_fourier_image_dims(info3d) == 1:3
    end

    @testset "get_nonfourier_kspace_dims" begin
        # 2D with batch dimension
        ksp = randn(ComplexF32, 8, 8, 5)
        info = AcquisitionInfo(ksp; image_size=(8, 8))
        @test get_nonfourier_kspace_dims(info) == 3:3

        # 2D with coils and batch dimension
        smaps = randn(ComplexF32, 8, 8, 4)
        ksp_coil = randn(ComplexF32, 8, 8, 4, 5)
        info_coil = AcquisitionInfo(ksp_coil; image_size=(8, 8), sensitivity_maps=smaps)
        @test get_nonfourier_kspace_dims(info_coil) == 4:4
    end

    @testset "get_image_dims" begin
        ksp = randn(ComplexF32, 8, 8, 5)
        info = AcquisitionInfo(ksp; image_size=(8, 8))
        @test get_image_dims(info) == 1:3
    end

    @testset "Non-Cartesian dimensions" begin
        traj = randn(Float32, 2, 64)
        ksp = randn(ComplexF32, 64, 5)
        info = AcquisitionInfo(ksp; trajectory=traj, image_size=(8, 8))
        @test get_fourier_kspace_dims(info) == 1:1  # 1 sample dim for non-cartesian
        @test get_image_size(info) == (8, 8, 5)
    end
end

@testitem "Sampling patterns" tags = [:simulation] begin
    using MriReconstructionToolbox

    @testset "PoissonDiskSampling" begin
        pattern = PoissonDiskSampling(4.0)
        # PoissonDisk requires 3D dims with subsample_freq_encoding=false → returns (:, 2D mask)
        result = create_sampling_pattern(pattern, (64, 64, 64))
        @test result isa Tuple
        @test result[1] === Colon()
        mask = result[2]
        @test size(mask) == (64, 64)
        @test eltype(mask) == Bool
        @test sum(mask) < prod(size(mask))
        @test sum(mask) > 0
    end

    @testset "PoissonDiskSampling 2D with freq encoding" begin
        pattern = PoissonDiskSampling(4.0)
        # PoissonDisk with 2D + subsample_freq_encoding=true → returns plain 2D mask
        mask = create_sampling_pattern(pattern, (32, 32); subsample_freq_encoding=true)
        @test size(mask) == (32, 32)
        @test eltype(mask) == Bool
        @test sum(mask) > 0
    end

    @testset "VariableDensitySampling with Gaussian" begin
        pattern = VariableDensitySampling(GaussianDistribution(), 4.0)
        # 2D, subsample_freq_encoding=false → returns (:, 1D mask)
        result = create_sampling_pattern(pattern, (64, 64))
        @test result isa Tuple
        mask = result[2]
        @test size(mask) == (64,)
        @test eltype(mask) == Bool
        @test sum(mask) < length(mask)
    end

    @testset "VariableDensitySampling with Polynomial" begin
        pattern = VariableDensitySampling(PolynomialDistribution(), 4.0)
        result = create_sampling_pattern(pattern, (64, 64))
        @test result isa Tuple
        mask = result[2]
        @test size(mask) == (64,)
        @test eltype(mask) == Bool
    end

    @testset "to_displayable_mask" begin
        # to_displayable_mask expects (:, vector) or (:, array) or plain array
        inner_mask = BitVector(rand(Bool, 8))
        pattern = (:, inner_mask)
        displayable = to_displayable_mask(pattern, (8, 8))
        @test size(displayable) == (8, 8)
        @test eltype(displayable) == Bool
    end
end
