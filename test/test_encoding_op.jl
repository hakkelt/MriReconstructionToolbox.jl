@testitem "Fourier Operator" tags = [:encoding, :operators, :fourier] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator
    using FFTW
    using NamedDims

    @testset "Fourier Operator" begin
        ksp = rand(ComplexF32, 64, 64)
        wrapped_ksp = NamedDimsArray{(:kx, :ky)}(ksp)
        @testset "simple 2D" for threaded in (true, false), fast_planning in (true, false)
            ℱ = get_fourier_operator(ksp, false; threaded, fast_planning)
            img = ℱ' * ksp
            @test img ≈ ifft(fftshift(ksp))

            ℱ = get_fourier_operator(wrapped_ksp; threaded, fast_planning)
            @test ℱ isa MriReconstructionToolbox.NamedDimsOp
            img = ℱ' * wrapped_ksp
            @test unname(img) ≈ ifft(fftshift(ksp))
            @test dimnames(img) == (:x, :y)
        end

        ksp = rand(ComplexF32, 64, 64, 64)
        wrapped_ksp = NamedDimsArray{(:kx, :ky, :kz)}(ksp)
        @testset "simple 3D" for threaded in (true, false), fast_planning in (true, false)
            ℱ = get_fourier_operator(ksp, true; threaded, fast_planning)
            img = ℱ' * ksp
            @test img ≈ ifft(fftshift(ksp))

            ℱ = get_fourier_operator(wrapped_ksp; threaded, fast_planning)
            @test ℱ isa MriReconstructionToolbox.NamedDimsOp
            img = ℱ' * wrapped_ksp
            @test unname(img) ≈ ifft(fftshift(ksp))
            @test dimnames(img) == (:x, :y, :z)
        end

        ksp = rand(ComplexF32, 64, 64, 8)
        wrapped_ksp = NamedDimsArray{(:kx, :ky, :z)}(ksp)
        @testset "multiplanar" for threaded in (true, false), fast_planning in (true, false)
            ℱ = get_fourier_operator(ksp, false; threaded, fast_planning)
            img = ℱ' * ksp
            @test img ≈ ifft(fftshift(ksp, (1, 2)), (1, 2))

            ℱ = get_fourier_operator(wrapped_ksp; threaded, fast_planning)
            @test ℱ isa MriReconstructionToolbox.NamedDimsOp
            img = ℱ' * wrapped_ksp
            @test unname(img) ≈ ifft(fftshift(ksp, (1, 2)), (1, 2))
            @test dimnames(img) == (:x, :y, :z)
        end
    end
end

@testitem "AcquisitionInfo API" tags = [:encoding, :operators, :acquisition_info] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator
    using FFTW
    using NamedDims

    @testset "AcquisitionInfo API" begin
        ksp = rand(ComplexF32, 32, 32)
        info = MriReconstructionToolbox.AcquisitionInfo(ksp; is3D = false)
        @testset "2D Array fully-sampled" for threaded in (true, false), fast_planning in (true, false)
            ℱ = get_fourier_operator(info; threaded, fast_planning)
            img = ℱ' * ksp
            @test img ≈ ifft(fftshift(ksp))
            𝒜 = get_encoding_operator(info; threaded, fast_planning)
            @test 𝒜' * ksp ≈ ifft(fftshift(ksp))
            @test 𝒜 * img ≈ ksp
        end

        ksp = rand(ComplexF32, 32, 32, 4)
        smaps = rand(ComplexF32, 32, 32, 4)
        wrapped_ksp = NamedDimsArray{(:kx, :ky, :coil)}(ksp)
        wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
        info = MriReconstructionToolbox.AcquisitionInfo(wrapped_ksp; sensitivity_maps = wrapped_smaps)
        @testset "2D NamedDims with smaps" for threaded in (true, false), fast_planning in (true, false)
            𝒜 = get_encoding_operator(info; threaded, fast_planning)
            img = 𝒜' * wrapped_ksp
            @test unname(img) ≈ unname(dropdims(sum(conj.(wrapped_smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims = :coil), dims = :coil))
            @test dimnames(img) == (:x, :y)
            ksp2 = 𝒜 * img
            @test unname(ksp2) ≈ unname(fftshift(fft(reshape(img, 32, 32, 1) .* wrapped_smaps, (1, 2)), (1, 2)))
            @test dimnames(ksp2) == (:kx, :ky, :coil)
        end

        full_ksp = rand(ComplexF32, 32, 32)
        mask = rand(Bool, 32, 32)
        subs_ksp = full_ksp[mask]
        info = MriReconstructionToolbox.AcquisitionInfo(subs_ksp; image_size = (32, 32), subsampling = mask)
        @testset "2D Array subsampled mask" for threaded in (true, false), fast_planning in (true, false)
            𝒜 = get_encoding_operator(info; threaded, fast_planning)
            img = 𝒜' * subs_ksp
            masked_ksp = similar(full_ksp)
            masked_ksp .= 0
            masked_ksp[mask] .= subs_ksp
            @test img ≈ ifft(fftshift(masked_ksp))
            @test 𝒜 * img ≈ fftshift(fft(img))[mask]
        end
    end
end

@testitem "NFFT operating point (S6)" tags = [:encoding, :operators, :nfft] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator
    using NFFTOperators: NFFTOp
    import NFFTOperators
    using LinearAlgebra, Random

    Random.seed!(0)
    image_size = (32, 32)
    n_samples = 300
    traj = rand(2, n_samples) .- 0.5
    ksp = rand(ComplexF64, n_samples)
    x = rand(ComplexF64, image_size)

    @testset "defaults are unchanged" begin
        op_no_kwargs = get_fourier_operator(ksp, image_size, traj)
        op_explicit_nothing = get_fourier_operator(
            ksp, image_size, traj; m = nothing, sigma = nothing, precompute = nothing
        )
        @test op_no_kwargs * x ≈ op_explicit_nothing * x
    end

    @testset "operating point is actually forwarded" begin
        op_default = get_fourier_operator(ksp, image_size, traj)
        # MRIReco's operating point (`TODO.md` §8): far cheaper, deliberately less accurate.
        op_low_acc = get_fourier_operator(
            ksp, image_size, traj; m = 3, sigma = 1.25, precompute = NFFTOperators.NFFT.TENSOR
        )
        @test op_default.plan.params.m != op_low_acc.plan.params.m
        @test op_default.plan.params.σ != op_low_acc.plan.params.σ

        # Not identical (different gridding kernel/oversampling), but not a different
        # transform either -- both approximate the same NDFT, so a coarser grid still lands
        # close to the fine one on a small, well-conditioned trajectory like this one.
        y_default = op_default * x
        y_low_acc = op_low_acc * x
        @test y_default ≈ y_low_acc rtol = 1.0e-2
    end

    @testset "get_encoding_operator forwards the same keywords" begin
        info = NonCartesianAcquisitionInfo(
            ksp; trajectory = traj, image_size,
        )
        𝒜_default = get_encoding_operator(info)
        𝒜_low_acc = get_encoding_operator(info; m = 3, sigma = 1.25, precompute = NFFTOperators.NFFT.TENSOR)
        @test 𝒜_default.plan.params.m != 𝒜_low_acc.plan.params.m
        @test (𝒜_default * x) ≈ (𝒜_low_acc * x) rtol = 1.0e-2
    end
end

@testitem "Sensitivity Map Operator" tags = [:encoding, :operators, :sensitivity_maps] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator
    using AbstractOperators
    using NamedDims

    @testset "Sensitivity Map Operator" begin
        @testset "2D sensitivity maps" for threaded in (true, false)
            img = rand(ComplexF32, 64, 64)
            smaps = rand(ComplexF32, 64, 64, 8)
            wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)

            𝒮 = get_sensitivity_map_operator(smaps, false; threaded)
            @test 𝒮 isa Compose
            img2 = 𝒮 * img
            @test img2 ≈ reshape(img, 64, 64, 1) .* smaps
            img3 = 𝒮' * img2
            @test img3 ≈ dropdims(sum(conj.(smaps) .* img2, dims = 3), dims = 3)

            𝒮 = get_sensitivity_map_operator(wrapped_smaps; threaded)
            @test 𝒮 isa MriReconstructionToolbox.NamedDimsOp
            img2 = 𝒮 * img
            @test unname(img2) ≈ unname(reshape(img, 64, 64, 1) .* wrapped_smaps)
            @test dimnames(img2) == (:x, :y, :coil)
            img3 = 𝒮' * img2
            @test unname(img3) ≈ unname(dropdims(sum(conj.(wrapped_smaps) .* img2, dims = :coil), dims = :coil))
            @test dimnames(img3) == (:x, :y)
        end

        @testset "3D sensitivity maps" for threaded in (true, false)
            img = rand(ComplexF32, 64, 64, 64)
            smaps = rand(ComplexF32, 64, 64, 64, 8)
            wrapped_smaps = NamedDimsArray{(:x, :y, :z, :coil)}(smaps)

            𝒮 = get_sensitivity_map_operator(smaps, true; threaded)
            @test 𝒮 isa Compose
            img2 = 𝒮 * img
            @test img2 ≈ reshape(img, 64, 64, 64, 1) .* smaps
            img3 = 𝒮' * img2
            @test img3 ≈ dropdims(sum(conj.(smaps) .* img2, dims = 4), dims = 4)

            𝒮 = get_sensitivity_map_operator(wrapped_smaps; threaded)
            @test 𝒮 isa MriReconstructionToolbox.NamedDimsOp
            img2 = 𝒮 * img
            @test unname(img2) ≈ unname(reshape(img, 64, 64, 64, 1) .* wrapped_smaps)
            @test dimnames(img2) == (:x, :y, :z, :coil)
            img3 = 𝒮' * img2
            @test unname(img3) ≈ unname(dropdims(sum(conj.(wrapped_smaps) .* img2, dims = :coil), dims = :coil))
            @test dimnames(img3) == (:x, :y, :z)
        end
    end
end

@testitem "Full Encoding Operator" tags = [:encoding, :operators, :nfft] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator
    using FFTW
    using NamedDims

    @testset "Full Encoding Operator" begin
        @testset "2D Cartesian" begin
            ksp = rand(ComplexF32, 64, 64)
            ℱ = get_encoding_operator(ksp, false)
            img = ℱ' * ksp
            @test img ≈ ifft(fftshift(ksp))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp

            wrapped_ksp = NamedDimsArray{(:kx, :ky)}(ksp)
            ℱ = get_encoding_operator(wrapped_ksp)
            img2 = ℱ' * wrapped_ksp
            @test unname(img2) ≈ ifft(fftshift(ksp))
            @test dimnames(img2) == (:x, :y)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(wrapped_ksp)
            @test dimnames(wrapped_ksp2) == (:kx, :ky)
        end

        @testset "2D Cartesian batched" begin
            ksp = rand(ComplexF32, 64, 64, 10)
            ℱ = get_encoding_operator(ksp, false)
            img = ℱ' * ksp
            @test img ≈ ifft(fftshift(ksp, (1, 2)), (1, 2))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp

            wrapped_ksp = NamedDimsArray{(:kx, :ky, :batch)}(ksp)
            ℱ = get_encoding_operator(wrapped_ksp)
            img2 = ℱ' * wrapped_ksp
            @test unname(img2) ≈ ifft(fftshift(ksp, (1, 2)), (1, 2))
            @test dimnames(img2) == (:x, :y, :batch)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(wrapped_ksp)
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :batch)
        end

        @testset "2D Cartesian PI" begin
            ksp = rand(ComplexF32, 64, 64, 8)
            smaps = rand(ComplexF32, 64, 64, 8)
            ℱ = get_encoding_operator(ksp, false; sensitivity_maps = smaps)
            img = ℱ' * ksp
            @test img ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims = 3), dims = 3)
            ksp2 = ℱ * img
            @test ksp2 ≈ fftshift(fft(reshape(img, 64, 64, 1) .* smaps, (1, 2)), (1, 2))

            wrapped_ksp = NamedDimsArray{(:kx, :ky, :coil)}(ksp)
            wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
            ℱ = get_encoding_operator(wrapped_ksp; sensitivity_maps = wrapped_smaps)
            img2 = ℱ' * wrapped_ksp
            @test unname(img2) ≈ unname(dropdims(sum(conj.(wrapped_smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims = :coil), dims = :coil))
            @test dimnames(img2) == (:x, :y)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(fftshift(fft(reshape(img2, 64, 64, 1) .* wrapped_smaps, (1, 2)), (1, 2)))
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :coil)
        end

        @testset "Multiplanar Cartesian PI" begin
            ksp = rand(ComplexF32, 64, 64, 8, 10)
            smaps = rand(ComplexF32, 64, 64, 8)
            ℱ = get_encoding_operator(ksp, false; sensitivity_maps = smaps)
            img = ℱ' * ksp
            @test img ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims = 3), dims = 3)
            ksp2 = ℱ * img
            @test ksp2 ≈ fftshift(fft(reshape(img, 64, 64, 1, 10) .* smaps, (1, 2)), (1, 2))

            wrapped_ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(ksp)
            @test dimnames(wrapped_ksp) == (:kx, :ky, :coil, :z)
        end

        @testset "Dynamic 2D Cartesian PI" begin
            # Regression: composing the sensitivity operator (which has no :coil axis on the image
            # side) against the wrong batch-dim offset lost the trailing batch dimension entirely,
            # throwing a `DomainError` from `ℱ * 𝒮`.
            ksp = rand(ComplexF32, 64, 64, 8, 10)
            smaps = rand(ComplexF32, 64, 64, 8)
            wrapped_ksp = NamedDimsArray{(:kx, :ky, :coil, :t)}(ksp)
            wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
            ℱ = get_encoding_operator(wrapped_ksp; sensitivity_maps = wrapped_smaps)
            img = ℱ' * wrapped_ksp
            @test unname(img) ≈
                dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims = 3), dims = 3)
            @test dimnames(img) == (:x, :y, :t)
            ksp2 = ℱ * img
            @test unname(ksp2) ≈
                fftshift(fft(reshape(unname(img), 64, 64, 1, 10) .* smaps, (1, 2)), (1, 2))
            @test dimnames(ksp2) == (:kx, :ky, :coil, :t)
        end

        @testset "Dynamic 2D Cartesian PI (plain arrays)" begin
            # Same off-by-one as above, on the un-named branch of `_compose_with_sensitivity`.
            ksp = rand(ComplexF32, 64, 64, 8, 10)
            smaps = rand(ComplexF32, 64, 64, 8)
            ℱ = get_encoding_operator(ksp, false; sensitivity_maps = smaps)
            @test size(ℱ, 2) == (64, 64, 10)
            img = ℱ' * ksp
            @test size(img) == (64, 64, 10)
            @test img ≈
                dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims = 3), dims = 3)
            ksp2 = ℱ * img
            @test ksp2 ≈ fftshift(fft(reshape(img, 64, 64, 1, 10) .* smaps, (1, 2)), (1, 2))
        end

        @testset "Dynamic 2D Cartesian PI with a shared 3D map" begin
            # A single 3D sensitivity map (:x, :y, :coil, :z) shared across time frames of a
            # multi-slice + time acquisition: :z is a batch dim of the map itself, on top of the
            # acquisition's own :t batch dim.
            ksp = rand(ComplexF32, 64, 64, 8, 5, 10)
            smaps = rand(ComplexF32, 64, 64, 8, 5)
            wrapped_ksp = NamedDimsArray{(:kx, :ky, :coil, :z, :t)}(ksp)
            wrapped_smaps = NamedDimsArray{(:x, :y, :coil, :z)}(smaps)
            ℱ = get_encoding_operator(wrapped_ksp; sensitivity_maps = wrapped_smaps)
            img = ℱ' * wrapped_ksp
            @test dimnames(img) == (:x, :y, :z, :t)
            @test size(img) == (64, 64, 5, 10)
            ksp2 = ℱ * img
            @test dimnames(ksp2) == (:kx, :ky, :coil, :z, :t)
            @test size(ksp2) == size(ksp)
            # Slice-by-slice, this must agree with the plain 2D-multislice (no time) case.
            for t in 1:10
                ℱ_slice = get_encoding_operator(
                    NamedDimsArray{(:kx, :ky, :coil, :z)}(ksp[:, :, :, :, t]);
                    sensitivity_maps = wrapped_smaps,
                )
                @test unname(ℱ_slice' * ksp[:, :, :, :, t]) ≈ unname(img[:, :, :, t])
            end
        end

        @testset "3D Cartesian PI" begin
            ksp = rand(ComplexF32, 64, 64, 64, 8)
            smaps = rand(ComplexF32, 64, 64, 64, 8)
            ℱ = get_encoding_operator(ksp, true; sensitivity_maps = smaps)
            img = ℱ' * ksp
            @test img ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2, 3)), (1, 2, 3)), dims = 4), dims = 4)
            ksp2 = ℱ * img
            @test ksp2 ≈ fftshift(fft(reshape(img, 64, 64, 64, 1) .* smaps, (1, 2, 3)), (1, 2, 3))

            wrapped_ksp = NamedDimsArray{(:kx, :ky, :kz, :coil)}(ksp)
            wrapped_smaps = NamedDimsArray{(:x, :y, :z, :coil)}(smaps)
            ℱ = get_encoding_operator(wrapped_ksp; sensitivity_maps = wrapped_smaps)
            img2 = ℱ' * wrapped_ksp
            @test unname(img2) ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2, 3)), (1, 2, 3)), dims = 4), dims = 4)
            @test dimnames(img2) == (:x, :y, :z)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(fftshift(fft(reshape(img2, 64, 64, 64, 1) .* smaps, (1, 2, 3)), (1, 2, 3)))
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :kz, :coil)
        end

        @testset "2D Cartesian undersampling with mask" begin
            ksp = rand(ComplexF32, 64, 64)
            mask = rand(Bool, 64, 64)
            ksp_subsampled = ksp[mask]
            ℱ = get_encoding_operator(ksp_subsampled, false; subsampling = mask)
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64)
            temp[mask] .= ksp_subsampled
            @test img ≈ ifft(fftshift(temp, (1, 2)), (1, 2))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp_subsampled

            wrapped_ksp_subsampled = NamedDimsArray{(:kxy,)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; subsampling = mask)
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            temp[mask] .= wrapped_ksp_subsampled
            @test unname(img2) ≈ ifft(fftshift(temp, (1, 2)), (1, 2))
            @test dimnames(img2) == (:x, :y)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(wrapped_ksp_subsampled)
            @test dimnames(wrapped_ksp2) == (:kxy,)
        end

        @testset "2D Cartesian undersampling with colon + linear indices" begin
            ksp = rand(ComplexF32, 64, 64)
            idx = 1:5:64
            ksp_subsampled = fftshift(ksp, (1, 2))[:, idx]
            ℱ = get_encoding_operator(ksp_subsampled, false; image_size = (64, 64), subsampling = (:, idx))
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64)
            temp[:, idx] .= ksp_subsampled
            @test img ≈ ifft(fftshift(temp, (1, 2)), (1, 2))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp_subsampled

            wrapped_ksp_subsampled = NamedDimsArray{(:kx, :ky)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; image_size = (64, 64), subsampling = (:, idx))
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            temp[:, idx] .= wrapped_ksp_subsampled
            @test unname(img2) ≈ ifft(fftshift(temp, (1, 2)), (1, 2))
            @test dimnames(img2) == (:x, :y)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(wrapped_ksp_subsampled)
            @test dimnames(wrapped_ksp2) == (:kx, :ky)
        end

        @testset "2D Cartesian PI undersampling with linear indices + 1D-mask" begin
            ksp = rand(ComplexF32, 64, 64, 8)
            smaps = rand(ComplexF32, 64, 64, 8)
            idx = 1:5:64
            mask = rand(Bool, 64)
            ksp_subsampled = fftshift(ksp, (1, 2))[idx, mask, :]
            ℱ = get_encoding_operator(ksp_subsampled, false; image_size = (64, 64), sensitivity_maps = smaps, subsampling = (idx, mask))
            img = ℱ' * ksp_subsampled
            temp = zero(ksp)
            temp[idx, mask, :] .= ksp_subsampled
            @test img ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(temp, (1, 2)), (1, 2)), dims = 3), dims = 3)
            ksp2 = ℱ * img
            @test ksp2 ≈ fftshift(fft(reshape(img, 64, 64, 1) .* smaps, (1, 2)), (1, 2))[idx, mask, :]

            wrapped_ksp_subsampled = NamedDimsArray{(:kx, :ky, :coil)}(ksp_subsampled)
            wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; image_size = (64, 64), sensitivity_maps = wrapped_smaps, subsampling = (idx, mask))
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            temp[idx, mask, :] .= wrapped_ksp_subsampled
            @test unname(img2) ≈ dropdims(sum(conj.(wrapped_smaps) .* ifft(fftshift(temp, (1, 2)), (1, 2)), dims = :coil), dims = :coil)
            @test dimnames(img2) == (:x, :y)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(fftshift(fft(reshape(img2, 64, 64, 1) .* wrapped_smaps, (1, 2)), (1, 2))[idx, mask, :])
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :coil)
        end

        @testset "3D Cartesian undersampling with mask" begin
            ksp = rand(ComplexF32, 64, 64, 64)
            mask = rand(Bool, 64, 64, 64)
            ksp_subsampled = fftshift(ksp, (1, 2, 3))[mask]
            ℱ = get_encoding_operator(ksp_subsampled, true; subsampling = (mask,))
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64, 64)
            temp[mask] .= ksp_subsampled
            @test img ≈ ifft(fftshift(temp, (1, 2, 3)), (1, 2, 3))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp_subsampled

            wrapped_ksp_subsampled = NamedDimsArray{(:kxyz,)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; subsampling = (mask,))
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            temp[mask] .= wrapped_ksp_subsampled
            @test unname(img2) ≈ ifft(fftshift(temp, (1, 2, 3)), (1, 2, 3))
            @test dimnames(img2) == (:x, :y, :z)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ wrapped_ksp_subsampled
            @test dimnames(wrapped_ksp2) == (:kxyz,)
        end

        @testset "3D Cartesian undersampling with colon + Cartesian indices" begin
            ksp = rand(ComplexF32, 64, 64, 64)
            idx = CartesianIndices((64, 64))[1:5:(64 * 64)]
            ksp_subsampled = fftshift(ksp, (1, 2, 3))[:, idx]
            ℱ = get_encoding_operator(ksp_subsampled, true; image_size = (64, 64, 64), subsampling = (:, idx))
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64, 64)
            temp[:, idx] .= ksp_subsampled
            @test img ≈ ifft(fftshift(temp, (1, 2, 3)), (1, 2, 3))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp_subsampled

            wrapped_ksp_subsampled = NamedDimsArray{(:kx, :kyz)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; image_size = (64, 64, 64), subsampling = (:, idx))
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            temp[:, idx] .= wrapped_ksp_subsampled
            @test unname(img2) ≈ ifft(fftshift(temp, (1, 2, 3)), (1, 2, 3))
            @test dimnames(img2) == (:x, :y, :z)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(wrapped_ksp_subsampled)
            @test dimnames(wrapped_ksp2) == (:kx, :kyz)
        end

        @testset "Multiplanar Cartesian with different mask for each slice" begin
            ksp = rand(ComplexF32, 64, 64, 8, 10)
            masks = map(1:10) do _
                mask = zeros(Bool, 64, 64)
                while count(mask) != 100
                    mask[rand(1:64), rand(1:64)] = true
                end
                mask
            end
            ksp_subsampled = zeros(ComplexF32, 100, 8, 10)
            for slice in axes(ksp_subsampled, 3)
                ksp_subsampled[:, :, slice] .= ksp[masks[slice], :, slice]
            end

            ℱ = get_encoding_operator(ksp_subsampled, false; subsampling = masks)
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64, 8, 10)
            for slice in axes(temp, 4)
                temp[masks[slice], :, slice] .= ksp_subsampled[:, :, slice]
            end
            @test img ≈ ifft(fftshift(temp, (1, 2)), (1, 2))
            @test ℱ * img ≈ ksp_subsampled

            wrapped_ksp_subsampled = NamedDimsArray{(:kxy, :coil, :z)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; subsampling = masks)
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            for slice in axes(temp, 4)
                temp[masks[slice], :, slice] .= wrapped_ksp_subsampled[:, :, slice]
            end
            @test unname(img2) ≈ ifft(fftshift(temp, (1, 2)), (1, 2))
            @test dimnames(img2) == (:x, :y, :coil, :z)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(wrapped_ksp_subsampled)
            @test dimnames(wrapped_ksp2) == (:kxy, :coil, :z)
        end

        @testset "2D non-Cartesian" begin
            img = rand(ComplexF32, 8, 8)
            trajectory = rand(Float32, 2, 16, 3) .- 0.5f0
            ksp = rand(ComplexF32, 16, 3)
            info = AcquisitionInfo(ksp; trajectory, image_size = (8, 8))
            @test info isa NonCartesianAcquisitionInfo

            𝒩 = get_encoding_operator(info; threaded = false)
            raw = MriReconstructionToolbox.NFFTOp((8, 8), trajectory; threaded = false)

            @test size(𝒩, 1) == size(ksp)
            @test size(𝒩, 2) == size(img)
            @test 𝒩 * img ≈ raw * img
            @test 𝒩' * ksp ≈ raw' * ksp
        end

        @testset "2D non-Cartesian with explicit DCF" begin
            img = rand(ComplexF32, 8, 8)
            trajectory = rand(Float32, 2, 10, 4) .- 0.5f0
            dcf = rand(Float32, 10, 4)
            ksp = rand(ComplexF32, 10, 4)
            info = AcquisitionInfo(ksp; trajectory, dcf, image_size = (8, 8))

            𝒩 = get_encoding_operator(info; threaded = false)
            raw = MriReconstructionToolbox.NFFTOp((8, 8), trajectory, dcf; threaded = false)

            @test 𝒩 * img ≈ raw * img
            @test 𝒩' * ksp ≈ raw' * ksp
        end

        @testset "2D non-Cartesian PI" begin
            img = rand(ComplexF32, 8, 8)
            trajectory = rand(Float32, 2, 12, 2) .- 0.5f0
            smaps = rand(ComplexF32, 8, 8, 3)
            ksp = zeros(ComplexF32, 12, 2, 3)
            info = AcquisitionInfo(ksp; trajectory, image_size = (8, 8), sensitivity_maps = smaps)

            𝒜 = get_encoding_operator(info; threaded = false)
            raw = MriReconstructionToolbox.NFFTOp((8, 8), trajectory; threaded = false)
            coil_imgs = reshape(img, 8, 8, 1) .* smaps
            expected = similar(ksp)
            for coil in axes(expected, 3)
                expected[:, :, coil] .= raw * view(coil_imgs, :, :, coil)
            end

            @test 𝒜 * img ≈ expected
            @test size(𝒜' * expected) == size(img)
        end

        @testset "2D non-Cartesian NamedDims" begin
            img = NamedDimsArray{(:x, :y)}(rand(ComplexF32, 8, 8))
            trajectory = NamedDimsArray{(:coord, :sample, :shot)}(rand(Float32, 2, 9, 5) .- 0.5f0)
            ksp = NamedDimsArray{(:sample, :shot)}(rand(ComplexF32, 9, 5))
            info = AcquisitionInfo(ksp; trajectory, image_size = (8, 8))

            𝒩 = get_encoding_operator(info; threaded = false)
            @test 𝒩 isa MriReconstructionToolbox.NamedDimsOp

            ksp2 = 𝒩 * img
            img2 = 𝒩' * ksp2

            @test dimnames(ksp2) == (:sample, :shot)
            @test dimnames(img2) == (:x, :y)
        end
    end
end

@testitem "simulate_acquisition NamedDims and real image" tags = [:encoding, :simulation] begin
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator
    using NamedDims

    @testset "NamedDims 2D image with smaps" begin
        nx, ny, nc = 16, 16, 2
        img = NamedDimsArray{(:x, :y)}(rand(ComplexF32, nx, ny))
        smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(nx, ny, nc))
        acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
        result = simulate_acquisition(img, acq)
        @test result.kspace_data isa NamedDimsArray
        @test dimnames(result.kspace_data) == (:kx, :ky, :coil)
    end

    @testset "NamedDims 2D image no smaps" begin
        nx, ny = 16, 16
        img = NamedDimsArray{(:x, :y)}(rand(ComplexF32, nx, ny))
        acq = AcquisitionInfo(is3D = false, image_size = (nx, ny))
        result = simulate_acquisition(img, acq)
        @test result.kspace_data isa NamedDimsArray
        @test dimnames(result.kspace_data) == (:kx, :ky)
    end

    @testset "NamedDims 3D image with smaps" begin
        nx, ny, nz, nc = 8, 8, 4, 2
        img = NamedDimsArray{(:x, :y, :z)}(rand(ComplexF32, nx, ny, nz))
        smaps = NamedDimsArray{(:x, :y, :z, :coil)}(coil_sensitivities(nx, ny, nz, nc))
        acq = AcquisitionInfo(is3D = true, sensitivity_maps = smaps)
        result = simulate_acquisition(img, acq)
        @test result.kspace_data isa NamedDimsArray
        @test dimnames(result.kspace_data) == (:kx, :ky, :kz, :coil)
    end

    @testset "NamedDims 3D image no smaps" begin
        nx, ny, nz = 8, 8, 4
        img = NamedDimsArray{(:x, :y, :z)}(rand(ComplexF32, nx, ny, nz))
        acq = AcquisitionInfo(is3D = true, image_size = (nx, ny, nz))
        result = simulate_acquisition(img, acq)
        @test result.kspace_data isa NamedDimsArray
        @test dimnames(result.kspace_data) == (:kx, :ky, :kz)
    end

    @testset "Real (non-complex) image input" begin
        nx, ny, nc = 16, 16, 2
        img = rand(Float32, nx, ny)
        smaps = ComplexF32.(coil_sensitivities(nx, ny, nc))
        acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
        result = simulate_acquisition(img, acq)
        @test eltype(result.kspace_data) == ComplexF32
    end

    @testset "NamedDims 2D with subsampling" begin
        nx, ny, nc = 16, 16, 2
        img = NamedDimsArray{(:x, :y)}(rand(ComplexF32, nx, ny))
        mask = rand(Bool, nx, ny)
        smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(nx, ny, nc))
        acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = (mask,))
        result = simulate_acquisition(img, acq)
        @test result.kspace_data isa NamedDimsArray
    end
end

@testitem "simulate_acquisition for NonCartesianAcquisitionInfo" tags = [:encoding, :simulation, :nfft] begin
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using NamedDims
    using LinearAlgebra
    using Random

    Random.seed!(0)
    nx, ny, nc, nspokes, nread = 16, 16, 3, 8, 16
    x = rand(ComplexF32, nx, ny)
    angles = range(0, pi; length = nspokes + 1)[1:(end - 1)]
    r = range(-0.5f0, 0.5f0; length = nread)
    traj = zeros(Float32, 2, nread, nspokes)
    for (s, ang) in enumerate(angles), (i, rr) in enumerate(r)
        traj[1, i, s] = rr * cos(ang)
        traj[2, i, s] = rr * sin(ang)
    end
    smaps = coil_sensitivities(nx, ny, nc)

    @testset "matches building the encoding operator by hand" begin
        acq = NonCartesianAcquisitionInfo(nothing; trajectory = traj, image_size = (nx, ny), sensitivity_maps = smaps)
        result = simulate_acquisition(x, acq)
        @test size(result.kspace_data) == (nread, nspokes, nc)

        placeholder = zeros(ComplexF32, nread, nspokes, nc)
        acq_manual = NonCartesianAcquisitionInfo(placeholder; trajectory = traj, image_size = (nx, ny), sensitivity_maps = smaps)
        y_manual = get_encoding_operator(acq_manual) * x
        @test result.kspace_data == y_manual
    end

    @testset "no sensitivity maps" begin
        acq = NonCartesianAcquisitionInfo(nothing; trajectory = traj, image_size = (nx, ny))
        result = simulate_acquisition(x, acq)
        @test size(result.kspace_data) == (nread, nspokes)
    end

    @testset "real image input is cast to complex" begin
        acq = NonCartesianAcquisitionInfo(nothing; trajectory = traj, image_size = (nx, ny))
        result = simulate_acquisition(real.(x), acq)
        @test eltype(result.kspace_data) == ComplexF32
    end
end

@testitem "Fourier operator helpers match raw FFT (even and odd sizes)" tags = [:reconstruction, :encoding] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator
    using NamedDims
    using FFTW
    const MRT = MriReconstructionToolbox

    for (Nx, Ny, Nc) in ((32, 32, 3), (31, 33, 3))
        k = randn(ComplexF64, Nx, Ny, Nc)
        x = randn(ComplexF64, Nx, Ny, Nc)
        even = iseven(Nx) && iseven(Ny)

        # _cartesian_fourier_op must be a consistent forward/adjoint pair, and on even sizes match
        # the pre-refactor `_direct_fft` / `_direct_ifft` bit for bit.
        acq = CartesianAcquisitionInfo(
            NamedDimsArray{(:kx, :ky, :coil)}(k); is3D = false, image_size = (Nx, Ny),
        )
        ℱ = MRT._cartesian_fourier_op(acq, k)
        @test ℱ' * (ℱ * x) ≈ x
        if even
            @test ℱ' * k ≈ ifft(ifftshift(k, (1, 2)), (1, 2))
            @test ℱ * x ≈ fftshift(fft(x, (1, 2)), (1, 2))
        end

        # _axis_dft_op: readout-only, k-space-side shift — exact on even and odd
        ro = MRT._axis_dft_op(k, (1,); kspace_shift = true)
        @test ro' * k ≈ ifft(ifftshift(k, 1), 1)
        @test ro * x ≈ fftshift(fft(x, 1), 1)

        # _axis_dft_op: no shift — exact on even and odd
        plain = MRT._axis_dft_op(x, (1, 2))
        @test plain * x ≈ fft(x, (1, 2))
        @test plain' * k ≈ ifft(k, (1, 2))
    end
end
