using TestItems

@testitem "Encoding operators" tags = [:encoding, :operators, :nfft] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using FFTW
    using NamedDims

@testset "Operators" begin
    @testset "Fourier Operator" begin

        @testset "simple 2D" for threaded in (true, false), fast_planning in (true, false)
            ksp = rand(ComplexF32, 64, 64)
            wrapped_ksp = NamedDimsArray{(:kx, :ky)}(ksp)
        
            ℱ = get_fourier_operator(ksp, false; threaded, fast_planning)
            img = ℱ' * ksp
            @test img ≈ ifft(fftshift(ksp))

            ℱ = get_fourier_operator(wrapped_ksp; threaded, fast_planning)
            @test ℱ isa MriReconstructionToolbox.NamedDimsOp
            img = ℱ' * wrapped_ksp
            @test unname(img) ≈ ifft(fftshift(ksp))
            @test dimnames(img) == (:x, :y)
        end

        @testset "simple 3D" for threaded in (true, false), fast_planning in (true, false)
            ksp = rand(ComplexF32, 64, 64, 64)
            wrapped_ksp = NamedDimsArray{(:kx, :ky, :kz)}(ksp)
            
            ℱ = get_fourier_operator(ksp, true; threaded, fast_planning)
            img = ℱ' * ksp
            @test img ≈ ifft(fftshift(ksp))
            
            ℱ = get_fourier_operator(wrapped_ksp; threaded, fast_planning)
            @test ℱ isa MriReconstructionToolbox.NamedDimsOp
            img = ℱ' * wrapped_ksp
            @test unname(img) ≈ ifft(fftshift(ksp))
            @test dimnames(img) == (:x, :y, :z)
        end

        @testset "multiplanar" for threaded in (true, false), fast_planning in (true, false)
            ksp = rand(ComplexF32, 64, 64, 8)
            wrapped_ksp = NamedDimsArray{(:kx, :ky, :z)}(ksp)
            
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

    @testset "AcquisitionInfo API" begin
        @testset "2D Array fully-sampled" for threaded in (true, false), fast_planning in (true, false)
            ksp = rand(ComplexF32, 32, 32)
            info = MriReconstructionToolbox.AcquisitionInfo(ksp; is3D=false)
            ℱ = get_fourier_operator(info; threaded, fast_planning)
            img = ℱ' * ksp
            @test img ≈ ifft(fftshift(ksp))
            𝒜 = get_encoding_operator(info; threaded, fast_planning)
            @test 𝒜' * ksp ≈ ifft(fftshift(ksp))
            @test 𝒜 * img ≈ ksp
        end

        @testset "2D NamedDims with smaps" for threaded in (true, false), fast_planning in (true, false)
            ksp = rand(ComplexF32, 32, 32, 4);
            smaps = rand(ComplexF32, 32, 32, 4);
            wrapped_ksp = NamedDimsArray{(:kx, :ky, :coil)}(ksp);
            wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps);
            info = MriReconstructionToolbox.AcquisitionInfo(wrapped_ksp; sensitivity_maps=wrapped_smaps)
            𝒜 = get_encoding_operator(info; threaded, fast_planning)
            img = 𝒜' * wrapped_ksp
            @test unname(img) ≈ unname(dropdims(sum(conj.(wrapped_smaps) .* ifft(fftshift(ksp, (1,2)), (1, 2)), dims=:coil), dims=:coil))
            @test dimnames(img) == (:x, :y)
            ksp2 = 𝒜 * img
            @test unname(ksp2) ≈ unname(fftshift(fft(reshape(img, 32, 32, 1) .* wrapped_smaps, (1, 2)), (1,2)))
            @test dimnames(ksp2) == (:kx, :ky, :coil)
        end

        @testset "2D Array subsampled mask" for threaded in (true, false), fast_planning in (true, false)
            full_ksp = rand(ComplexF32, 32, 32);
            mask = rand(Bool, 32, 32);
            subs_ksp = full_ksp[mask];
            info = MriReconstructionToolbox.AcquisitionInfo(subs_ksp; image_size=(32, 32), subsampling=mask)
            𝒜 = get_encoding_operator(info; threaded, fast_planning)
            img = 𝒜' * subs_ksp;
            masked_ksp = similar(full_ksp);
            masked_ksp .= 0;
            masked_ksp[mask] .= subs_ksp;
            @test img ≈ ifft(fftshift(masked_ksp))
            @test 𝒜 * img ≈ fftshift(fft(img))[mask]
        end
    end

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
            @test img3 ≈ dropdims(sum(conj.(smaps) .* img2, dims=3), dims=3)
            
            𝒮 = get_sensitivity_map_operator(wrapped_smaps; threaded)
            @test 𝒮 isa MriReconstructionToolbox.NamedDimsOp
            img2 = 𝒮 * img
            @test unname(img2) ≈ unname(reshape(img, 64, 64, 1) .* wrapped_smaps)
            @test dimnames(img2) == (:x, :y, :coil)
            img3 = 𝒮' * img2
            @test unname(img3) ≈ unname(dropdims(sum(conj.(wrapped_smaps) .* img2, dims=:coil), dims=:coil))
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
            @test img3 ≈ dropdims(sum(conj.(smaps) .* img2, dims=4), dims=4)
            
            𝒮 = get_sensitivity_map_operator(wrapped_smaps; threaded)
            @test 𝒮 isa MriReconstructionToolbox.NamedDimsOp
            img2 = 𝒮 * img
            @test unname(img2) ≈ unname(reshape(img, 64, 64, 64, 1) .* wrapped_smaps)
            @test dimnames(img2) == (:x, :y, :z, :coil)
            img3 = 𝒮' * img2
            @test unname(img3) ≈ unname(dropdims(sum(conj.(wrapped_smaps) .* img2, dims=:coil), dims=:coil))
            @test dimnames(img3) == (:x, :y, :z)
        end
    end

    @testset "Full Encoding Operator" begin
        @testset "2D Cartesian" begin
            ksp = rand(ComplexF32, 64, 64);
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

            wrapped_ksp = NamedDimsArray{(:kx, :ky, :batch)}(ksp);
            ℱ = get_encoding_operator(wrapped_ksp)
            img2 = ℱ' * wrapped_ksp
            @test unname(img2) ≈ ifft(fftshift(ksp, (1,2)), (1, 2))
            @test dimnames(img2) == (:x, :y, :batch)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(wrapped_ksp)
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :batch)
        end

        @testset "2D Cartesian PI" begin
            ksp = rand(ComplexF32, 64, 64, 8)
            smaps = rand(ComplexF32, 64, 64, 8)
            ℱ = get_encoding_operator(ksp, false; sensitivity_maps=smaps)
            img = ℱ' * ksp
            @test img ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims=3), dims=3)
            ksp2 = ℱ * img
            @test ksp2 ≈ fftshift(fft(reshape(img, 64, 64, 1) .* smaps, (1, 2)), (1, 2))

            wrapped_ksp = NamedDimsArray{(:kx, :ky, :coil)}(ksp)
            wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
            ℱ = get_encoding_operator(wrapped_ksp; sensitivity_maps=wrapped_smaps)
            img2 = ℱ' * wrapped_ksp
            @test unname(img2) ≈ unname(dropdims(sum(conj.(wrapped_smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims=:coil), dims=:coil))
            @test dimnames(img2) == (:x, :y)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(fftshift(fft(reshape(img2, 64, 64, 1) .* wrapped_smaps, (1, 2)), (1, 2)))
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :coil)
        end

        @testset "Multiplanar Cartesian PI" begin
            ksp = rand(ComplexF32, 64, 64, 8, 10)
            smaps = rand(ComplexF32, 64, 64, 8)
            ℱ = get_encoding_operator(ksp, false; sensitivity_maps=smaps)
            img = ℱ' * ksp
            @test img ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims=3), dims=3)
            ksp2 = ℱ * img
            @test ksp2 ≈ fftshift(fft(reshape(img, 64, 64, 1, 10) .* smaps, (1, 2)), (1, 2))

            wrapped_ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(ksp)
            wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
            ℱ = get_encoding_operator(wrapped_ksp; sensitivity_maps=wrapped_smaps)
            img2 = ℱ' * wrapped_ksp
            @test unname(img2) ≈ unname(dropdims(sum(conj.(wrapped_smaps) .* ifft(fftshift(ksp, (1, 2)), (1, 2)), dims=:coil), dims=:coil))
            @test dimnames(img2) == (:x, :y, :z)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(fftshift(fft(reshape(img2, 64, 64, 1, 10) .* wrapped_smaps, (1, 2)), (1, 2)))
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :coil, :z)
        end

        @testset "3D Cartesian PI" begin
            ksp = rand(ComplexF32, 64, 64, 64, 8)
            smaps = rand(ComplexF32, 64, 64, 64, 8)
            ℱ = get_encoding_operator(ksp, true; sensitivity_maps=smaps)
            img = ℱ' * ksp
            @test img ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2, 3)), (1, 2, 3)), dims=4), dims=4)
            ksp2 = ℱ * img
            @test ksp2 ≈ fftshift(fft(reshape(img, 64, 64, 64, 1) .* smaps, (1, 2, 3)), (1, 2, 3))

            wrapped_ksp = NamedDimsArray{(:kx, :ky, :kz, :coil)}(ksp)
            wrapped_smaps = NamedDimsArray{(:x, :y, :z, :coil)}(smaps)
            ℱ = get_encoding_operator(wrapped_ksp; sensitivity_maps=wrapped_smaps)
            img2 = ℱ' * wrapped_ksp
            @test unname(img2) ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(ksp, (1, 2, 3)), (1, 2, 3)), dims=4), dims=4)
            @test dimnames(img2) == (:x, :y, :z)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(fftshift(fft(reshape(img2, 64, 64, 64, 1) .* smaps, (1, 2, 3)), (1, 2, 3)))
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :kz, :coil)
        end

        @testset "2D Cartesian undersampling with mask" begin
            ksp = rand(ComplexF32, 64, 64)
            mask = rand(Bool, 64, 64)
            ksp_subsampled = ksp[mask]
            ℱ = get_encoding_operator(ksp_subsampled, false; subsampling=mask)
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64)
            temp[mask] .= ksp_subsampled
            @test img ≈ ifft(fftshift(temp, (1, 2)), (1, 2))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp_subsampled

            wrapped_ksp_subsampled = NamedDimsArray{(:kxy,)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; subsampling=mask)
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
            ℱ = get_encoding_operator(ksp_subsampled, false; image_size=(64, 64), subsampling=(:, idx))
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64)
            temp[:, idx] .= ksp_subsampled
            @test img ≈ ifft(fftshift(temp, (1, 2)), (1, 2))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp_subsampled

            wrapped_ksp_subsampled = NamedDimsArray{(:kx, :ky)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; image_size=(64, 64), subsampling=(:, idx))
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            temp[:, idx] .= wrapped_ksp_subsampled
            @test unname(img2) ≈ ifft(fftshift(temp, (1, 2)), (1, 2))
            @test dimnames(img2) == (:x, :y)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(wrapped_ksp_subsampled)
            @test dimnames(wrapped_ksp2) == (:kx, :ky,)
        end

        @testset "2D Cartesian PI undersampling with linear indices + 1D-mask" begin
            ksp = rand(ComplexF32, 64, 64, 8)
            smaps = rand(ComplexF32, 64, 64, 8)
            idx = 1:5:64
            mask = rand(Bool, 64)
            ksp_subsampled = fftshift(ksp, (1, 2))[idx, mask, :]
            ℱ = get_encoding_operator(ksp_subsampled, false; image_size=(64, 64), sensitivity_maps=smaps, subsampling=(idx, mask))
            img = ℱ' * ksp_subsampled
            temp = zero(ksp)
            temp[idx, mask, :] .= ksp_subsampled
            @test img ≈ dropdims(sum(conj.(smaps) .* ifft(fftshift(temp, (1, 2)), (1, 2)), dims=3), dims=3)
            ksp2 = ℱ * img
            @test ksp2 ≈ fftshift(fft(reshape(img, 64, 64, 1) .* smaps, (1, 2)), (1, 2))[idx, mask, :]

            wrapped_ksp_subsampled = NamedDimsArray{(:kx, :ky, :coil)}(ksp_subsampled)
            wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; image_size=(64, 64), sensitivity_maps=wrapped_smaps, subsampling=(idx, mask))
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            temp[idx, mask, :] .= wrapped_ksp_subsampled
            @test unname(img2) ≈ dropdims(sum(conj.(wrapped_smaps) .* ifft(fftshift(temp, (1, 2)), (1, 2)), dims=:coil), dims=:coil)
            @test dimnames(img2) == (:x, :y)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(fftshift(fft(reshape(img2, 64, 64, 1) .* wrapped_smaps, (1, 2)), (1, 2))[idx, mask, :])
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :coil)
        end

        @testset "3D Cartesian undersampling with mask" begin
            ksp = rand(ComplexF32, 64, 64, 64)
            mask = rand(Bool, 64, 64, 64)
            ksp_subsampled = fftshift(ksp, (1, 2, 3))[mask]
            ℱ = get_encoding_operator(ksp_subsampled, true; subsampling=(mask,))
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64, 64)
            temp[mask] .= ksp_subsampled
            @test img ≈ ifft(fftshift(temp, (1, 2, 3)), (1, 2, 3))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp_subsampled

            wrapped_ksp_subsampled = NamedDimsArray{(:kxyz,)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; subsampling=(mask,))
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
            idx = CartesianIndices((64, 64))[1:5:(64*64)]
            ksp_subsampled = fftshift(ksp, (1, 2, 3))[:, idx]
            ℱ = get_encoding_operator(ksp_subsampled, true; image_size=(64, 64, 64), subsampling=(:, idx))
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64, 64)
            temp[:, idx] .= ksp_subsampled
            @test img ≈ ifft(fftshift(temp, (1, 2, 3)), (1, 2, 3))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp_subsampled

            wrapped_ksp_subsampled = NamedDimsArray{(:kx, :kyz)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; image_size=(64, 64, 64), subsampling=(:, idx))
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            temp[:, idx] .= wrapped_ksp_subsampled
            @test unname(img2) ≈ ifft(fftshift(temp, (1, 2, 3)), (1, 2, 3))
            @test dimnames(img2) == (:x, :y, :z)
            wrapped_ksp2 = ℱ * img2
            @test unname(wrapped_ksp2) ≈ unname(wrapped_ksp_subsampled)
            @test dimnames(wrapped_ksp2) == (:kx, :kyz,)
        end

        #=@testset "Multiplanar Cartesian with different mask for each slice" begin
            ksp = rand(ComplexF32, 64, 64, 8, 10);
            masks = [];
            ksp_subsampled = zeros(ComplexF32, 100, 8, 10);
            for i in 1:10
                mask = zeros(Bool, 64, 64);
                while sum(mask) != 100
                    mask[rand(1:64), rand(1:64)] = true;
                end
                push!(masks, mask);
                ksp_subsampled[:, :, i] .= ksp[mask, :, i];
            end
            ℱ = get_encoding_operator(ksp_subsampled, false; subsampling=masks)
            img = ℱ' * ksp_subsampled
            temp = zeros(ComplexF32, 64, 64, 8, 10)
            for i in 1:10
                temp[masks[i], :, i] .= ksp_subsampled[:, :, i]
            end
            @test img ≈ bfft(temp, (1, 2))
            ksp2 = ℱ * img
            @test ksp2 ≈ ksp_subsampled .* (64 * 64)

            wrapped_ksp_subsampled = NamedDimsArray{(:undersampled_kxy, :coil, :z)}(ksp_subsampled)
            ℱ = get_encoding_operator(wrapped_ksp_subsampled; subsampling=masks)
            img2 = ℱ' * wrapped_ksp_subsampled
            temp .= 0
            for i in 1:10
                temp[masks[i], :, i] .= wrapped_ksp_subsampled[:, :, i]
            end
            @test img2 ≈ bfft(temp, (1, 2))
            @test dimnames(img2) == (:x, :y, :z)
            wrapped_ksp2 = ℱ * img2
            @test wrapped_ksp2 ≈ wrapped_ksp_subsampled .* (64 * 64)
            @test dimnames(wrapped_ksp2) == (:kx, :ky, :coil, :z)
        end=#

        @testset "2D non-Cartesian" begin
            img = rand(ComplexF32, 8, 8)
            trajectory = rand(Float32, 2, 16, 3) .- 0.5f0
            ksp = rand(ComplexF32, 16, 3)
            info = AcquisitionInfo(ksp; trajectory, image_size=(8, 8))
            @test info isa NonCartesianAcquisitionInfo

            𝒩 = get_encoding_operator(info; threaded=false)
            raw = MriReconstructionToolbox.NFFTOp((8, 8), trajectory; threaded=false)

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
            info = AcquisitionInfo(ksp; trajectory, dcf, image_size=(8, 8))

            𝒩 = get_encoding_operator(info; threaded=false)
            raw = MriReconstructionToolbox.NFFTOp((8, 8), trajectory, dcf; threaded=false)

            @test 𝒩 * img ≈ raw * img
            @test 𝒩' * ksp ≈ raw' * ksp
        end

        @testset "2D non-Cartesian PI" begin
            img = rand(ComplexF32, 8, 8)
            trajectory = rand(Float32, 2, 12, 2) .- 0.5f0
            smaps = rand(ComplexF32, 8, 8, 3)
            ksp = zeros(ComplexF32, 12, 2, 3)
            info = AcquisitionInfo(ksp; trajectory, image_size=(8, 8), sensitivity_maps=smaps)

            𝒜 = get_encoding_operator(info; threaded=false)
            raw = MriReconstructionToolbox.NFFTOp((8, 8), trajectory; threaded=false)
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
            info = AcquisitionInfo(ksp; trajectory, image_size=(8, 8))

            𝒩 = get_encoding_operator(info; threaded=false)
            @test 𝒩 isa MriReconstructionToolbox.NamedDimsOp

            ksp2 = 𝒩 * img
            img2 = 𝒩' * ksp2

            @test dimnames(ksp2) == (:sample, :shot)
            @test dimnames(img2) == (:x, :y)
        end
    end
end
end
