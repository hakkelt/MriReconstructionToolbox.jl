using MriReconstructionToolbox
using GeometricMedicalPhantoms
using Statistics
using FFTW
using Test
using PyCall
using MRIReco
using LinearAlgebra
using Random
using BartIO

include("../src/ComparisonHarness.jl")
using .ComparisonHarness

sigpy = pyimport("sigpy")
sp_mri = pyimport("sigpy.mri")

function compute_nrmse(x::AbstractArray, xref::AbstractArray)
    return norm(x[:] .- xref[:]) / norm(xref[:])
end

@testset "Reconstruction Benchmark" begin
    # Base setup
    N = 128
    Nc = 8

    # Generate test phantoms and sensitivity maps
    img_1c, kspace_1c, smaps_1c_raw = ComparisonHarness.generate_multicoil_brain(N=N, num_coils=1)
    img_mc, kspace_mc, cmap = ComparisonHarness.generate_multicoil_brain(N=N, num_coils=Nc)

    kdata_1c = NamedDimsArray(kspace_1c, (:kx, :ky, :coil))
    smaps_1c = NamedDimsArray(smaps_1c_raw, (:x, :y, :coil))
    acq_1c = CartesianAcquisitionInfo(kdata_1c; is3D=false, sensitivity_maps=smaps_1c, shifted_image_dims=(:x, :y))

    kdata_mc = NamedDimsArray(kspace_mc, (:kx, :ky, :coil))
    smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))
    acq_mc = CartesianAcquisitionInfo(kdata_mc; is3D=false, sensitivity_maps=smaps_mc, shifted_image_dims=(:x, :y))

    @testset "Base Reconstruction (Unregularized)" begin
        @testset "Cartesian Adjoint" begin
            # 1-Coil fully-sampled
            E_1c = MriReconstructionToolbox.get_encoding_operator(acq_1c)
            mrt_adj_1c = E_1c' * kdata_1c

            # SigPy Adjoint (1-coil)
            kdata_sp_1c = parent(permutedims(kspace_1c, (3, 2, 1)))
            smaps_sp_1c = parent(permutedims(smaps_1c, (3, 2, 1)))
            S_sp_1c = sp_mri.linop.Sense(smaps_sp_1c, ishape=(N, N))
            sp_adj_1c = permutedims(S_sp_1c.H(kdata_sp_1c), (2, 1))

            scale_1c = norm(abs.(sp_adj_1c)) / norm(abs.(mrt_adj_1c))
            nrmse_1c_sp = compute_nrmse(mrt_adj_1c .* scale_1c, sp_adj_1c)
            @info "MRT vs SigPy Cartesian 1C Adjoint NRMSE: $nrmse_1c_sp"
            @test nrmse_1c_sp < 1e-4

            # Compare MRT and SigPy to Ground Truth (normalized by |s|^2)
            recon_1c_mrt = mrt_adj_1c ./ abs2.(smaps_1c)
            nrmse_1c_mrt_gt = compute_nrmse(recon_1c_mrt .* (norm(abs.(img_1c)) / norm(abs.(recon_1c_mrt))), img_1c)
            @info "MRT vs Ground Truth Cartesian 1C Adjoint NRMSE: $nrmse_1c_mrt_gt"
            @test nrmse_1c_mrt_gt < 1e-4

            recon_1c_sp = sp_adj_1c ./ parent(abs2.(smaps_1c))
            nrmse_1c_sp_gt = compute_nrmse(recon_1c_sp .* (norm(abs.(img_1c)) / norm(abs.(recon_1c_sp))), img_1c)
            @info "SigPy vs Ground Truth Cartesian 1C Adjoint NRMSE: $nrmse_1c_sp_gt"
            @test nrmse_1c_sp_gt < 1e-4

            # Multi-coil fully-sampled MRT Adjoint
            E_mc = MriReconstructionToolbox.get_encoding_operator(acq_mc)
            mrt_adj_mc = E_mc' * kdata_mc

            # SigPy Adjoint (multi-coil)
            kdata_sp_mc = parent(permutedims(kspace_mc, (3, 2, 1)))
            smaps_sp_mc = parent(permutedims(cmap, (3, 2, 1)))
            S_sp_mc = sp_mri.linop.Sense(smaps_sp_mc, ishape=(N, N))
            sp_adj_mc = permutedims(S_sp_mc.H(kdata_sp_mc), (2, 1))

            scale_mc = norm(abs.(sp_adj_mc)) / norm(abs.(mrt_adj_mc))
            nrmse_mc_sp = compute_nrmse(mrt_adj_mc .* scale_mc, sp_adj_mc)
            @info "MRT vs SigPy Cartesian MC Adjoint NRMSE: $nrmse_mc_sp"
            @test nrmse_mc_sp < 1e-4

            # MRIReco Adjoint (multi-coil fully sampled)
            kdata_mr = reshape(kspace_mc, N, N, 1, Nc, 1, 1)
            acq_mr = AcquisitionData(kdata_mr)
            smaps_mr = reshape(cmap, N, N, 1, Nc)
            recoParams = Dict{Symbol, Any}(:reco => "direct", :reconSize => (N, N), :senseMaps => smaps_mr)
            img_mr_direct = MRIReco.reconstruction(acq_mr, recoParams)[:, :, 1, 1, :]
            mr_adj_mc = sum(img_mr_direct .* conj.(cmap), dims=3)[:, :, 1]
            scale_mr_mc = norm(abs.(mr_adj_mc)) / norm(abs.(mrt_adj_mc))
            nrmse_mc_mr = compute_nrmse(mrt_adj_mc .* scale_mr_mc, mr_adj_mc)
            @info "MRT vs MRIReco Cartesian MC Adjoint NRMSE: $nrmse_mc_mr"
            @test nrmse_mc_mr < 1e-4

            # BART Adjoint (multi-coil fully sampled via IFFT)
            kdata_bart_cart = reshape(kspace_mc, N, N, 1, Nc)
            bart_ifft = ComparisonHarness.run_bart(1, "fft -i 3", ComplexF32.(kdata_bart_cart))
            bart_adj_mc = sum(bart_ifft[:, :, 1, :] .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims=3)[:, :, 1]
            scale_bart_mc = norm(abs.(bart_adj_mc)) / norm(abs.(mrt_adj_mc))
            nrmse_mc_bart = compute_nrmse(mrt_adj_mc .* scale_bart_mc, bart_adj_mc)
            @info "MRT vs BART Cartesian MC Adjoint NRMSE: $nrmse_mc_bart"
            @test nrmse_mc_bart < 1e-4

            # Compare all multi-coil fully sampled adjoints to Ground Truth (normalized by sum|s|^2)
            recon_mc_mrt = mrt_adj_mc ./ sum(abs2.(smaps_mc), dims=3)[:, :, 1]
            nrmse_mc_mrt_gt = compute_nrmse(recon_mc_mrt .* (norm(abs.(img_mc)) / norm(abs.(recon_mc_mrt))), img_mc)
            @info "MRT vs Ground Truth Cartesian MC Adjoint NRMSE: $nrmse_mc_mrt_gt"
            @test nrmse_mc_mrt_gt < 1e-4

            recon_mc_sp = sp_adj_mc ./ sum(abs2.(cmap), dims=3)[:, :, 1]
            nrmse_mc_sp_gt = compute_nrmse(recon_mc_sp .* (norm(abs.(img_mc)) / norm(abs.(recon_mc_sp))), img_mc)
            @info "SigPy vs Ground Truth Cartesian MC Adjoint NRMSE: $nrmse_mc_sp_gt"
            @test nrmse_mc_sp_gt < 1e-4

            recon_mc_mr = mr_adj_mc ./ sum(abs2.(cmap), dims=3)[:, :, 1]
            nrmse_mc_mr_gt = compute_nrmse(recon_mc_mr .* (norm(abs.(img_mc)) / norm(abs.(recon_mc_mr))), img_mc)
            @info "MRIReco vs Ground Truth Cartesian MC Adjoint NRMSE: $nrmse_mc_mr_gt"
            @test nrmse_mc_mr_gt < 1e-4

            recon_mc_bart = bart_adj_mc ./ sum(abs2.(ComplexF32.(cmap)), dims=3)[:, :, 1]
            nrmse_mc_bart_gt = compute_nrmse(recon_mc_bart .* (norm(abs.(img_mc)) / norm(abs.(recon_mc_bart))), img_mc)
            @info "BART vs Ground Truth Cartesian MC Adjoint NRMSE: $nrmse_mc_bart_gt"
            @test nrmse_mc_bart_gt < 1e-4

            # Multi-coil undersampled MRT Adjoint
            mask = rand(MersenneTwister(42), Bool, N, N)
            kdata_us = NamedDimsArray(kspace_mc[mask, :], (:kxy, :coil))
            acq_us = CartesianAcquisitionInfo(kdata_us; is3D=false, image_size=(N, N), sensitivity_maps=smaps_mc, shifted_image_dims=(:x, :y), subsampling=mask)
            E_us = MriReconstructionToolbox.get_encoding_operator(acq_us)
            mrt_adj_us = E_us' * kdata_us

            # SigPy Adjoint (undersampled)
            kspace_sp_us = copy(kspace_mc)
            kspace_sp_us[.!mask, :] .= 0
            kdata_sp_us = parent(permutedims(kspace_sp_us, (3, 2, 1)))
            sp_adj_us = permutedims(S_sp_mc.H(kdata_sp_us), (2, 1))

            scale_us = norm(abs.(sp_adj_us)) / norm(abs.(mrt_adj_us))
            nrmse_us_sp = compute_nrmse(mrt_adj_us .* scale_us, sp_adj_us)
            @info "MRT vs SigPy Cartesian US Adjoint NRMSE: $nrmse_us_sp"
            @test nrmse_us_sp < 1e-4

            # MRIReco Adjoint (undersampled)
            kdata_mr_us = reshape(kspace_sp_us, N, N, 1, Nc, 1, 1)
            acq_mr_us = AcquisitionData(kdata_mr_us)
            img_mr_us_direct = MRIReco.reconstruction(acq_mr_us, recoParams)[:, :, 1, 1, :]
            mr_adj_us = sum(img_mr_us_direct .* conj.(cmap), dims=3)[:, :, 1]
            scale_mr_us = norm(abs.(mr_adj_us)) / norm(abs.(mrt_adj_us))
            nrmse_us_mr = compute_nrmse(mrt_adj_us .* scale_mr_us, mr_adj_us)
            @info "MRT vs MRIReco Cartesian US Adjoint NRMSE: $nrmse_us_mr"
            @test nrmse_us_mr < 1e-4
        end

        @testset "Non-Cartesian Adjoint" begin
            # Setup radial trajectory
            t = RadialTrajectory(Float32, N, N; TE=0.0f0, AQ=1.0f-3)
            traj_named = NamedDimsArray(t.nodes, (:dim, :k))
            smaps_nc = NamedDimsArray(ComplexF32.(cmap), (:x, :y, :coil))

            # Simulate Non-Cartesian multi-coil data
            kdata_nc_zeros = NamedDimsArray(zeros(ComplexF32, 16384, Nc), (:k, :coil))
            acq_nc_sim = NonCartesianAcquisitionInfo(kdata_nc_zeros; trajectory=traj_named, image_size=(N, N), sensitivity_maps=smaps_nc, shifted_image_dims=(:x, :y))
            E_nc_sim = MriReconstructionToolbox.get_encoding_operator(acq_nc_sim)
            kdata_nc_sim = E_nc_sim * NamedDimsArray(ComplexF32.(img_mc), (:x, :y))

            # 1) Density-compensated Adjoint (Gridding / Direct Reconstruction)
            acq_nc_dcf = NonCartesianAcquisitionInfo(kdata_nc_sim; trajectory=traj_named, image_size=(N, N), sensitivity_maps=smaps_nc, shifted_image_dims=(:x, :y))
            E_nc_dcf = MriReconstructionToolbox.get_encoding_operator(acq_nc_dcf)
            mrt_adj_dcf = E_nc_dcf' * kdata_nc_sim

            # Compare Density-Compensated MRT and MRIReco Adjoint to Ground Truth Phantom
            scale_nc_gt = norm(abs.(img_mc)) / norm(abs.(mrt_adj_dcf))
            nrmse_nc_mrt_gt = compute_nrmse(mrt_adj_dcf .* scale_nc_gt, img_mc)
            @info "MRT vs Ground Truth Non-Cartesian DCF Adjoint NRMSE: $nrmse_nc_mrt_gt"
            @test nrmse_nc_mrt_gt < 0.10

            # MRIReco Non-Cartesian Direct Reconstruction
            kdata_mr_nc = reshape(kdata_nc_sim, 16384, Nc, 1, 1)
            acq_mr_nc = AcquisitionData(t, fill(kdata_mr_nc[:, :, 1, 1], 1, 1, 1))
            smaps_mr_f32 = reshape(ComplexF32.(cmap), N, N, 1, Nc)
            recoParams_nc = Dict{Symbol, Any}(:reco => "direct", :reconSize => (N, N), :senseMaps => smaps_mr_f32)
            img_mr_nc_direct = MRIReco.reconstruction(acq_mr_nc, recoParams_nc)[:, :, 1, 1, :]
            mr_adj_nc = sum(img_mr_nc_direct .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims=3)[:, :, 1]
            scale_mr_nc = norm(abs.(mr_adj_nc)) / norm(abs.(mrt_adj_dcf))
            nrmse_nc_mr = compute_nrmse(mrt_adj_dcf .* scale_mr_nc, mr_adj_nc)
            @info "MRT vs MRIReco Non-Cartesian DCF Adjoint NRMSE: $nrmse_nc_mr"
            @test nrmse_nc_mr < 0.05

            nrmse_nc_mr_gt = compute_nrmse(mr_adj_nc .* (norm(abs.(img_mc)) / norm(abs.(mr_adj_nc))), img_mc)
            @info "MRIReco vs Ground Truth Non-Cartesian DCF Adjoint NRMSE: $nrmse_nc_mr_gt"
            @test nrmse_nc_mr_gt < 0.10

            # 2) Raw Non-Cartesian Adjoint (without DCF)
            acq_nc_nodcf = NonCartesianAcquisitionInfo(kdata_nc_sim; trajectory=traj_named, dcf=ones(Float32, 16384), image_size=(N, N), sensitivity_maps=smaps_nc, shifted_image_dims=(:x, :y))
            E_nc_nodcf = MriReconstructionToolbox.get_encoding_operator(acq_nc_nodcf)
            mrt_adj_nodcf = E_nc_nodcf' * kdata_nc_sim

            # SigPy Raw Non-Cartesian Adjoint
            coord_sp_yx = Float32.(reverse(parent(t.nodes), dims=1)') .* Float32(N)
            kdata_sp_nc = parent(permutedims(kdata_nc_sim, (2, 1)))
            smaps_sp_mc_f32 = ComplexF32.(parent(permutedims(cmap, (3, 2, 1))))
            S_sp_nc = sp_mri.linop.Sense(smaps_sp_mc_f32, coord=coord_sp_yx, ishape=(N, N))
            sp_adj_nodcf = permutedims(S_sp_nc.H(kdata_sp_nc), (2, 1))

            scale_sp_nc = norm(abs.(sp_adj_nodcf)) / norm(abs.(mrt_adj_nodcf))
            nrmse_nc_sp = compute_nrmse(mrt_adj_nodcf .* scale_sp_nc, sp_adj_nodcf)
            @info "MRT vs SigPy Non-Cartesian Raw Adjoint NRMSE: $nrmse_nc_sp"
            @test nrmse_nc_sp < 0.01

            # BART Raw Non-Cartesian Adjoint (via run_bart / BartIO)
            traj_bart = zeros(ComplexF32, 3, 128, 128)
            traj_bart[1, :, :] .= reshape(t.nodes[1, :], 128, 128) .* 128.0f0
            traj_bart[2, :, :] .= reshape(t.nodes[2, :], 128, 128) .* 128.0f0
            kdata_bart = reshape(kdata_nc_sim, 1, 128, 128, Nc)
            res_bart = ComparisonHarness.run_bart(1, "nufft -a", traj_bart, kdata_bart)
            bart_adj_mc = sum(res_bart[:, :, 1, :] .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims=3)[:, :, 1]

            scale_bart_nc = norm(abs.(bart_adj_mc)) / norm(abs.(mrt_adj_nodcf))
            nrmse_nc_bart = compute_nrmse(mrt_adj_nodcf .* scale_bart_nc, bart_adj_mc)
            @info "MRT vs BART Non-Cartesian Raw Adjoint NRMSE: $nrmse_nc_bart"
            @test nrmse_nc_bart < 0.01
        end

        @testset "CG-SENSE" begin
            # MRT CG-SENSE (with tight tolerance to run all 10 iterations)
            method = IterativeReconstruction(regularization=(), algorithm=MriReconstructionToolbox.CGNR(maxit=10, tol=1e-14))
            x_mrt = reconstruct(acq_mc, method; tol=1e-14, maxit=10)

            # Compare MRT CG-SENSE to Ground Truth
            scale_mrt_gt = norm(abs.(img_mc)) / norm(abs.(x_mrt))
            nrmse_mrt_gt = compute_nrmse(x_mrt .* scale_mrt_gt, img_mc)
            @info "MRT vs Ground Truth CG-SENSE NRMSE: $nrmse_mrt_gt"
            @test nrmse_mrt_gt < 1e-8

            # SigPy CG-SENSE
            kdata_sp_mc = parent(permutedims(kspace_mc, (3, 2, 1)))
            smaps_sp_mc = parent(permutedims(cmap, (3, 2, 1)))
            sp_cg = sp_mri.app.SenseRecon(kdata_sp_mc, smaps_sp_mc, max_iter=10, show_pbar=false).run()
            sp_cg = permutedims(sp_cg, (2, 1))
            scale_sp = norm(abs.(sp_cg)) / norm(abs.(x_mrt))
            nrmse_sp = compute_nrmse(x_mrt .* scale_sp, sp_cg)
            @info "MRT vs SigPy CG-SENSE NRMSE: $nrmse_sp"
            @test nrmse_sp < 1e-12

            nrmse_sp_gt = compute_nrmse(sp_cg .* (norm(abs.(img_mc)) / norm(abs.(sp_cg))), img_mc)
            @info "SigPy vs Ground Truth CG-SENSE NRMSE: $nrmse_sp_gt"
            @test nrmse_sp_gt < 1e-8

            # MRIReco CG-SENSE
            kdata_mr = reshape(kspace_mc, N, N, 1, Nc, 1, 1)
            acq_mr = AcquisitionData(kdata_mr)
            smaps_mr = reshape(cmap, N, N, 1, Nc)
            recoParams = Dict{Symbol, Any}(:reco => "multiCoil", :reconSize => (N, N), :senseMaps => smaps_mr, :iterations => 10, :solver => MRIReco.CGNR, :reg => [MRIReco.L2Regularization(0.0)])
            img_mr = MRIReco.reconstruction(acq_mr, recoParams)[:, :, 1, 1, 1]
            scale_mr = norm(abs.(img_mr)) / norm(abs.(x_mrt))
            nrmse_mr = compute_nrmse(x_mrt .* scale_mr, img_mr)
            @info "MRT vs MRIReco CG-SENSE NRMSE: $nrmse_mr"
            @test nrmse_mr < 1e-12

            nrmse_mr_gt = compute_nrmse(img_mr .* (norm(abs.(img_mc)) / norm(abs.(img_mr))), img_mc)
            @info "MRIReco vs Ground Truth CG-SENSE NRMSE: $nrmse_mr_gt"
            @test nrmse_mr_gt < 1e-8

            # BART CG-SENSE (via ComparisonHarness.run_bart / BartIO)
            kdata_bart_cart = reshape(kspace_mc, N, N, 1, Nc)
            smaps_bart_cart = reshape(cmap, N, N, 1, Nc)
            bart_cg = ComparisonHarness.run_bart(1, "pics -S -i 10", ComplexF32.(kdata_bart_cart), ComplexF32.(smaps_bart_cart))[:, :, 1]
            scale_bart = norm(abs.(bart_cg)) / norm(abs.(x_mrt))
            nrmse_bart = compute_nrmse(x_mrt .* scale_bart, bart_cg)
            @info "MRT vs BART CG-SENSE NRMSE: $nrmse_bart"
            @test nrmse_bart < 1e-6

            nrmse_bart_gt = compute_nrmse(bart_cg .* (norm(abs.(img_mc)) / norm(abs.(bart_cg))), img_mc)
            @info "BART vs Ground Truth CG-SENSE NRMSE: $nrmse_bart_gt"
            @test nrmse_bart_gt < 1e-6
        end
    end
end
