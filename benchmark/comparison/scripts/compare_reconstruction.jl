using MriReconstructionToolbox
using MriReconstructionToolbox: NonCartesianAcquisitionInfo
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
using .ComparisonHarness: check_nrmse, nrmse, run_bart, generate_multicoil_brain

sigpy = pyimport("sigpy")
sp_mri = pyimport("sigpy.mri")

@testset "Reconstruction Benchmark" begin
    # Base setup
    N = 128
    Nc = 8

    # Generate test phantoms and sensitivity maps
    img_1c, kspace_1c, smaps_1c_raw = ComparisonHarness.generate_multicoil_brain(N = N, num_coils = 1)
    img_mc, kspace_mc, cmap = ComparisonHarness.generate_multicoil_brain(N = N, num_coils = Nc)

    kdata_1c = NamedDimsArray(kspace_1c, (:kx, :ky, :coil))
    smaps_1c = NamedDimsArray(smaps_1c_raw, (:x, :y, :coil))
    acq_1c = CartesianAcquisitionInfo(kdata_1c; is3D = false, sensitivity_maps = smaps_1c, shifted_image_dims = (:x, :y))

    kdata_mc = NamedDimsArray(kspace_mc, (:kx, :ky, :coil))
    smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))
    acq_mc = CartesianAcquisitionInfo(kdata_mc; is3D = false, sensitivity_maps = smaps_mc, shifted_image_dims = (:x, :y))

    @testset "Base Reconstruction (Unregularized)" begin
        @testset "Cartesian Adjoint" begin
            # 1-Coil fully-sampled
            E_1c = MriReconstructionToolbox.get_encoding_operator(acq_1c)
            mrt_adj_1c = E_1c' * kdata_1c

            # SigPy Adjoint (1-coil)
            kdata_sp_1c = parent(permutedims(kspace_1c, (3, 2, 1)))
            smaps_sp_1c = parent(permutedims(smaps_1c, (3, 2, 1)))
            S_sp_1c = sp_mri.linop.Sense(smaps_sp_1c, ishape = (N, N))
            sp_adj_1c = permutedims(S_sp_1c.H(kdata_sp_1c), (2, 1))

            check_nrmse(mrt_adj_1c, sp_adj_1c, 1.0e-4; label = "MRT vs SigPy Cartesian 1C Adjoint")

            # Compare MRT and SigPy to Ground Truth (normalized by |s|^2)
            recon_1c_mrt = mrt_adj_1c ./ abs2.(smaps_1c)
            check_nrmse(recon_1c_mrt, img_1c, 1.0e-4; label = "MRT vs Ground Truth Cartesian 1C Adjoint")

            recon_1c_sp = sp_adj_1c ./ parent(abs2.(smaps_1c))
            check_nrmse(recon_1c_sp, img_1c, 1.0e-4; label = "SigPy vs Ground Truth Cartesian 1C Adjoint")

            # Multi-coil fully-sampled MRT Adjoint
            E_mc = MriReconstructionToolbox.get_encoding_operator(acq_mc)
            mrt_adj_mc = E_mc' * kdata_mc

            # SigPy Adjoint (multi-coil)
            kdata_sp_mc = parent(permutedims(kspace_mc, (3, 2, 1)))
            smaps_sp_mc = parent(permutedims(cmap, (3, 2, 1)))
            S_sp_mc = sp_mri.linop.Sense(smaps_sp_mc, ishape = (N, N))
            sp_adj_mc = permutedims(S_sp_mc.H(kdata_sp_mc), (2, 1))

            check_nrmse(mrt_adj_mc, sp_adj_mc, 1.0e-4; label = "MRT vs SigPy Cartesian MC Adjoint")

            # MRIReco Adjoint (multi-coil fully sampled)
            kdata_mr = reshape(kspace_mc, N, N, 1, Nc, 1, 1)
            acq_mr = AcquisitionData(kdata_mr)
            smaps_mr = reshape(cmap, N, N, 1, Nc)
            recoParams = Dict{Symbol, Any}(:reco => "direct", :reconSize => (N, N), :senseMaps => smaps_mr)
            img_mr_direct = MRIReco.reconstruction(acq_mr, recoParams)[:, :, 1, 1, :]
            mr_adj_mc = sum(img_mr_direct .* conj.(cmap), dims = 3)[:, :, 1]
            check_nrmse(mrt_adj_mc, mr_adj_mc, 1.0e-4; label = "MRT vs MRIReco Cartesian MC Adjoint")

            # BART Adjoint (multi-coil fully sampled via IFFT)
            kdata_bart_cart = reshape(kspace_mc, N, N, 1, Nc)
            bart_ifft = ComparisonHarness.run_bart(1, "fft -i 3", ComplexF32.(kdata_bart_cart))
            bart_adj_mc = sum(bart_ifft[:, :, 1, :] .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims = 3)[:, :, 1]
            check_nrmse(mrt_adj_mc, bart_adj_mc, 1.0e-4; label = "MRT vs BART Cartesian MC Adjoint")

            # Compare all multi-coil fully sampled adjoints to Ground Truth (normalized by sum|s|^2)
            coil_norm_mc = sum(abs2.(cmap), dims = 3)[:, :, 1]
            check_nrmse(mrt_adj_mc ./ coil_norm_mc, img_mc, 1.0e-4; label = "MRT vs Ground Truth Cartesian MC Adjoint")
            check_nrmse(sp_adj_mc ./ coil_norm_mc, img_mc, 1.0e-4; label = "SigPy vs Ground Truth Cartesian MC Adjoint")
            check_nrmse(mr_adj_mc ./ coil_norm_mc, img_mc, 1.0e-4; label = "MRIReco vs Ground Truth Cartesian MC Adjoint")
            check_nrmse(bart_adj_mc ./ coil_norm_mc, img_mc, 1.0e-4; label = "BART vs Ground Truth Cartesian MC Adjoint")

            # Multi-coil undersampled MRT Adjoint
            mask = rand(MersenneTwister(42), Bool, N, N)
            kdata_us = NamedDimsArray(kspace_mc[mask, :], (:kxy, :coil))
            acq_us = CartesianAcquisitionInfo(kdata_us; is3D = false, image_size = (N, N), sensitivity_maps = smaps_mc, shifted_image_dims = (:x, :y), subsampling = mask)
            E_us = MriReconstructionToolbox.get_encoding_operator(acq_us)
            mrt_adj_us = E_us' * kdata_us

            # SigPy Adjoint (undersampled)
            kspace_sp_us = copy(kspace_mc)
            kspace_sp_us[.!mask, :] .= 0
            kdata_sp_us = parent(permutedims(kspace_sp_us, (3, 2, 1)))
            sp_adj_us = permutedims(S_sp_mc.H(kdata_sp_us), (2, 1))

            check_nrmse(mrt_adj_us, sp_adj_us, 1.0e-4; label = "MRT vs SigPy Cartesian US Adjoint")

            # MRIReco Adjoint (undersampled)
            kdata_mr_us = reshape(kspace_sp_us, N, N, 1, Nc, 1, 1)
            acq_mr_us = AcquisitionData(kdata_mr_us)
            img_mr_us_direct = MRIReco.reconstruction(acq_mr_us, recoParams)[:, :, 1, 1, :]
            mr_adj_us = sum(img_mr_us_direct .* conj.(cmap), dims = 3)[:, :, 1]
            check_nrmse(mrt_adj_us, mr_adj_us, 1.0e-4; label = "MRT vs MRIReco Cartesian US Adjoint")
        end

        @testset "Non-Cartesian Adjoint" begin
            # Setup radial trajectory
            t = RadialTrajectory(Float32, N, N; TE = 0.0f0, AQ = 1.0f-3)
            traj_named = NamedDimsArray(t.nodes, (:dim, :k))
            smaps_nc = NamedDimsArray(ComplexF32.(cmap), (:x, :y, :coil))

            # Simulate Non-Cartesian multi-coil data
            kdata_nc_zeros = NamedDimsArray(zeros(ComplexF32, 16384, Nc), (:k, :coil))
            acq_nc_sim = NonCartesianAcquisitionInfo(kdata_nc_zeros; trajectory = traj_named, image_size = (N, N), sensitivity_maps = smaps_nc, shifted_image_dims = (:x, :y))
            E_nc_sim = MriReconstructionToolbox.get_encoding_operator(acq_nc_sim)
            kdata_nc_sim = E_nc_sim * NamedDimsArray(ComplexF32.(img_mc), (:x, :y))

            # 1) Density-compensated Adjoint (Gridding / Direct Reconstruction)
            acq_nc_dcf = NonCartesianAcquisitionInfo(kdata_nc_sim; trajectory = traj_named, image_size = (N, N), sensitivity_maps = smaps_nc, shifted_image_dims = (:x, :y))
            E_nc_dcf = MriReconstructionToolbox.get_encoding_operator(acq_nc_dcf)
            mrt_adj_dcf = E_nc_dcf' * kdata_nc_sim

            # Compare Density-Compensated MRT and MRIReco Adjoint to Ground Truth Phantom
            check_nrmse(mrt_adj_dcf, img_mc, 0.1; label = "MRT vs Ground Truth Non-Cartesian DCF Adjoint")

            # MRIReco Non-Cartesian Direct Reconstruction
            kdata_mr_nc = reshape(kdata_nc_sim, 16384, Nc, 1, 1)
            acq_mr_nc = AcquisitionData(t, fill(kdata_mr_nc[:, :, 1, 1], 1, 1, 1))
            smaps_mr_f32 = reshape(ComplexF32.(cmap), N, N, 1, Nc)
            recoParams_nc = Dict{Symbol, Any}(:reco => "direct", :reconSize => (N, N), :senseMaps => smaps_mr_f32)
            img_mr_nc_direct = MRIReco.reconstruction(acq_mr_nc, recoParams_nc)[:, :, 1, 1, :]
            mr_adj_nc = sum(img_mr_nc_direct .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims = 3)[:, :, 1]
            check_nrmse(mrt_adj_dcf, mr_adj_nc, 0.05; label = "MRT vs MRIReco Non-Cartesian DCF Adjoint")
            check_nrmse(mr_adj_nc, img_mc, 0.1; label = "MRIReco vs Ground Truth Non-Cartesian DCF Adjoint")

            # 2) Raw Non-Cartesian Adjoint (without DCF)
            acq_nc_nodcf = NonCartesianAcquisitionInfo(kdata_nc_sim; trajectory = traj_named, dcf = ones(Float32, 16384), image_size = (N, N), sensitivity_maps = smaps_nc, shifted_image_dims = (:x, :y))
            E_nc_nodcf = MriReconstructionToolbox.get_encoding_operator(acq_nc_nodcf)
            mrt_adj_nodcf = E_nc_nodcf' * kdata_nc_sim

            # SigPy Raw Non-Cartesian Adjoint
            coord_sp_yx = Float32.(reverse(parent(t.nodes), dims = 1)') .* Float32(N)
            kdata_sp_nc = parent(permutedims(kdata_nc_sim, (2, 1)))
            smaps_sp_mc_f32 = ComplexF32.(parent(permutedims(cmap, (3, 2, 1))))
            S_sp_nc = sp_mri.linop.Sense(smaps_sp_mc_f32, coord = coord_sp_yx, ishape = (N, N))
            sp_adj_nodcf = permutedims(S_sp_nc.H(kdata_sp_nc), (2, 1))

            check_nrmse(mrt_adj_nodcf, sp_adj_nodcf, 0.01; label = "MRT vs SigPy Non-Cartesian Raw Adjoint")

            # BART Raw Non-Cartesian Adjoint (via run_bart / BartIO)
            traj_bart = zeros(ComplexF32, 3, 128, 128)
            traj_bart[1, :, :] .= reshape(t.nodes[1, :], 128, 128) .* 128.0f0
            traj_bart[2, :, :] .= reshape(t.nodes[2, :], 128, 128) .* 128.0f0
            kdata_bart = reshape(kdata_nc_sim, 1, 128, 128, Nc)
            res_bart = ComparisonHarness.run_bart(1, "nufft -a", traj_bart, kdata_bart)
            bart_adj_mc = sum(res_bart[:, :, 1, :] .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims = 3)[:, :, 1]

            check_nrmse(mrt_adj_nodcf, bart_adj_mc, 0.01; label = "MRT vs BART Non-Cartesian Raw Adjoint")
        end

        @testset "CG-SENSE" begin
            # MRT CG-SENSE (with tight tolerance to run all 10 iterations)
            method = IterativeReconstruction(regularization = (), algorithm = MriReconstructionToolbox.CGNR(maxit = 10, tol = 1.0e-14); maxit = 10, tol = 1.0e-14)
            x_mrt = reconstruct(acq_mc, method)

            # Compare MRT CG-SENSE to Ground Truth
            check_nrmse(x_mrt, img_mc, 1.0e-8; label = "MRT vs Ground Truth CG-SENSE")

            # SigPy CG-SENSE
            kdata_sp_mc = parent(permutedims(kspace_mc, (3, 2, 1)))
            smaps_sp_mc = parent(permutedims(cmap, (3, 2, 1)))
            sp_cg = sp_mri.app.SenseRecon(kdata_sp_mc, smaps_sp_mc, max_iter = 10, show_pbar = false).run()
            sp_cg = permutedims(sp_cg, (2, 1))
            check_nrmse(x_mrt, sp_cg, 1.0e-12; label = "MRT vs SigPy CG-SENSE")
            check_nrmse(sp_cg, img_mc, 1.0e-8; label = "SigPy vs Ground Truth CG-SENSE")

            # MRIReco CG-SENSE
            kdata_mr = reshape(kspace_mc, N, N, 1, Nc, 1, 1)
            acq_mr = AcquisitionData(kdata_mr)
            smaps_mr = reshape(cmap, N, N, 1, Nc)
            recoParams = Dict{Symbol, Any}(:reco => "multiCoil", :reconSize => (N, N), :senseMaps => smaps_mr, :iterations => 10, :solver => MRIReco.CGNR, :reg => [MRIReco.L2Regularization(0.0)])
            img_mr = MRIReco.reconstruction(acq_mr, recoParams)[:, :, 1, 1, 1]
            check_nrmse(x_mrt, img_mr, 1.0e-12; label = "MRT vs MRIReco CG-SENSE")
            check_nrmse(img_mr, img_mc, 1.0e-8; label = "MRIReco vs Ground Truth CG-SENSE")

            # BART CG-SENSE (via ComparisonHarness.run_bart / BartIO)
            kdata_bart_cart = reshape(kspace_mc, N, N, 1, Nc)
            smaps_bart_cart = reshape(cmap, N, N, 1, Nc)
            bart_cg = ComparisonHarness.run_bart(1, "pics -S -i 10", ComplexF32.(kdata_bart_cart), ComplexF32.(smaps_bart_cart))[:, :, 1]
            check_nrmse(x_mrt, bart_cg, 1.0e-6; label = "MRT vs BART CG-SENSE")
            check_nrmse(bart_cg, img_mc, 1.0e-6; label = "BART vs Ground Truth CG-SENSE")
        end
    end

    @testset "Sparsity-Based Regularization" begin
        # 2x Undersampled test setup with center calibration region
        mask_reg = rand(MersenneTwister(42), Bool, N, N)
        mask_reg[(N ÷ 2 - 8):(N ÷ 2 + 8), :] .= true

        kdata_reg_us = NamedDimsArray(kspace_mc[mask_reg, :], (:kxy, :coil))
        acq_reg_us = CartesianAcquisitionInfo(kdata_reg_us; is3D = false, image_size = (N, N), sensitivity_maps = smaps_mc, shifted_image_dims = (:x, :y), subsampling = mask_reg)

        kspace_reg_sp = copy(kspace_mc)
        kspace_reg_sp[.!mask_reg, :] .= 0
        kdata_sp_reg = parent(permutedims(kspace_reg_sp, (3, 2, 1)))
        smaps_sp_reg = parent(permutedims(cmap, (3, 2, 1)))

        kdata_bart_reg = reshape(kspace_reg_sp, N, N, 1, Nc)
        smaps_bart_reg = reshape(cmap, N, N, 1, Nc)

        @testset "Total Variation (TV)" begin
            λ_tv = 0.01

            # MRT TV
            method_tv = IterativeReconstruction(regularization = TotalVariation2D(λ_tv); maxit = 30, tol = 1.0e-5)
            x_mrt_tv = reconstruct(acq_reg_us, method_tv)
            check_nrmse(x_mrt_tv, img_mc, 0.05; label = "MRT TV vs Ground Truth")

            # BART TV
            bart_tv = ComparisonHarness.run_bart(1, "pics -S -i 30 -R T:3:0:0.01", ComplexF32.(kdata_bart_reg), ComplexF32.(smaps_bart_reg))[:, :, 1]
            check_nrmse(bart_tv, img_mc, 0.05; label = "BART TV vs Ground Truth")
            check_nrmse(x_mrt_tv, bart_tv, 0.05; label = "MRT vs BART TV")

            # MRIReco TV
            kdata_mr_tv = reshape(kspace_reg_sp, N, N, 1, Nc, 1, 1)
            acq_mr_tv = AcquisitionData(kdata_mr_tv)
            smaps_mr_tv = reshape(cmap, N, N, 1, Nc)
            recoParams_tv = Dict{Symbol, Any}(:reco => "multiCoil", :reconSize => (N, N), :senseMaps => smaps_mr_tv, :iterations => 30, :solver => MRIReco.ADMM, :reg => [MRIReco.TVRegularization(λ_tv; shape = (N, N))])
            img_mr_tv = MRIReco.reconstruction(acq_mr_tv, recoParams_tv)[:, :, 1, 1, 1]
            check_nrmse(img_mr_tv, img_mc, 0.05; label = "MRIReco TV vs Ground Truth")
            check_nrmse(x_mrt_tv, img_mr_tv, 0.05; label = "MRT vs MRIReco TV")

            # SigPy TV
            sp_tv = sp_mri.app.TotalVariationRecon(kdata_sp_reg, smaps_sp_reg, lamda = Float64(λ_tv), max_iter = 30, show_pbar = false).run()
            sp_tv = permutedims(sp_tv, (2, 1))
            check_nrmse(sp_tv, img_mc, 0.15; label = "SigPy TV vs Ground Truth")
            check_nrmse(x_mrt_tv, sp_tv, 0.15; label = "MRT vs SigPy TV")
        end

        @testset "L1-Wavelet" begin
            λ_wav = 0.005

            # MRT L1-Wavelet
            method_wav = IterativeReconstruction(regularization = L1Wavelet2D(λ_wav); maxit = 30, tol = 1.0e-5)
            x_mrt_wav = reconstruct(acq_reg_us, method_wav)
            check_nrmse(x_mrt_wav, img_mc, 0.05; label = "MRT Wavelet vs Ground Truth")

            # BART L1-Wavelet (ADMM without cycle spinning)
            bart_wav = ComparisonHarness.run_bart(1, "pics -m -l1 -r 0.005 -n -S -i 30", ComplexF32.(kdata_bart_reg), ComplexF32.(smaps_bart_reg))[:, :, 1]
            check_nrmse(bart_wav, img_mc, 0.05; label = "BART Wavelet vs Ground Truth")
            check_nrmse(x_mrt_wav, bart_wav, 0.05; label = "MRT vs BART Wavelet")

            # SigPy L1-Wavelet
            sp_wav = sp_mri.app.L1WaveletRecon(kdata_sp_reg, smaps_sp_reg, lamda = Float64(λ_wav), max_iter = 30, show_pbar = false).run()
            sp_wav = permutedims(sp_wav, (2, 1))
            check_nrmse(sp_wav, img_mc, 0.1; label = "SigPy Wavelet vs Ground Truth")
            check_nrmse(x_mrt_wav, sp_wav, 0.1; label = "MRT vs SigPy Wavelet")
        end

        @testset "Total Generalized Variation (TGV)" begin
            λ_tgv = 0.01

            # MRT TGV
            method_tgv = IterativeReconstruction(regularization = TotalGeneralizedVariation2D(λ_tgv; ratio = 2.0); maxit = 30, tol = 1.0e-5)
            x_mrt_tgv = reconstruct(acq_reg_us, method_tgv)
            check_nrmse(x_mrt_tgv, img_mc, 0.05; label = "MRT TGV vs Ground Truth")

            # BART TGV
            bart_tgv = ComparisonHarness.run_bart(1, "pics -S -i 30 -R G:3:0:0.01", ComplexF32.(kdata_bart_reg), ComplexF32.(smaps_bart_reg))[:, :, 1]
            check_nrmse(bart_tgv, img_mc, 0.05; label = "BART TGV vs Ground Truth")
            check_nrmse(x_mrt_tgv, bart_tgv, 0.05; label = "MRT vs BART TGV")
        end
    end

    @testset "Dynamic / Low-Rank Regularization" begin
        Nd = 64
        Ncd = 4
        Td = 8
        img_dyn, kspace_dyn, cmap_dyn = ComparisonHarness.generate_dynamic_multicoil_brain(N = Nd, num_coils = Ncd, num_frames = Td)

        # 2x Phase encoding undersampling
        mask_pe = rand(MersenneTwister(42), Bool, Nd)
        mask_pe[(Nd ÷ 2 - 4):(Nd ÷ 2 + 4)] .= true

        kspace_dyn_us = kspace_dyn[:, mask_pe, :, :]
        kdata_dyn_us = NamedDimsArray(permutedims(kspace_dyn_us, (1, 2, 4, 3)), (:kx, :ky, :coil, :time))
        smaps_dyn_named = NamedDimsArray(cmap_dyn, (:x, :y, :coil))

        acq_dyn = CartesianAcquisitionInfo(
            kdata_dyn_us;
            is3D = false,
            image_size = (Nd, Nd),
            sensitivity_maps = smaps_dyn_named,
            subsampling = (:, mask_pe),
            shifted_image_dims = (:x, :y)
        )

        kdata_bart_dyn = zeros(ComplexF32, Nd, Nd, 1, Ncd, 1, Td)
        for t in 1:Td
            kdata_bart_dyn[:, mask_pe, 1, :, 1, t] .= ComplexF32.(kspace_dyn[:, mask_pe, t, :])
        end
        smaps_bart_dyn = reshape(ComplexF32.(cmap_dyn), Nd, Nd, 1, Ncd)

        @testset "Global Low-Rank" begin
            λ_lr = 0.01

            # MRT Low-Rank
            method_lr = IterativeReconstruction(regularization = LowRank(λ_lr; time_dim = :time); maxit = 20, tol = 1.0e-4)
            x_mrt_lr = reconstruct(acq_dyn, method_lr)
            check_nrmse(x_mrt_lr, img_dyn, 0.2; label = "MRT Low-Rank vs Ground Truth")
        end

        @testset "Locally Low-Rank (LLR)" begin
            λ_llr = 0.01

            # MRT Locally Low-Rank
            method_llr = IterativeReconstruction(regularization = LocallyLowRank(λ_llr; block_size = (8, 8), time_dim = :time); maxit = 20, tol = 1.0e-4)
            x_mrt_llr = reconstruct(acq_dyn, method_llr)
            check_nrmse(x_mrt_llr, img_dyn, 0.2; label = "MRT LLR vs Ground Truth")

            # BART Locally Low-Rank
            bart_llr = ComparisonHarness.run_bart(1, "pics -S -i 20 -b 8 -R L:3:3:0.01", kdata_bart_dyn, smaps_bart_dyn)
            bart_llr_img = dropdims(bart_llr, dims = (3, 4, 5))
            check_nrmse(bart_llr_img, img_dyn, 0.2; label = "BART LLR vs Ground Truth")
            check_nrmse(x_mrt_llr, bart_llr_img, 0.1; label = "MRT vs BART LLR")
        end

        @testset "Temporal Total Variation (tTV)" begin
            λ_ttv = 0.01

            # MRT Temporal TV
            method_ttv = IterativeReconstruction(regularization = TemporalTotalVariation(λ_ttv; time_dim = :time); maxit = 20, tol = 1.0e-4)
            x_mrt_ttv = reconstruct(acq_dyn, method_ttv)
            check_nrmse(x_mrt_ttv, img_dyn, 0.2; label = "MRT Temporal TV vs Ground Truth")

            # BART Temporal TV (time is dimension index 5)
            bart_ttv = ComparisonHarness.run_bart(1, "pics -S -i 20 -R T:32:0:0.01", kdata_bart_dyn, smaps_bart_dyn)
            bart_ttv_img = dropdims(bart_ttv, dims = (3, 4, 5))
            check_nrmse(bart_ttv_img, img_dyn, 0.2; label = "BART Temporal TV vs Ground Truth")
            check_nrmse(x_mrt_ttv, bart_ttv_img, 0.1; label = "MRT vs BART Temporal TV")
        end
    end

    @testset "K-Space Methods" begin
        # 2x Undersampling with 24 ACS lines along ky
        mask_grappa = falses(N, N)
        mask_grappa[:, 1:2:N] .= true
        mask_grappa[:, (N ÷ 2 - 12):(N ÷ 2 + 11)] .= true

        kdata_grappa_us = NamedDimsArray(kspace_mc[mask_grappa, :], (:kxy, :coil))
        acq_grappa = CartesianAcquisitionInfo(
            kdata_grappa_us;
            is3D = false,
            image_size = (N, N),
            sensitivity_maps = smaps_mc,
            subsampling = mask_grappa,
            shifted_image_dims = (:x, :y)
        )

        @testset "GRAPPA" begin
            # MRT GRAPPA (RootSumSquares)
            method_grappa_rss = GRAPPA(kernel_size = (4, 3), calib_size = (24, 24), coil_combination = RootSumSquares())
            x_mrt_grappa_rss = reconstruct(acq_grappa, method_grappa_rss)
            check_nrmse(x_mrt_grappa_rss, img_mc, 0.05; label = "MRT GRAPPA (RSS) vs Ground Truth")

            # MRT GRAPPA (AdjointSensitivity)
            method_grappa_sens = GRAPPA(kernel_size = (4, 3), calib_size = (24, 24), coil_combination = AdjointSensitivity())
            x_mrt_grappa_sens = reconstruct(acq_grappa, method_grappa_sens)
            check_nrmse(x_mrt_grappa_sens, img_mc, 0.05; label = "MRT GRAPPA (Sensitivity) vs Ground Truth")
        end
    end
end
