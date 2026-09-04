# Section: 1-coil + multi-coil Cartesian adjoint (MRT / SigPy / BART / MRIReco).
#   julia --project=comparison -t N comparison/scripts/run_base.jl --threads=N [--use-mkl]
#
# **BART is not timed in this section — it cannot be, and the row it used to print was noise.** The
# in-process adjoints here take 3–6 ms, while a `bart` invocation costs ~100–130 ms of process spawn
# plus disk I/O, with ±30 ms of run-to-run jitter on the login node (measured: `bart version`
# 94–124 ms, `bart copy` 152–215 ms, `bart fft -i 3` 163–253 ms on this input). Subtracting the
# `bart_overhead` estimate from a 2 ms compute leaves a difference far inside that jitter, and
# `time_bart`'s `max(1e-5, …)` floor then reported BART as 0.01 ms — i.e. 300× faster than everyone
# else, which is an artifact. BART still runs, and its output is still checked against MRT's for
# agreement, but its `time_ms` is recorded as unmeasurable (-1). The iterative sections are
# unaffected: there the solver dominates the fixed overhead.
#
# Measured, single thread, multi-coil adjoint: MRT 3.04 ms, MRIReco 4.57 ms, SigPy 6.05 ms — all at
# NRMSE 0 against the phantom (a fully sampled adjoint is exact).
include(joinpath(@__DIR__, "_setup.jl"))

img_mc, kspace_mc, cmap = IMG_MC, KSPACE_MC, CMAP
smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))

# ---- 1-coil adjoint ----
println("--> 1-coil adjoint")
kspace_1c = ifftshift(fft(fftshift(img_mc)))
acq_1c = CartesianAcquisitionInfo(NamedDimsArray(kspace_1c, (:kx, :ky)); is3D = false, shifted_image_dims = (:x, :y))
E_1c = MriReconstructionToolbox.get_encoding_operator(acq_1c)
t_mrt, _, mrt_raw = time_reconstruction(() -> E_1c' * NamedDimsArray(kspace_1c, (:kx, :ky)))
mrt_1c = mrt_raw .* (norm(abs.(img_mc)) / norm(abs.(mrt_raw)))
push!(results, BenchResult("Base 1C", "1-Coil Adjoint", FW, NUM_THREADS, t_mrt * 1000, nrmse(mrt_1c, img_mc), 0.0))

kdata_sp_1c = parent(permutedims(kspace_1c, (2, 1)))
F_sp_1c = sp_mri.linop.Sense(ones(ComplexF64, 1, N, N), ishape = (N, N))
t_sp, _, sp_raw = time_reconstruction(() -> F_sp_1c.H(reshape(kdata_sp_1c, 1, N, N)))
sp_1c = permutedims(sp_raw, (2, 1)); sp_1c = sp_1c .* (norm(abs.(img_mc)) / norm(abs.(sp_1c)))
push!(results, BenchResult("Base 1C", "1-Coil Adjoint", "SigPy", NUM_THREADS, t_sp * 1000, nrmse(sp_1c, img_mc), nrmse(mrt_1c, sp_1c)))

_, _, b_raw = time_bart("fft -i 3", ComplexF32.(reshape(kspace_1c, N, N, 1, 1)))
b_1c = b_raw[:, :, 1, 1]; b_1c = b_1c .* (norm(abs.(img_mc)) / norm(abs.(b_1c)))
push!(results, BenchResult("Base 1C", "1-Coil Adjoint", BART_FW, NUM_THREADS, NaN, nrmse(b_1c, img_mc), nrmse(mrt_1c, b_1c)))

# ---- multi-coil adjoint ----
println("--> multi-coil adjoint")
kdata_mc = NamedDimsArray(kspace_mc, (:kx, :ky, :coil))
acq_mc = CartesianAcquisitionInfo(kdata_mc; is3D = false, sensitivity_maps = smaps_mc, shifted_image_dims = (:x, :y))
E_mc = MriReconstructionToolbox.get_encoding_operator(acq_mc)
t_mrt, _, mrt_adj = time_reconstruction(() -> E_mc' * kdata_mc)
recon = mrt_adj ./ sum(abs2.(smaps_mc), dims = 3)[:, :, 1]
push!(results, BenchResult("Base MC", "Cartesian Adjoint", FW, NUM_THREADS, t_mrt * 1000, mag_nrmse(recon, img_mc), 0.0))

kdata_sp_mc = parent(permutedims(kspace_mc, (3, 2, 1)))
smaps_sp_mc = parent(permutedims(cmap, (3, 2, 1)))
S_sp = sp_mri.linop.Sense(smaps_sp_mc, ishape = (N, N))
t_sp, _, sp_raw = time_reconstruction(() -> S_sp.H(kdata_sp_mc))
sp_adj = permutedims(sp_raw, (2, 1)) ./ sum(abs2.(cmap), dims = 3)[:, :, 1]
push!(results, BenchResult("Base MC", "Cartesian Adjoint", "SigPy", NUM_THREADS, t_sp * 1000, mag_nrmse(sp_adj, img_mc), mag_nrmse(recon, sp_adj)))

kdata_bart_cart = reshape(kspace_mc, N, N, 1, Nc)
_, _, b_ifft = time_bart("fft -i 3", ComplexF32.(kdata_bart_cart))
b_adj = sum(b_ifft[:, :, 1, :] .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims = 3)[:, :, 1] ./ sum(abs2.(ComplexF32.(cmap)), dims = 3)[:, :, 1]
push!(results, BenchResult("Base MC", "Cartesian Adjoint", BART_FW, NUM_THREADS, NaN, mag_nrmse(b_adj, img_mc), mag_nrmse(recon, b_adj)))

# MRIReco multi-coil adjoint (Cartesian direct): AcquisitionData accepts a dense
# (x, y, z, channel, echo, rep) k-space array directly (`enc2D` for a 2D encode).
try
    acq_mr = AcquisitionData(reshape(ComplexF64.(kspace_mc), N, N, 1, Nc, 1, 1); enc2D = true)
    rp = Dict{Symbol, Any}(:reco => "direct", :reconSize => (N, N), :senseMaps => reshape(ComplexF64.(cmap), N, N, 1, Nc))
    t_mr, _, mr_img = time_reconstruction(() -> MRIReco.reconstruction(acq_mr, rp)[:, :, 1, 1, :])
    mr = sum(mr_img .* conj.(reshape(ComplexF64.(cmap), N, N, Nc)), dims = 3)[:, :, 1] ./ sum(abs2.(cmap), dims = 3)[:, :, 1]
    push!(results, BenchResult("Base MC", "Cartesian Adjoint", "MRIReco", NUM_THREADS, t_mr * 1000, mag_nrmse(mr, img_mc), mag_nrmse(recon, mr)))
catch e
    @warn "MRIReco Cartesian adjoint failed" exception = (e, catch_backtrace())
end

write_section("base")
