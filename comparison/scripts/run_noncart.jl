# Section: non-Cartesian radial DCF adjoint / gridding — MRT vs MRIReco (vs BART nufft).
#   julia --project=comparison -t N comparison/scripts/run_noncart.jl --threads=N [--use-mkl]
#
# ## The two toolkits do not grid at the same accuracy by default
#
# MRT's `NFFTOp` takes NFFT.jl's own defaults — kernel half-width `m = 5`, oversampling `σ = 2.0`,
# `precompute = POLYNOMIAL`. MRIReco (`LinearOperatorCollection`'s `NFFTOp`) hardcodes `m = 3`,
# `σ = 1.25`, `precompute = TENSOR`. That is not a small difference: measured on this trajectory,
# single-coil adjoint, single thread,
#
# | m | σ | precompute | adjoint / coil | forward error vs `m=8, σ=2` |
# |---|---|---|---|---|
# | 5 | 2.00 | POLYNOMIAL (MRT)     | 4.28 ms | 1.6e-7 |
# | 4 | 2.00 | POLYNOMIAL           | 3.63 ms | 2.6e-7 |
# | 3 | 1.25 | TENSOR (MRIReco)     | 1.00 ms | 5.7e-5 |
#
# so MRT pays 4.3× per coil for an NFFT that is ~360× more accurate — accuracy this reconstruction
# never uses, since gridding-adjoint NRMSE against the phantom is 0.085 either way. Whole multi-coil
# DCF adjoint, one run, single thread: MRT 36.5 ms at its default, **8.35 ms** at MRIReco's operating
# point, MRIReco 47.1 ms — i.e. MRT wins even while gridding more accurately, and by 5.6× at matched
# accuracy. (Only compare figures from one run: an earlier login-node run under contention read
# 115.7 / 16.0 / 37.5 ms for the same three rows.) Both MRT rows are reported: `MRT` at its default
# and `MRT (m=3, σ=1.25)` matched to MRIReco.
# MRT's public API does not currently forward NFFT plan options through
# `NonCartesianAcquisitionInfo` / `get_encoding_operator` (see TODO.md), which is why the matched row
# builds the operator by hand.
include(joinpath(@__DIR__, "_setup.jl"))
using LinearAlgebra: mul!

img_mc, cmap = IMG_MC, CMAP

t = RadialTrajectory(Float32, N, N; TE = 0.0f0, AQ = 1.0f-3)
traj_named = NamedDimsArray(t.nodes, (:dim, :k))
smaps_nc = NamedDimsArray(ComplexF32.(cmap), (:x, :y, :coil))

acq_sim = NonCartesianAcquisitionInfo(
    NamedDimsArray(zeros(ComplexF32, 16384, Nc), (:k, :coil));
    trajectory = traj_named, image_size = (N, N), sensitivity_maps = smaps_nc, shifted_image_dims = (:x, :y),
)
kdata_nc = MriReconstructionToolbox.get_encoding_operator(acq_sim) * NamedDimsArray(ComplexF32.(img_mc), (:x, :y))

acq_dcf = NonCartesianAcquisitionInfo(kdata_nc; trajectory = traj_named, image_size = (N, N), sensitivity_maps = smaps_nc, shifted_image_dims = (:x, :y))
E_dcf = MriReconstructionToolbox.get_encoding_operator(acq_dcf)
tm, _, xm_raw = time_reconstruction(() -> E_dcf' * kdata_nc)
xm = xm_raw .* (norm(abs.(img_mc)) / norm(abs.(xm_raw)))
push!(results, BenchResult("Non-Cartesian", "DCF Adjoint (Gridding)", FW, NUM_THREADS, tm * 1000, nrmse(xm, img_mc), 0.0))

# MRT at MRIReco's NFFT operating point. `NFFTOp` is not exported, and the `precompute` enum lives in
# NFFT.jl, which the comparison environment does not depend on directly — both are reached through
# the operator MRT already built.
let
    nfft_inner = E_dcf.L.A[3].operator          # Compose(broadcast, dcf-diag, BatchOp(NFFTOp))
    NFFTmod = parentmodule(typeof(nfft_inner.plan))
    o = MriReconstructionToolbox.NFFTOp(
        (N, N), parent(traj_named), nfft_inner.dcf;
        threaded = false, m = 3, σ = 1.25f0, precompute = NFFTmod.TENSOR,
    )
    kdm = collect(parent(kdata_nc))
    smap = ComplexF32.(cmap)
    function adj_matched()
        acc = zeros(ComplexF32, N, N)
        buf = zeros(ComplexF32, N, N)
        for c in 1:Nc
            mul!(buf, o', @view kdm[:, c])
            @. acc += buf * conj(@view smap[:, :, c])
        end
        return acc
    end
    tmm, _, x_raw = time_reconstruction(adj_matched)
    x = x_raw .* (norm(abs.(img_mc)) / norm(abs.(x_raw)))
    push!(results, BenchResult("Non-Cartesian", "DCF Adjoint (Gridding)", "$FW (m=3, σ=1.25)", NUM_THREADS, tmm * 1000, nrmse(x, img_mc), nrmse(xm, x)))
end

try
    kdata_mr = reshape(kdata_nc, 16384, Nc, 1, 1)
    acq_mr = AcquisitionData(t, fill(kdata_mr[:, :, 1, 1], 1, 1, 1))
    smap_mr = reshape(ComplexF32.(cmap), N, N, 1, Nc)
    rp = Dict{Symbol, Any}(:reco => "direct", :reconSize => (N, N), :senseMaps => smap_mr)
    # The coil combination is inside the timed closure: MRT's row includes the sensitivity adjoint,
    # and MRIReco's `direct` reco returns per-coil images, so combining outside would undercount it.
    tr, _, xr_raw = time_reconstruction() do
        imr = MRIReco.reconstruction(acq_mr, rp)[:, :, 1, 1, :]
        return sum(imr .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims = 3)[:, :, 1]
    end
    xr = xr_raw .* (norm(abs.(img_mc)) / norm(abs.(xr_raw)))
    push!(results, BenchResult("Non-Cartesian", "DCF Adjoint (Gridding)", "MRIReco", NUM_THREADS, tr * 1000, nrmse(xr, img_mc), nrmse(xm, xr)))
catch e
    @warn "MRIReco non-Cartesian failed" exception = (e, catch_backtrace())
end

write_section("noncart")
