# Section: k-space GRAPPA (MRT only — kept as MRT reference rows) on the regularly sampled variant
# of the 2D and multislice multichannel cases (`get_case(id; pattern = :regular)`: every other phase
# encode plus a 24-line calibration block, since GRAPPA cannot fit a kernel to a random pattern).
#
# No cross-toolkit row is possible here, re-verified against each toolkit: BART 0.9.00's command
# list has no `grappa` (its k-space methods are `caldir` / `ecalib` / `sake` / `nlinv` / `pocsense`,
# none of which is autocalibrated GRAPPA kernel fitting), `sigpy.mri.app` ships SENSE / L1-wavelet /
# TV / JSENSE but no GRAPPA app, and MRIReco's reconstruction API exposes only `direct` /
# `multiCoil` solvers. There is nothing to compare against, not merely nothing convenient.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_kspace.jl --threads=N [--use-mkl]
include(joinpath(@__DIR__, "_setup.jl"))

for id in ("shepp_logan_2d_8ch_cartesian", "shepp_logan_multislice_8ch_cartesian")
    DATA == "real" && break
    should_run_case(id) || continue
    c = get_case(id; pattern = :regular)
    acq = mrt_acquisition(c)
    for (meth, cc) in (("GRAPPA (RSS)", RootSumSquares()), ("GRAPPA (Sensitivity)", AdjointSensitivity()))
        should_run(c.id, meth) || should_run("K-Space", meth) || continue
        println("--> $(c.id): $meth")
        calib = small_mode() ? 12 : 24
        m = GRAPPA(kernel_size = (4, 3), calib_size = (calib, calib), coil_combination = cc)
        t, _, x = time_run(() -> reconstruct(acq, m; verbosity = Silent()); runs = timed_runs(c))
        push!(results, BenchResult("K-Space", meth, FW, NUM_THREADS, t * 1000, mag_nrmse(Array(parent(x)), c.reference), 0.0, c.id, c.source))
        flush_results!("kspace")
    end
end

write_section("kspace")
