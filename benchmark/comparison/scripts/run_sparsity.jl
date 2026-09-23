# Section: spatial sparsity — TV / L1-wavelet / TGV — on every static catalog case, MRT vs BART vs
# SigPy vs MRIReco.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_sparsity.jl --threads=N [--use-mkl] [--data=synthetic|real|all]
#
# Matched-effort comparison: CMP_OUTER (20) outer iterations, CMP_CG_ITERS (10) inner CG, fixed
# ADMM ρ, no early stop for the in-process toolkits. k-space is unit-RMS normalised and each
# toolkit uses its own calibrated λ (`load_lambda`), so the operating point, not the nominal λ, is
# matched. See `run_accuracy_race.jl` for the time-to-target-NRMSE view; this section is the
# fixed-effort snapshot. The volume runs 3D TV / 3D wavelet; TGV is 2D only (`applicable_methods`).
#
# ## BART flags — four non-obvious ones, all verified against the 0.9.00 source
#
# * **`-i N` is not the outer iteration count for ADMM.** `src/iter/admm.c:453` breaks on
#   `nr_invokes > maxiter`, and `nr_invokes` accumulates ≈ (outer iterations + cumulative inner CG
#   iterations) — so `-i N` is effectively a budget of N applications of the normal operator.
#   Measured: `-i 20` → 3 outer / 21 CG, `-i 80` → 29 outer / 52 CG, `-i 150` → 99 outer / 52 CG.
#   `bart_cmd` passes `CMP_OUTER × CMP_CG_ITERS` (see `BART_BUDGET`).
# * **`-w 1` disables BART's internal k-space rescaling.** `pics` otherwise divides k-space so the
#   ~90th-percentile magnitude of a low-res RSS image is ≈1 (`src/sense/optcom.c:71-87`) and leaves
#   λ untouched, putting its λ on a private scale that no calibration can transfer.
# * **`-F` changes the algorithm, not the iteration count.** The eps_pri/eps_dual break it disables
#   is already dead in `pics` (`src/grecon/italgo.c:142-143` hardcodes `ABSTOL = RELTOL = 0`); what
#   `-F` actually does is switch off over-relaxation, α 1.6 → 1.0 (`admm.c:341-354`). Kept because
#   plain ADMM is what MRT, MRIReco and SigPy run.
# * **`-e` is mandatory for the FISTA (wavelet) path.** Without it `pics` assumes λ_max = 1 and uses
#   a hardcoded step of 0.95 (`src/pics.c:793-794, 801`); with `-w 1` the true λ_max(𝒜ᴴ𝒜) is above
#   2/0.95 here and the recon stalls for every λ. `-e` runs a 30-iteration power method
#   (`src/iter/misc.c:51-65`) — a real one-off cost, counted honestly.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
include(joinpath(@__DIR__, "_methods.jl"))

for c in section_cases(c -> c.family !== :cine), m in (:tv, :wavelet, :tgv)
    m in applicable_methods(c) || continue
    run_method_rows!("Sparsity", c, m)
end

write_section("sparsity")
