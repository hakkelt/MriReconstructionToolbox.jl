# Section: CG-SENSE (10 iterations, no early stop) on every multichannel catalog case — MRT vs
# SigPy vs BART vs MRIReco vs MIRT.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_cgsense.jl --threads=N [--use-mkl] [--data=synthetic|real|all]
#
# The cases are undersampled, so the normal equations are not trivially conditioned and every
# toolkit has ten iterations of real work to do. (On a fully sampled acquisition with normalised
# maps 𝒜ᴴ𝒜 = I: BART's `pics` hit its own residual tolerance and stopped after one iteration while
# the others ran all ten, measured as a flat 0.58 ms/it for BART against 4.82 for MRT on the same
# problem — an early exit, not a faster iteration. BART's inner tolerance is hardcoded, so the only
# way to hold every toolkit to ten iterations is a problem that needs ten.)
#
# `tol = 0.0` everywhere: with a small positive tolerance MRT's CGNR exits as soon as the residual
# underflows, which is a real capability but not the same work.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
include(joinpath(@__DIR__, "_methods.jl"))

for c in section_cases(c -> :cgsense in applicable_methods(c))
    run_method_rows!("CG-SENSE", c, :cgsense)
end

write_section("cgsense")
