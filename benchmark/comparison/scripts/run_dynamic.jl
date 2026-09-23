# Section: temporal priors on every cine catalog case — global low-rank ↔ BART `-R L -b <image>`,
# locally low-rank ↔ `-R L -b 8`, temporal TV ↔ `-R T:32` — MRT vs BART vs MRIReco vs SigPy vs MIRT.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_dynamic.jl --threads=N [--use-mkl] [--data=synthetic|real|all]
#
# SigPy and MIRT have no stock low-rank MRI app, but both take an arbitrary prox, so they join the
# **global** low-rank row through `sigpy_lowrank` / `mirt_lowrank` — a nuclear norm on the Casorati
# matrix is all that row is. Neither joins LLR (block extraction and cycle spinning are conventions
# the harness would be inventing) nor temporal TV. MRIReco joins the two low-rank rows through
# `mrireco_dynamic` (frames as contrasts); it cannot express temporal TV — see that docstring.
#
# ## Three BART `-R L` flags this section needs, all established by measurement
#
# * **`-b` defaults to 8, so a bare `-R L:3:3:λ` is *locally* low rank with 8×8 blocks.** A
#   "global" row without `-b` was the same reconstruction as the LLR row bit for bit; `-b <image
#   size>` makes the image one block.
# * **`-R L` selects FISTA, not ADMM**, because the low-rank prox needs no linear transform. `-m`
#   forces ADMM so both sides run the same algorithm and `-u` / `-C` mean something.
# * **`-n` disables random block cycle spinning**, which BART applies to LLR by default. MRT's
#   `LocallyLowRank` defaults to a fixed tiling, so `-n` makes the two the same objective.
#
# ## Temporal TV is the one method whose accuracy depends on the *inner* solve
#
# Fixed-ρ ADMM on `D_t` (a large null space: anything constant in time) barely couples `z` to
# `D_t x` at ρ = CMP_RHO, so a *more* exact inner CG makes the result worse (measured 0.0769 at
# cg = 10, 0.2440 at cg = 80); BART hides this behind its hardcoded `1e-3 · ‖rhs‖` inner tolerance.
# At CMP_CG_ITERS = 10 MRT and BART agree closely, which is the operating point λ is calibrated at.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
include(joinpath(@__DIR__, "_methods.jl"))

for c in section_cases(c -> c.family === :cine), m in (:lowrank, :llr, :ttv)
    run_method_rows!("Dynamic", c, m)
end

write_section("dynamic")
