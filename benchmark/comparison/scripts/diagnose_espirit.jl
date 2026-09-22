# Why do MRT's ESPIRiT maps disagree with BART's, MRIReco's and SigPy's?
#
# `compare_preprocessing.jl` compares masked magnitudes and asserts NRMSE < 0.1; MRT sits at 0.77
# against both. This script separates the candidate explanations that error cannot tell apart:
# a genuine map difference, a per-pixel scale convention (every toolbox here normalises the map
# vector to unit RSS per pixel, so the maps carry no magnitude information), a global scale, and
# the *subspace* threshold that decides how many k-space kernels the eigen-decomposition sees.
#
#   julia --project=benchmark/comparison benchmark/comparison/scripts/diagnose_espirit.jl

using MriReconstructionToolbox
using LinearAlgebra
using Statistics
using FFTW

include("../src/ComparisonHarness.jl")
using .ComparisonHarness: nrmse, generate_multicoil_brain

import MRICoilSensitivities

const N = 64
const NC = 8
const CALIB = 24
const KSIZE = 6

img, kspace, cmap = generate_multicoil_brain(N = N, num_coils = NC)
mask = repeat(abs.(img) .> 1.0e-4, 1, 1, NC)

rss(m) = sqrt.(sum(abs2, m; dims = 3))
unit(m) = m ./ max.(rss(m), eps(Float32))
masked_nrmse(a, b) = nrmse(unit(abs.(a))[mask], unit(abs.(b))[mask])

mrt_maps(; eig = 0.0, sub = 0.0) = MriReconstructionToolbox.estimate_sensitivities(
    kspace; method = MriReconstructionToolbox.ESPIRiT(calib_size = CALIB, kernel_size = KSIZE, eigenvalue_threshold = eig, subspace_threshold = sub)
)

bart = dropdims(ComparisonHarness.run_bart(2, "ecalib -r $CALIB -c 0 -m 1", ComplexF32.(reshape(kspace, N, N, 1, NC)))[1], dims = 3)
sp = permutedims(
    ComparisonHarness.sigpy_mri_app.EspiritCalib(permutedims(kspace, (3, 2, 1)), calib_width = CALIB, crop = 0.0, show_pbar = false).run(),
    (2, 3, 1)
)

# MRIReco's `espirit` takes the calibration block itself, and returns maps on `fftshift`ed axes.
# `eigThresh_1` is its subspace threshold (default 0.02) and `eigThresh_2` its eigenvalue crop.
calib = let c = N ÷ 2 + 1, r = (c - CALIB ÷ 2):(c + CALIB ÷ 2 - 1)
    ComplexF32.(kspace[r, r, :])
end
mrireco = MRICoilSensitivities.espirit(calib, (N, N), (KSIZE, KSIZE); eigThresh_1 = 0.02, eigThresh_2 = 0.0)
mrireco = dropdims(mrireco; dims = ndims(mrireco))

ucmap = unit(abs.(cmap))
gt(m) = nrmse(ucmap[mask], unit(abs.(m))[mask])

println("per-pixel RSS over the mask (1.0 everywhere = unit-normalised maps):")
for (name, m) in (("MRT (sub=0)", mrt_maps()), ("BART", bart), ("SigPy", sp), ("MRIReco", mrireco))
    r = vec(rss(abs.(m)))[vec(abs.(img) .> 1.0e-4)]
    println("  $name  mean $(round(mean(r), digits = 4))  min $(round(minimum(r), digits = 4))  max $(round(maximum(r), digits = 4))")
end

# The raw-array `estimate_sensitivities` returns maps in MRT's *default* image convention (origin at
# index 1), while every other toolbox here returns centred maps. Which layout each one is in is the
# first thing to settle, before any of the errors below mean anything.
println()
println("layout: NRMSE against the ground-truth maps under each candidate transform")
for (name, m) in (("MRT (sub=0)", mrt_maps()), ("BART", bart), ("SigPy", sp), ("MRIReco", mrireco))
    a = unit(abs.(m))
    variants = (
        "as returned" => a,
        "fftshift(1:2)" => fftshift(a, 1:2),
        "transpose" => permutedims(a, (2, 1, 3)),
        "fftshift ∘ transpose" => fftshift(permutedims(a, (2, 1, 3)), 1:2),
    )
    best = argmin(v -> nrmse(ucmap[mask], v[2][mask]), collect(variants))
    for (label, v) in variants
        println("  $name  $label  $(nrmse(ucmap[mask], v[mask]))")
    end
    println("  $name  best: $(best[1])")
end

println()
println("NRMSE against the simulated ground-truth maps (all unit-normalised magnitudes):")
println("  BART     $(gt(bart))")
println("  SigPy    $(gt(sp))")
println("  MRIReco  $(gt(mrireco))")
for sub in (0.0, 0.001, 0.01, 0.02, 0.05, 0.1)
    m = mrt_maps(sub = sub)
    # MRT returns maps in its default (origin-at-index-1) convention; every other toolbox here
    # returns centred ones, so the comparable form is the `fftshift`ed one.
    s = fftshift(m, 1:2)
    println("  MRT sub=$(sub)  as returned $(gt(m))   fftshifted $(gt(s))   vs BART $(masked_nrmse(bart, s))   vs MRIReco $(masked_nrmse(mrireco, s))")
end

# How many kernels each threshold keeps, and what the top eigenvalue of `V V'` looks like: BART and
# MRIReco normalise it to ≈ 1 so that `eigenvalue_threshold` is a crop in [0, 1].
println()
println("subspace size and eigenvalue scale:")
let c = N ÷ 2 + 1, r = (c - CALIB ÷ 2):(c + CALIB ÷ 2 - 1)
    cal = ComplexF32.(kspace[r, r, :])
    np = (CALIB - KSIZE + 1)^2
    C = zeros(ComplexF32, np, KSIZE * KSIZE * NC)
    i = 1
    for o in CartesianIndices((CALIB - KSIZE + 1, CALIB - KSIZE + 1))
        C[i, :] = reshape(cal[o[1]:(o[1] + KSIZE - 1), o[2]:(o[2] + KSIZE - 1), :], :)
        i += 1
    end
    S = svd(C).S
    println("  singular values: $(length(S)) total, S[1] = $(S[1]), S[end] = $(S[end])")
    for sub in (0.0, 0.001, 0.01, 0.02, 0.05, 0.1)
        println("  sub=$(sub) keeps $(max(1, count(>=(S[1] * sub), S))) of $(length(S)) kernels")
    end
end

# ESPIRiT is two steps — kernels out of the calibration matrix, then a per-pixel eigen-decomposition
# of those kernels in image space. Crossing MRT's step 1 with MRIReco's step 2 and vice versa says
# which of the two is wrong; comparing each to a literal transcription of Uecker's MATLAB says how.

# MRT's step 1, as `_estimate_sensitivities_core` does it: rows are patches, columns are
# `(kx, ky, coil)`, kernels are the leading right singular vectors.
function mrt_kernels(nv)
    c = N ÷ 2 + 1
    r = (c - CALIB ÷ 2):(c + CALIB ÷ 2 - 1)
    cal = ComplexF32.(kspace[r, r, :])
    npd = CALIB - KSIZE + 1
    C = zeros(ComplexF32, npd^2, KSIZE * KSIZE * NC)
    i = 1
    for o in CartesianIndices((npd, npd))
        C[i, :] = reshape(cal[o[1]:(o[1] + KSIZE - 1), o[2]:(o[2] + KSIZE - 1), :], :)
        i += 1
    end
    V = svd(C).V
    return reshape(V[:, 1:nv], KSIZE, KSIZE, NC, nv)
end

# MRT's step 2: flip-conjugate the kernel, drop it at the array origin, `circshift` it to centre,
# plain (uncentred) DFT, then the top eigenvector of `V V'` per pixel.
function mrt_step2(kern)
    nv = size(kern, 4)
    V_img = zeros(ComplexF64, N, N, NC, nv)
    for v in 1:nv
        flipped = reverse(conj(kern[:, :, :, v]), dims = (1, 2))
        for c in 1:NC
            padded = zeros(ComplexF64, N, N)
            padded[1:KSIZE, 1:KSIZE] = flipped[:, :, c]
            V_img[:, :, c, v] = fft(circshift(padded, (-(KSIZE ÷ 2), -(KSIZE ÷ 2))))
        end
    end
    maps = zeros(ComplexF64, N, N, NC)
    for i in CartesianIndices((N, N))
        V_r = reshape(V_img[i, :, :], NC, nv)
        F = eigen(Hermitian(V_r * V_r'))
        maps[i, :] = F.vectors[:, end]
    end
    return maps
end

# Uecker's `kernelEig`, transcribed: the same flip-conjugate, but a *centred* transform, and the
# eigenvalue normalised by the kernel size so the threshold means what it does everywhere else.
function ref_step2(kern)
    nv = size(kern, 4)
    kerimgs = zeros(ComplexF64, N, N, NC, nv)
    for v in 1:nv, c in 1:NC
        k = zeros(ComplexF64, N, N)
        k[1:KSIZE, 1:KSIZE] = conj(reverse(kern[:, :, c, v], dims = (1, 2)))
        kerimgs[:, :, c, v] = fftshift(fft(ifftshift(k))) ./ sqrt(KSIZE^2)
    end
    maps = zeros(ComplexF64, N, N, NC)
    vals = zeros(Float64, N, N)
    for i in CartesianIndices((N, N))
        G = reshape(kerimgs[i, :, :], NC, nv)
        F = eigen(Hermitian(G * G'))
        maps[i, :] = F.vectors[:, end]
        vals[i] = F.values[end]
    end
    println("  (reference step 2: top eigenvalue min $(round(minimum(vals), digits = 3)) max $(round(maximum(vals), digits = 3)))")
    return maps
end

println()
println("crossing the two steps (24-kernel subspace, both orderings of the layout):")
let nv = 24
    k_mrt = mrt_kernels(nv)
    k_mri, S_mri = MRICoilSensitivities.dat2Kernel(ComplexF32.(kspace[(N ÷ 2 + 1 - CALIB ÷ 2):(N ÷ 2 + CALIB ÷ 2), (N ÷ 2 + 1 - CALIB ÷ 2):(N ÷ 2 + CALIB ÷ 2), :]), (KSIZE, KSIZE))
    k_mri = k_mri[:, :, :, 1:nv]
    println("  kernel subspaces agree to: $(opnorm(reshape(k_mrt, :, nv) * reshape(k_mrt, :, nv)' - reshape(k_mri, :, nv) * reshape(k_mri, :, nv)'))  (0 = identical subspace)")
    for (name, m) in (
            "MRT step1 + MRT step2" => mrt_step2(k_mrt),
            "MRIReco step1 + MRT step2" => mrt_step2(k_mri),
            "MRT step1 + reference step2" => ref_step2(k_mrt),
            "MRIReco step1 + reference step2" => ref_step2(k_mri),
        )
        a = unit(abs.(m))
        println("  $name: as returned $(nrmse(ucmap[mask], a[mask]))  fftshift $(nrmse(ucmap[mask], fftshift(a, 1:2)[mask]))")
    end
end
