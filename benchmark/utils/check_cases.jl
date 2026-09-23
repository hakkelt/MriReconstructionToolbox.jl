# Sanity checks for the benchmark case catalog.
#
#   MRT_BENCH_SMALL=1 julia --project=benchmark benchmark/utils/check_cases.jl [--cases=pat,...] [--real]
#
# For every case: array layouts agree with each other, the Cartesian acceleration is within 5% of
# the design value, a direct reconstruction of the *noiseless, fully sampled* problem reproduces
# the phantom (exactly for Cartesian, to gridding accuracy for non-Cartesian), and the k-space is
# bit-identical to what a fresh process generates (the hash printed here is compared across two
# runs by eye or by script). Prints one line per case and exits non-zero on any failure.

include(joinpath(@__DIR__, "bench_utils.jl"))
using .BenchUtils
using .BenchUtils: centred_fft
using MriReconstructionToolbox
using Printf
using SHA

load_site_env!()
const PATTERNS = let i = findfirst(a -> startswith(a, "--cases="), ARGS)
    i === nothing ? nothing : split(ARGS[i][(length("--cases=") + 1):end], ",")
end
const IDS = filter_case_ids(case_ids(; real = "--real" in ARGS), PATTERNS)

# Design accelerations of the synthetic Cartesian cases (real cases intersect the pattern with what
# the scanner sampled, so theirs is only reported).
const DESIGN_R = Dict(
    "shepp_logan_2d_1ch_cartesian" => 2.5, "shepp_logan_2d_8ch_cartesian" => 4.0,
    "shepp_logan_multislice_8ch_cartesian" => 4.0, "shepp_logan_3d_8ch_cartesian" => 6.0,
    "torso_cine_8ch_cartesian" => 4.0,
)

failures = String[]
check(ok, msg) = ok || push!(failures, msg)

for id in IDS
    t = @elapsed c = get_case(id)
    h = bytes2hex(SHA.sha1(reinterpret(UInt8, vec(c.kspace))))[1:12]
    line = @sprintf("%-40s %6.1f s  kspace %s  sha1 %s", id, t, join(size(c.kspace), "×"), h)
    # layouts
    fam = c.family
    nc = ncoils(c)
    if c.smaps !== nothing
        cd = fam === :volume ? 4 : 3
        check(size(c.smaps, cd) == nc, "$id: maps have $(size(c.smaps, cd)) coils, k-space $nc")
    end
    if c.trajectory === :cartesian
        R = acceleration(c)
        line *= @sprintf("  R %.2f", R)
        haskey(DESIGN_R, id) && check(abs(R / DESIGN_R[id] - 1) <= 0.05, "$id: R = $R, designed $(DESIGN_R[id])")
        # Every unsampled entry is zero.
        m = BenchUtils._broadcast_mask(c.mask, fam, size(c.kspace))
        check(all(iszero, c.kspace .* .!m), "$id: nonzero k-space outside the mask")
    else
        check(size(c.kspace)[1:2] == size(c.traj)[2:3], "$id: k-space and trajectory disagree")
    end
    # A direct reconstruction of the noiseless, fully sampled version of the synthetic problem.
    if !c.real && fam === :single_slice
        n = c.image_size[1]
        img = c.reference
        maps = c.smaps === nothing ? ones(ComplexF32, n, n, 1) : c.smaps
        if c.trajectory === :cartesian
            k = centred_fft(img .* maps, (1, 2))
            full = BenchUtils.BenchCase(;
                id, family = fam, trajectory = :cartesian, reference = img, smaps = c.smaps,
                kspace = ComplexF32.(k), mask = trues(n, n), image_size = c.image_size,
            )
            x = Array(parent(mrt_reconstructor(full, :adjoint)()))
            c.smaps === nothing || (x ./= dropdims(sum(abs2, maps; dims = 3); dims = 3))
            e = BenchUtils.nrmse(x, img)
            line *= @sprintf("  full adjoint %.1e", e)
            check(e < 1.0e-5, "$id: noiseless fully sampled adjoint NRMSE $e")
        else
            # Noiseless, Nyquist-sampled (π/2 · n spokes) radial data through the same simulation,
            # gridded by MRT with the ramp DCF: validates the trajectory and FFT-shift conventions.
            # Scored against the phantom low-passed to the |k| ≤ 1/2 disc a radial trajectory covers
            # (the corners hold 14% of Shepp-Logan's energy at 128²). Ramp-DCF gridding without
            # deapodisation then scores 0.10 at 128² and 0.15 at 32², while a transposed or
            # half-FOV-shifted convention scores above 1, so 0.25 separates the two.
            traj = BenchUtils.golden_angle_radial(2n, ceil(Int, π / 2 * n))
            k = BenchUtils._nfft_forward(reshape(img .* maps, n, n, :), traj)
            full = BenchUtils.BenchCase(;
                id, family = fam, trajectory = :noncartesian, reference = img, smaps = c.smaps,
                kspace = k, traj, dcf = BenchUtils.ramp_dcf(traj), image_size = c.image_size,
            )
            x = Array(parent(mrt_reconstructor(full, :gridding)())) ./ dropdims(sum(abs2, maps; dims = 3); dims = 3)
            kr = ((-(n ÷ 2)):(n - n ÷ 2 - 1)) ./ n
            disc = [hypot(a, b) <= 0.5 for a in kr, b in kr]
            e = mag_nrmse(x, BenchUtils.centred_ifft(centred_fft(img, (1, 2)) .* disc, (1, 2)))
            line *= @sprintf("  full gridding %.3f", e)
            check(e < 0.25, "$id: noiseless Nyquist-sampled gridding NRMSE $e (vs disc-limited phantom)")
        end
    end
    println(line)
end

if isempty(failures)
    println("all $(length(IDS)) cases passed")
else
    println("FAILED:\n  ", join(failures, "\n  "))
    exit(1)
end
