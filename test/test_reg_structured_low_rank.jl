using TestItems

@testitem "StructuredLowRank regularization" tags = [:regularization] setup = [RegTestSetup, ProxOf] begin
    using LinearAlgebra
    using FFTW: fft, fftshift
    import Random
    Random.seed!(0)

    @testset "Constructor" begin
        reg = StructuredLowRank(λ = 0.1, window = (5, 5))
        @test reg.λ == 0.1
        @test reg.max_rank === nothing
        @test reg.window == (5, 5)
        @test reg.structure === :c
        @test reg.batch_dims === nothing

        reg2 = StructuredLowRank(max_rank = 8, window = (4, 4, 3), batch_dims = (:time,))
        @test reg2.λ === nothing
        @test reg2.max_rank == 8
        @test reg2.batch_dims == (:time,)

        # `λ` and `max_rank` are mutually exclusive (NAMING.md rule 1.3)
        @test_throws ArgumentError StructuredLowRank(window = (5, 5))
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, max_rank = 4, window = (5, 5))
        @test_throws ArgumentError StructuredLowRank(λ = -0.1, window = (5, 5))
        @test_throws ArgumentError StructuredLowRank(max_rank = 0, window = (5, 5))
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5,))
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 0))
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 5), structure = :hankel)

        # the LORAKS phase structures
        @test StructuredLowRank(λ = 0.1, window = (5, 5), structure = :s).structure === :s
        @test StructuredLowRank(max_rank = 20, window = (5, 5), structure = :g).structure === :g
        @test StructuredLowRank(λ = 0.1, window = (5, 5)).kspace_center === nothing
        @test StructuredLowRank(λ = 0.1, window = (5, 5), structure = :s, kspace_center = (1, 1)).kspace_center == (1, 1)
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 5), structure = :s, kspace_center = (0, 3))
        # ALOHA's weighting is defined for the C-matrix lift only
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 5), structure = :s, weights = :tv)
        # and the structure is carried through a rescale
        @test scale_regularization(
            StructuredLowRank(λ = 0.2, window = (4, 3), structure = :s, kspace_center = (3, 2)), 3.0
        ).structure === :s
        @test scale_regularization(
            StructuredLowRank(λ = 0.2, window = (4, 3), structure = :s, kspace_center = (3, 2)), 3.0
        ).kspace_center == (3, 2)
    end

    @testset "get_operator is the identity" for threaded in [false, true]
        x = randn(ComplexF64, 12, 10, 4)
        op = get_operator(StructuredLowRank(λ = 0.1, window = (4, 3)), x; threaded)
        @test op * x ≈ x
        named = NamedDimsArray{(:kx, :ky, :coil)}(x)
        @test get_operator(StructuredLowRank(λ = 0.1, window = (4, 3)), named; threaded) isa
            MriReconstructionToolbox.NamedDimsOp
    end

    @testset "get_affected_dims" begin
        reg = StructuredLowRank(λ = 0.1, window = (4, 3))
        @test get_affected_dims(reg, nothing, (:kx, :ky, :coil)) == (:kx, :ky, :coil)
        reg_t = StructuredLowRank(λ = 0.1, window = (4, 3), batch_dims = (:time,))
        @test get_affected_dims(reg_t, nothing, (:kx, :ky, :coil, :time)) == (:kx, :ky, :coil)
    end

    @testset "scale_regularization" begin
        # the nuclear norm is homogeneous of degree 1, a rank cap is scale-invariant
        @test scale_regularization(StructuredLowRank(λ = 0.2, window = (4, 3)), 3.0).λ ≈ 0.6
        scaled = scale_regularization(StructuredLowRank(max_rank = 5, window = (4, 3)), 3.0)
        @test scaled.max_rank == 5
        @test scaled.λ === nothing
    end

    @testset "materialize both forms" for threaded in [false, true]
        for w in ((4, 3), (3, 3, 2))
            dims = length(w) == 2 ? (12, 10, 4) : (8, 7, 6, 3)
            x = Variable(randn(ComplexF64, dims...))
            @test materialize(StructuredLowRank(λ = 0.05, window = w), x; threaded) !== nothing
            @test materialize(StructuredLowRank(max_rank = 4, window = w), x; threaded) !== nothing
        end
    end

    @testset "materialize rejects an input that cannot be lifted" begin
        # fewer dims than the window plus a channel axis
        @test_throws ArgumentError materialize(
            StructuredLowRank(λ = 0.1, window = (4, 3)), Variable(randn(ComplexF64, 12, 10)); threaded = false
        )
        # window larger than the k-space grid
        @test_throws ArgumentError materialize(
            StructuredLowRank(λ = 0.1, window = (20, 3)), Variable(randn(ComplexF64, 12, 10, 2)); threaded = false
        )
    end

    @testset "nuclear-norm prox == Cadzow reference" begin
        gx, gy, nc = 12, 10, 3
        w = (4, 3)
        x = randn(ComplexF64, gx, gy, nc)
        γ, λ = 0.7, 0.15
        y, _ = prox_of(StructuredLowRank(λ = λ, window = w), x, γ)

        H = AbstractOperators.Hankel(ComplexF64, (gx, gy), w; nchannels = nc, channels = true)
        M = H * x
        F = svd(M)
        S = max.(0.0, F.S .- λ * γ)
        Mhat = F.U * Diagonal(S) * F.Vt
        invmult = 1.0 ./ real.(AbstractOperators.diag_AcA(H))
        ref = (H' * Mhat) .* invmult
        @test y ≈ ref
    end

    @testset "rank prox == hard-truncation Cadzow reference" begin
        gx, gy, nc = 16, 14, 2
        w = (5, 4)
        x = randn(ComplexF64, gx, gy, nc)
        r = 6
        y, _ = prox_of(StructuredLowRank(max_rank = r, window = w), x, 1.0)

        H = AbstractOperators.Hankel(ComplexF64, (gx, gy), w; nchannels = nc, channels = true)
        F = svd(H * x)
        S = copy(F.S)
        S[(r + 1):end] .= 0
        @test count(!iszero, S) == r                        # the lifted matrix was truncated to rank r
        Mhat = F.U * Diagonal(S) * F.Vt
        invmult = 1.0 ./ real.(AbstractOperators.diag_AcA(H))
        ref = (H' * Mhat) .* invmult
        @test y ≈ ref
    end

    @testset "the prox is threading-invariant" begin
        x = randn(ComplexF64, 12, 10, 3, 2)
        seq, _ = prox_of(StructuredLowRank(λ = 0.1, window = (4, 3)), x, 0.5)
        # `prox_of` builds the term with threaded = false; rebuild it threaded and compare
        term = materialize(StructuredLowRank(λ = 0.1, window = (4, 3)), Variable(x); threaded = true)
        f = MriReconstructionToolbox.StructuredOptimization.weighted_function(term)
        par = similar(x)
        MriReconstructionToolbox.ProximalCore.prox!(par, f, x, 0.5)
        @test par ≈ seq
    end

    @testset "calculate: nuclear norm of the lifted matrix" begin
        gx, gy, nc = 12, 10, 3
        w = (4, 3)
        x = randn(ComplexF64, gx, gy, nc)
        λ = 0.1
        v = calculate(StructuredLowRank(λ = λ, window = w), x; threaded = false)
        H = AbstractOperators.Hankel(ComplexF64, (gx, gy), w; nchannels = nc, channels = true)
        @test v ≈ λ * sum(svdvals(H * x))
    end

    @testset "weights: constructor validation" begin
        @test StructuredLowRank(λ = 0.1, window = (5, 5)).weights === nothing
        @test StructuredLowRank(λ = 0.1, window = (5, 5), weights = :tv).weights === :tv
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 5), weights = :haar)
        # one dimension per k-space encoding dimension
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 5), weights = randn(8, 8, 8))
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 5), weights = ())
        @test_throws ArgumentError StructuredLowRank(λ = 0.1, window = (5, 5), weights = (:tv,))
        # scale_regularization carries the weights through
        @test scale_regularization(StructuredLowRank(λ = 0.2, window = (4, 3), weights = :tv), 3.0).weights === :tv
    end

    @testset "weights: the built-in models" begin
        gx, gy = 12, 10
        tv = MriReconstructionToolbox._slr_weights(
            StructuredLowRank(λ = 0.1, window = (4, 3), weights = :tv), (gx, gy), ComplexF64
        )
        # the approximation band plus one first difference per encoding dimension
        @test length(tv) == 3
        # Stored separably -- a difference symbol varies along one encoding dimension only -- so
        # what is pinned is the behaviour that matters: each weight broadcasts to the full grid
        # with a singleton channel axis.
        @test all(size(w .* zeros(ComplexF64, gx, gy, 1)) == (gx, gy, 1) for w in tv)
        @test all(isone, tv[1])
        # a first difference vanishes at DC, which on MRT's centered grid is index N ÷ 2 + 1
        @test tv[2][gx ÷ 2 + 1, 1, 1] == 0
        @test tv[3][1, gy ÷ 2 + 1, 1] == 0
        # the pyramid adds the step-2 detail band per dimension
        wav = MriReconstructionToolbox._slr_weights(
            StructuredLowRank(λ = 0.1, window = (4, 3), weights = :wavelet), (gx, gy), ComplexF64
        )
        @test length(wav) == 5
        # several weights are averaged, a single one is used directly
        avg = MriReconstructionToolbox.StructuredOptimization.weighted_function(
            materialize(StructuredLowRank(λ = 0.1, window = (4, 3), weights = :tv), Variable(randn(ComplexF64, gx, gy, 2)); threaded = false)
        )
        @test avg isa MriReconstructionToolbox.ProximalAverage
        one_w = MriReconstructionToolbox.StructuredOptimization.weighted_function(
            materialize(
                StructuredLowRank(λ = 0.1, window = (4, 3), weights = ones(ComplexF64, gx, gy)),
                Variable(randn(ComplexF64, gx, gy, 2)); threaded = false,
            )
        )
        @test one_w isa MriReconstructionToolbox.HankelLowRankProx
    end

    @testset "weighted prox == weighted-Cadzow reference" begin
        gx, gy, nc = 12, 10, 3
        win = (4, 3)
        x = randn(ComplexF64, gx, gy, nc)
        γ, λ = 0.7, 0.15
        weight = randn(ComplexF64, gx, gy) .+ 2.0        # no zeros: the plain weighted formula
        y, _ = prox_of(StructuredLowRank(λ = λ, window = win, weights = weight), x, γ)

        H = AbstractOperators.Hankel(ComplexF64, (gx, gy), win; nchannels = nc, channels = true)
        W = reshape(weight, gx, gy, 1)
        F = svd(H * (W .* x))
        S = max.(0.0, F.S .- λ * γ)
        adjoint_lift = H' * (F.U * Diagonal(S) * F.Vt)
        mult = real.(AbstractOperators.diag_AcA(H))
        # `(𝓗∘diag(w))ᴴ(𝓗∘diag(w)) = diag(|w|² ⊙ mult)`
        @test y ≈ conj.(W) .* adjoint_lift ./ (abs2.(W) .* mult)
        # and the value is the nuclear norm of the *weighted* lift
        @test calculate(StructuredLowRank(λ = λ, window = win, weights = weight), x; threaded = false) ≈
            λ * sum(svdvals(H * (W .* x)))
    end

    @testset "a vanishing weight leaves its sample alone" begin
        gx, gy, nc = 12, 10, 2
        win = (4, 3)
        x = randn(ComplexF64, gx, gy, nc)
        weight = ones(ComplexF64, gx, gy)
        weight[3, 4] = 0
        weight[7, 1] = 0
        y, _ = prox_of(StructuredLowRank(λ = 0.15, window = win, weights = weight), x, 0.7)
        # the weighted term says nothing about a sample it zeroes out, so the prox is the identity
        # there -- not zero, which is what the raw least-squares formula would give
        @test y[3, 4, :] ≈ x[3, 4, :]
        @test y[7, 1, :] ≈ x[7, 1, :]
        @test !isapprox(y[5, 5, :], x[5, 5, :])
    end

    @testset "the ALOHA structure: weighting drops the lifted rank" begin
        # A step image has a two-spike x-difference, so the x-weighted k-space is annihilated by a
        # short filter and its block-Hankel lift is rank-deficient -- while the unweighted lift of
        # the same single-channel k-space is not. That gap is what `weights` exploits.
        N = 32
        img = zeros(ComplexF64, N, N)
        img[10:20, :] .= 1.0
        ksp = reshape(fftshift(fft(img)) / N, N, N, 1)
        H = AbstractOperators.Hankel(ComplexF64, (N, N), (5, 5); nchannels = 1, channels = true)
        w = MriReconstructionToolbox._slr_weights(
            StructuredLowRank(λ = 0.1, window = (5, 5), weights = :tv), (N, N), ComplexF64
        )[2]                                             # the x first difference
        numrank(σ) = count(>(1.0e-8 * σ[1]), σ)
        @test numrank(svdvals(H * ksp)) == 25            # full: 5 × 5 window, one channel
        @test numrank(svdvals(H * (w .* ksp))) < 15
    end

    @testset "calculate: the rank form is an indicator" begin
        # a single 2D complex exponential per coil lifts to a rank-1 block-Hankel matrix
        gx, gy, nc = 12, 10, 2
        w = (4, 3)
        x = ComplexF64[cispi(0.13 * (i - 1) + 0.21 * (j - 1)) * (c + 1) for i in 1:gx, j in 1:gy, c in 1:nc]
        @test calculate(StructuredLowRank(max_rank = 1, window = w), x; threaded = false) == 0
        noisy = x .+ 0.5 .* randn(ComplexF64, gx, gy, nc)
        @test calculate(StructuredLowRank(max_rank = 1, window = w), noisy; threaded = false) == Inf
    end
end

@testitem "StructuredLowRank end-to-end calibrationless" tags = [:regularization, :reconstruction, :integration] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using MriReconstructionToolbox
    using LinearAlgebra, NamedDims
    import Random
    Random.seed!(1)

    Nx, Ny, Nc = 24, 24, 6
    img = zeros(ComplexF64, Nx, Ny)
    img[7:18, 7:18] .= 1.0
    img[10:14, 10:14] .= 0.5
    sens = NamedDimsArray{(:x, :y, :coil)}(ComplexF64.(coil_sensitivities(Nx, Ny, Nc)))
    patt = create_sampling_pattern(UniformRandomSampling(2.0, 0.15), (Nx, Ny))

    # Simulate through the package's own forward model so that the amplitude convention of the
    # k-space matches what `reconstruct` inverts.
    acq_sim = CartesianAcquisitionInfo(;
        is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens, subsampling = patt,
    )
    ksp = simulate_acquisition(NamedDimsArray{(:x, :y)}(img), acq_sim).kspace_data

    # The reconstruction gets the coil data and nothing else: no sensitivity maps, no ACS
    # kernel. That is what makes this calibrationless.
    acq = CartesianAcquisitionInfo(
        ksp; is3D = false, image_size = (Nx, Ny), subsampling = patt,
    )

    rss(x) = sqrt.(sum(abs2, x; dims = 3))[:, :, 1]
    nrmse(a, b) = norm(a - b) / norm(b)
    # Reference magnitude and the zero-filled baseline, both in the package's own scale.
    acq_full = CartesianAcquisitionInfo(;
        is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens,
    )
    ksp_full = simulate_acquisition(NamedDimsArray{(:x, :y)}(img), acq_full).kspace_data
    truth = rss(
        abs.(
            unname(
                reconstruct(
                    CartesianAcquisitionInfo(ksp_full; is3D = false, image_size = (Nx, Ny)),
                    DirectReconstruction(); verbosity = Silent(),
                )
            )
        )
    )
    zf = rss(abs.(unname(reconstruct(acq, DirectReconstruction(); verbosity = Silent()))))

    solve(reg) = abs.(
        unname(
            reconstruct(
                acq,
                IterativeReconstruction(
                    reg; signal_model = KSpaceToImage(RootSumSquares()), algorithm = ADMM(), maxit = 100,
                );
                verbosity = Silent(),
            )
        )
    )

    e_zf = nrmse(zf, truth)
    # LORAKS-C: the convex nuclear-norm form
    e_nuc = nrmse(solve(StructuredLowRank(λ = 0.03, window = (5, 5))), truth)
    @test e_nuc < 0.08
    @test e_nuc < e_zf
    # SAKE: the non-convex hard-rank form, from the same zero-filled starting point
    e_rank = nrmse(solve(StructuredLowRank(max_rank = 30, window = (5, 5))), truth)
    @test e_rank < 0.08
    @test e_rank < e_zf
end

@testitem "LORAKS S- and G-matrix lifts" tags = [:regularization] setup = [RegTestSetup, ProxOf] begin
    using LinearAlgebra
    using FFTW: fft, fftshift, ifftshift
    import Random
    Random.seed!(0)
    MRT = MriReconstructionToolbox

    function lift_matrix(lift, x, st)
        M = Array{real(eltype(x))}(undef, MRT._loraks_matrix_size(lift, Val(st))...)
        return MRT._loraks_lift!(M, lift, x, Val(st))
    end
    lift_rows(lift) = CartesianIndices(lift.nwin)

    @testset "geometry" begin
        lift = MRT.LoraksLift((13, 11), (4, 3), 2, :s)
        @test lift.center == (7, 6)                             # MRT's centered k-space default
        @test lift.nwin == (10, 9)                              # every window position is a row
        @test MRT._loraks_matrix_size(lift, Val(:s)) == (2 * 90, 2 * 12 * 2)
        @test MRT._loraks_matrix_size(lift, Val(:g)) == (2 * 90, 2 * 12 * 2 + 2)
        # the reflection is `-ν mod N` about the centre, so DC is its own partner
        @test MRT._loraks_reflect(lift, CartesianIndex(7, 6)) == CartesianIndex(7, 6)
        @test MRT._loraks_reflect(lift, CartesianIndex(8, 7)) == CartesianIndex(6, 5)
        @test MRT._loraks_reflect(lift, CartesianIndex(1, 1)) == CartesianIndex(13, 11)
        # an explicit centre is honoured; a centre off the grid, a window that does not fit and
        # an unknown structure are rejected
        @test MRT.LoraksLift((13, 11), (4, 3), 1, :s; center = (1, 1)).center == (1, 1)
        @test_throws ArgumentError MRT.LoraksLift((13, 11), (4, 3), 1, :s; center = (0, 5))
        @test_throws ArgumentError MRT.LoraksLift((13, 11), (20, 3), 1, :s)
        @test_throws ArgumentError MRT.LoraksLift((13, 11), (4, 3), 1, :c)
    end

    @testset "the lift is a weighted tight frame ($st, grid $gs)" for st in (:s, :g),
            gs in ((12, 10), (13, 11), (8, 7, 6))

        ks = length(gs) == 2 ? (4, 3) : (3, 3, 2)
        nch = 2
        lift = MRT.LoraksLift(gs, ks, nch, st)
        x = randn(ComplexF64, gs..., nch)
        M = lift_matrix(lift, x, st)
        @test size(M, 1) == 2 * length(lift_rows(lift))

        y = similar(x)
        MRT._loraks_unlift!(y, lift, M, Val(st))
        mult = MRT._loraks_multiplicity(Float64, lift, Val(st))
        # `LᴴL` is the real diagonal `mult` -- the sign pattern of both constructions makes every
        # cross term between the two sides of k-space cancel. That is what makes the
        # multiplicity-weighted adjoint an exact left inverse, and the Cadzow prox exact.
        @test y ≈ mult .* x
        # with the modular reflection every sample is in the lift, so there is no sample the
        # prox has to hand through unchanged
        @test all(>(0), mult)

        # and `_loraks_unlift!` really is the *real* adjoint of the lift: ⟨D, L x⟩ = ⟨Lᴴ D, x⟩
        # over the real inner product, which is the one the lift is linear in.
        D = randn(size(M)...)
        MRT._loraks_unlift!(y, lift, D, Val(st))
        @test dot(D, M) ≈ real(dot(y, x))
    end

    @testset "the S matrix matches Haldar (2014) Eqs. 3-6 and 22" begin
        gs, ks, nch = (13, 11), (4, 3), 2
        lift = MRT.LoraksLift(gs, ks, nch, :s)
        x = randn(ComplexF64, gs..., nch)
        rows, offs = lift_rows(lift), CartesianIndices(ks)
        K, nblock = length(rows), prod(ks) * nch
        # Written out in the paper's own indexing: rows are neighbourhood centres ν, columns are
        # the in-window offsets p, and the four sub-blocks hold the real and imaginary parts of
        # k(ν - p) and k(-ν - p).
        at(coord) = mod1.(coord .+ lift.center, gs)              # `ν ↦ array index`, modulo the grid
        Sr₊, Si₊, Sr₋, Si₋ = (zeros(K, nblock) for _ in 1:4)
        for c in 1:nch, (jk, ko) in enumerate(offs), (jw, wi) in enumerate(rows)
            ν = Tuple(wi) .- lift.center
            p = .-(Tuple(ko) .- 1)
            col = (c - 1) * prod(ks) + jk
            zp = x[at(ν .- p)..., c]
            zm = x[at(.-ν .- p)..., c]
            Sr₊[jw, col], Si₊[jw, col] = real(zp), imag(zp)
            Sr₋[jw, col], Si₋[jw, col] = real(zm), imag(zm)
        end
        @test lift_matrix(lift, x, :s) ≈ [(Sr₊ .- Sr₋) (Si₋ .- Si₊); (Si₊ .+ Si₋) (Sr₊ .+ Sr₋)]
    end

    @testset "the G matrix matches Haldar (2014) Eqs. 17-21" begin
        gs, ks, nch = (13, 11), (4, 3), 2
        lift = MRT.LoraksLift(gs, ks, nch, :g)
        x = randn(ComplexF64, gs..., nch)
        rows, offs = lift_rows(lift), CartesianIndices(ks)
        K, nblock = length(rows), prod(ks) * nch
        at(coord) = mod1.(coord .+ lift.center, gs)
        gr, gi = zeros(K, nch), zeros(K, nch)
        Gr, Gi = zeros(K, nblock), zeros(K, nblock)
        for c in 1:nch
            for (jw, wi) in enumerate(rows)
                ν = Tuple(wi) .- lift.center
                z = x[at(.-ν)..., c]                     # the reflected sample, k(-ν)
                gr[jw, c], gi[jw, c] = real(z), imag(z)
            end
            for (jk, ko) in enumerate(offs), (jw, wi) in enumerate(rows)
                ν = Tuple(wi) .- lift.center
                p = .-(Tuple(ko) .- 1)
                z = x[at(ν .- p)..., c]
                Gr[jw, (c - 1) * prod(ks) + jk] = real(z)
                Gi[jw, (c - 1) * prod(ks) + jk] = imag(z)
            end
        end
        # `g` is one column per channel, which is the P-LORAKS channel-stacking applied to a
        # construction the papers only write down for a single channel.
        @test lift_matrix(lift, x, :g) ≈ [(-gr) Gr (-Gi); gi Gi Gr]
    end

    @testset "the S matrix sees conjugate symmetry" begin
        N = 32
        xs = range(-1, 1, N)
        supp = [(abs(u) < 0.3 && abs(v) < 0.3) ? 1.0 : 0.0 for u in xs, v in xs]
        ksp(im_) = reshape(fftshift(fft(ifftshift(im_))) / N, N, N, 1)
        lift = MRT.LoraksLift((N, N), (5, 5), 1, :s)
        σ_of(im_) = svdvals(lift_matrix(lift, ksp(im_), :s))
        H = AbstractOperators.Hankel(ComplexF64, (N, N), (5, 5); nchannels = 1, channels = true)
        numrank(s) = count(>(1.0e-3 * s[1]), s)

        Random.seed!(3)
        rough = supp .* cispi.(2 .* rand(N, N))
        σ_real, σ_rough = σ_of(complex(supp)), σ_of(rough)
        # A real-valued image has an exactly conjugate-symmetric k-space, and exposing that is
        # what the S matrix is for: it acquires a numerical null space...
        @test σ_real[end] / σ_real[1] < 1.0e-12
        # ...that a random-phase image of the same support does not give it, where S is no more
        # deficient than the two copies of the C matrix it is built from.
        @test numrank(σ_real) < numrank(σ_rough)
        @test σ_rough[end] / σ_rough[1] > 1.0e-8
        # the C matrix cannot tell the two images apart at all: it models support, not phase
        @test numrank(svdvals(H * ksp(complex(supp)))) == numrank(svdvals(H * ksp(rough)))
    end

    @testset "the prox is one Cadzow step of the $st matrix" for st in (:s, :g)
        gs, ks, nch = (14, 12), (4, 3), 2
        x = randn(ComplexF64, gs..., nch)
        γ, λ = 0.7, 0.15
        reg = StructuredLowRank(λ = λ, window = ks, structure = st)
        y, value = prox_of(reg, x, γ)

        lift = MRT.LoraksLift(gs, ks, nch, st)
        M = lift_matrix(lift, x, st)
        F = svd(M)
        Sσ = max.(0.0, F.S .- λ * γ)
        ref = similar(x)
        MRT._loraks_unlift!(ref, lift, F.U * Diagonal(Sσ) * F.Vt, Val(st))
        mult = MRT._loraks_multiplicity(Float64, lift, Val(st))
        @test y ≈ ref ./ mult
        @test value ≈ λ * sum(Sσ)
        @test calculate(reg, x; threaded = false) ≈ λ * sum(svdvals(M))
    end

    @testset "the rank form truncates, and threading does not change it" begin
        gs, ks, nch = (16, 14), (5, 4), 2
        x = randn(ComplexF64, gs..., nch, 2)                    # two batch slabs
        reg = StructuredLowRank(max_rank = 8, window = ks, structure = :s)
        seq, value = prox_of(reg, x, 1.0)
        @test value == 0                                        # an indicator, zero where projected

        lift = MRT.LoraksLift(gs, ks, nch, :s)
        ref = similar(x)
        mult = MRT._loraks_multiplicity(Float64, lift, Val(:s))
        for b in 1:2
            F = svd(lift_matrix(lift, x[:, :, :, b], :s))
            Sσ = copy(F.S)
            Sσ[9:end] .= 0                                      # rank 8, the hard-truncation form
            slab = similar(x, gs..., nch)
            MRT._loraks_unlift!(slab, lift, F.U * Diagonal(Sσ) * F.Vt, Val(:s))
            ref[:, :, :, b] = slab ./ mult
        end
        @test seq ≈ ref

        term = materialize(reg, Variable(x); threaded = true)
        f = MriReconstructionToolbox.StructuredOptimization.weighted_function(term)
        par = similar(x)
        MriReconstructionToolbox.ProximalCore.prox!(par, f, x, 1.0)
        @test par ≈ seq
        # the Cadzow step truncates the *lifted* matrix, so a generic iterate stays infeasible
        @test calculate(reg, x; threaded = false) == Inf
    end
end

@testitem "LORAKS S-matrix partial-Fourier reconstruction" tags = [:regularization, :reconstruction, :integration] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using MriReconstructionToolbox
    using LinearAlgebra, NamedDims

    # Single-channel partial Fourier: the phase constraints are the one structured-low-rank
    # prior that works without several coils, which is LORAKS' original selling point.
    N = 32
    g = range(-1, 1, N)
    img = ComplexF64[
        ((abs(x) < 0.45 && abs(y) < 0.45) ? (abs(x) < 0.2 && abs(y) < 0.2 ? 0.6 : 1.0) : 0.0) *
            cis(0.9 * (x + 0.5y)) for x in g, y in g
    ]
    mask = falses(N)
    mask[1:20] .= true                                          # 62 % of ky, from one side
    ksp_full = simulate_acquisition(
        NamedDimsArray{(:x, :y)}(img), CartesianAcquisitionInfo(; is3D = false, image_size = (N, N))
    ).kspace_data
    acq = CartesianAcquisitionInfo(
        ksp_full[:, mask]; is3D = false, image_size = (N, N), subsampling = (:, mask),
    )

    nrmse(a, b) = norm(abs.(a) .- abs.(b)) / norm(abs.(b))
    solve(reg) = unname(
        reconstruct(
            acq,
            IterativeReconstruction(
                reg; signal_model = KSpaceToImage(RootSumSquares()), algorithm = ADMM(), maxit = 120,
            );
            verbosity = Silent(),
        )
    )

    e_zf = nrmse(unname(reconstruct(acq, DirectReconstruction(); verbosity = Silent())), img)
    # The S matrix is the strong one, and it holds across λ: 0.123-0.136 against 0.182 zero-filled
    # over λ ∈ [0.005, 0.08].
    e_s = nrmse(solve(StructuredLowRank(λ = 0.02, window = (5, 5), structure = :s)), img)
    @test e_s < 0.85 * e_zf
    # The G matrix helps too, but it is the weaker construction (the paper expects it to be merely
    # rank-deficient), and its λ has to be picked rather than trusted: the same sweep runs
    # 0.132-0.173, non-monotonically.
    e_g = nrmse(solve(StructuredLowRank(λ = 0.01, window = (5, 5), structure = :g)), img)
    @test e_g < 0.9 * e_zf
end
