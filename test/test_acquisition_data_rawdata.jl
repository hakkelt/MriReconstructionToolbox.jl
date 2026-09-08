@testsnippet RawAcqHelpers begin
    using MRIBase: RawAcquisitionData, Profile, AcquisitionHeader, EncodingCounters, Limit

    function make_profile(
            data::Matrix{ComplexF32}; step1 = 0, step2 = 0, slice = 0, contrast = 0, phase = 0,
            repetition = 0, set = 0, average = 0, discard_pre = 0, discard_post = 0, center_sample = 0,
        )
        ncoil = size(data, 2)
        head = AcquisitionHeader(;
            number_of_samples = UInt16(size(data, 1)),
            available_channels = UInt16(ncoil),
            active_channels = UInt16(ncoil),
            discard_pre = UInt16(discard_pre),
            discard_post = UInt16(discard_post),
            center_sample = UInt16(center_sample),
            idx = EncodingCounters(;
                kspace_encode_step_1 = UInt16(step1),
                kspace_encode_step_2 = UInt16(step2),
                slice = UInt16(slice),
                contrast = UInt16(contrast),
                phase = UInt16(phase),
                repetition = UInt16(repetition),
                set = UInt16(set),
                average = UInt16(average),
            ),
        )
        return Profile(head, zeros(Float32, 1, 1), data)
    end

    # A minimal non-Cartesian ("custom" trajectory) profile: MRIBase's own `trajectory(f)` needs
    # a nonzero `sample_time_us` (it divides by it) and `trajectory_dimensions` set from `traj`.
    function make_traj_profile(
            data::Matrix{ComplexF32}, traj::Matrix{Float32}; slice = 0, contrast = 0, repetition = 0,
        )
        ncoil = size(data, 2)
        head = AcquisitionHeader(;
            number_of_samples = UInt16(size(data, 1)),
            available_channels = UInt16(ncoil),
            active_channels = UInt16(ncoil),
            trajectory_dimensions = UInt16(size(traj, 1)),
            sample_time_us = 5.0f0,
            idx = EncodingCounters(;
                slice = UInt16(slice), contrast = UInt16(contrast), repetition = UInt16(repetition),
            ),
        )
        return Profile(head, traj, data)
    end

    function make_raw(profiles; encoded_size, lim1, lim2 = Limit(0, 0, 0), trajectory = "cartesian")
        params = Dict{String, Any}(
            "encodedSize" => collect(encoded_size),
            "trajectory" => trajectory,
            "enc_lim_kspace_encoding_step_1" => lim1,
            "enc_lim_kspace_encoding_step_2" => lim2,
        )
        return RawAcquisitionData(params, profiles)
    end
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — Cartesian" tags = [:acquisition] setup = [RawAcqHelpers] begin
    using MriReconstructionToolbox
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using NamedDims: dimnames, unname

    @testset "asymmetric readout + partial-Fourier ky + multi-slice (:z)" begin
        # encodedSize = (10, 8); readout: 8 acquired samples, center_sample=3 (0-based) is off
        # from the geometric center (5), so placing them must shift by `10÷2 - 3 = 2` — this is
        # exactly the FFT-shift bug the constructor exists to avoid (see its docstring).
        # ky: only steps 0:5 of 0:7 are present (partial Fourier), and encoding_limits.center=2
        # is likewise off from the geometric center (4).
        ncoil = 2
        profiles = Profile[]
        for slice in (0, 1), step1 in 0:5
            data = ComplexF32[1000 * slice + 10 * step1 + i + 100im * c for i in 1:8, c in 1:ncoil]
            push!(profiles, make_profile(data; step1, slice, center_sample = 3))
        end
        raw = make_raw(profiles; encoded_size = (10, 8, 1), lim1 = Limit(0, 7, 2))

        info = AcquisitionInfo(raw)
        @test info isa CartesianAcquisitionInfo
        @test info.is3D == false
        @test info.image_size == (10, 8)
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil, :z)
        @test size(info.kspace_data) == (8, 6, ncoil, 2)
        @test info.subsampling == (3:10, 3:8)

        ksp = unname(info.kspace_data)
        @test ksp[:, 1, :, 1] == ComplexF32[i + 100im * c for i in 1:8, c in 1:ncoil] # slice=0, step1=0
        @test ksp[:, 6, :, 2] == ComplexF32[1000 + 50 + i + 100im * c for i in 1:8, c in 1:ncoil] # slice=1, step1=5
    end

    @testset "fully sampled -> Colon()/nothing subsampling, single slice" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, center_sample = 2)
                for step1 in 0:3
        ]
        raw = make_raw(profiles; encoded_size = (4, 4, 1), lim1 = Limit(0, 3, 2))

        info = AcquisitionInfo(raw)
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil)
        @test size(info.kspace_data) == (4, 4, 1)
        @test isnothing(info.subsampling)
        @test info.image_size == (4, 4)
        @test info.shifted_image_dims == (:x, :y)
    end

    @testset "3D sets shifted_image_dims on all three spatial axes" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, step2, center_sample = 2)
                for step1 in 0:3, step2 in 0:2
        ]
        raw = make_raw(
            vec(profiles); encoded_size = (4, 4, 3), lim1 = Limit(0, 3, 2), lim2 = Limit(0, 2, 1)
        )
        @test AcquisitionInfo(raw).shifted_image_dims == (:x, :y, :z)
    end

    @testset "irregular undersampled ky -> boolean mask" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, center_sample = 2)
                for step1 in (0, 2, 4, 6)
        ]
        raw = make_raw(profiles; encoded_size = (4, 8, 1), lim1 = Limit(0, 7, 4))

        info = AcquisitionInfo(raw)
        @test size(info.kspace_data) == (4, 4, 1)
        @test info.subsampling[1] == Colon()
        @test info.subsampling[2] == Bool[1, 0, 1, 0, 1, 0, 1, 0]
    end

    @testset "cardiac phase becomes the :time batch dimension" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, phase, center_sample = 2)
                for step1 in 0:3, phase in 0:2
        ]
        raw = make_raw(vec(profiles); encoded_size = (4, 4, 1), lim1 = Limit(0, 3, 2))

        info = AcquisitionInfo(raw)
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil, :time)
        @test size(info.kspace_data) == (4, 4, 1, 3)
    end

    @testset "3D acquisition (kspace_encode_step_2 varies)" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, step2, center_sample = 2)
                for step1 in 0:3, step2 in 0:2
        ]
        raw = make_raw(
            vec(profiles); encoded_size = (4, 4, 3), lim1 = Limit(0, 3, 2), lim2 = Limit(0, 2, 1)
        )

        info = AcquisitionInfo(raw)
        @test info.is3D == true
        @test dimnames(info.kspace_data) == (:kx, :ky, :kz, :coil)
        @test size(info.kspace_data) == (4, 4, 3, 1)
        @test info.image_size == (4, 4, 3)
    end

    @testset "multi-slab 3D (is3D and slice both vary) is rejected, not silently dropped" begin
        profiles = Profile[
            make_profile(ComplexF32[i + c * 1im for i in 1:4, c in 1:1]; step1, step2, slice, center_sample = 2)
                for step1 in 0:3, step2 in 0:2, slice in 0:1
        ]
        raw = make_raw(
            vec(profiles); encoded_size = (4, 4, 3), lim1 = Limit(0, 3, 2), lim2 = Limit(0, 2, 1)
        )
        @test_throws ArgumentError AcquisitionInfo(raw)
    end
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — object stays centred in the FOV" tags = [:acquisition, :reconstruction] setup = [RawAcqHelpers] begin
    using MriReconstructionToolbox
    using NamedDims: unname
    using FFTW: fft, fftshift

    # A scanner images an object centred in the FOV and stores k-space with DC at the centre.
    # MRT's plain-DFT default puts the image origin at index 1, so without `shifted_image_dims`
    # the reconstruction of such data comes out rolled by half the FOV along every spatial axis
    # (the whole object lands in the four corners). The constructor must set it for us.
    n = 16
    img = zeros(ComplexF32, n, n)
    img[6:9, 7:10] .= 1              # an off-centre blob, so a half-FOV roll is unambiguous
    img[7, 8] = 3
    ksp_true = fftshift(fft(img))    # DC at index n ÷ 2 + 1 = 9, as ISMRMRD stores it

    profiles = Profile[
        make_profile(ComplexF32.(reshape(ksp_true[:, j], n, 1)); step1 = j - 1, center_sample = n ÷ 2)
            for j in 1:n
    ]
    raw = make_raw(profiles; encoded_size = (n, n, 1), lim1 = Limit(0, n - 1, n ÷ 2))

    info = AcquisitionInfo(raw)
    @test info.shifted_image_dims == (:x, :y)

    rec = abs.(unname(reconstruct(info; verbosity = Silent()))[:, :, 1])
    @test Tuple(argmax(rec)) == (7, 8)                       # not (15, 16), the half-FOV-rolled peak
    @test rec ≈ (rec[7, 8] / 3) .* abs.(img)                 # whole image, not just the peak
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — non-Cartesian dispatch" tags = [:acquisition, :nfft] setup = [RawAcqHelpers] begin
    using MriReconstructionToolbox
    using MriReconstructionToolbox: NonCartesianAcquisitionInfo

    nsamp, ncoil = 5, 1
    profiles = Profile[]
    for k in 0:3
        traj = Float32.(hcat(range(-0.4f0, 0.4f0; length = nsamp), fill(Float32(k) / 10 - 0.2f0, nsamp))')
        data = ComplexF32.(reshape(1:nsamp, nsamp, 1) .+ 0im)
        push!(profiles, make_traj_profile(data, traj))
    end
    raw = make_raw(profiles; encoded_size = (8, 8, 1), lim1 = Limit(0, 0, 0), trajectory = "custom")

    info = AcquisitionInfo(raw)
    @test info isa NonCartesianAcquisitionInfo
    @test info.is3D == false
    @test info.image_size == (8, 8)
    @test size(info.trajectory) == (2, length(profiles) * nsamp)
    @test size(info.kspace_data) == (length(profiles) * nsamp, ncoil)
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — real M4Raw data" tags = [:acquisition, :integration] begin
    using MriReconstructionToolbox
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using NamedDims: dimnames, unname
    using MRITestData
    using MRIBase

    if MRITestData.get_download_path() === nothing
        MRITestData.set_download_path!(:cache)
    end
    entry = MRITestData.dataset(MRITestData.M4RAW, "multicoil_train/2022062402_T203"; offline = true)
    if !MRITestData.is_cached(entry)
        @test_skip "M4Raw sample not cached locally; not downloading it for this test"
    else
        raw = MRITestData.load_raw(entry)
        info = AcquisitionInfo(raw)
        @test info isa CartesianAcquisitionInfo
        @test info.is3D == false
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil, :z)
        @test size(info.kspace_data) == (256, 256, 4, 18)
        @test isnothing(info.subsampling) # M4Raw stores every encoded ky line (some are all-zero)
        @test info.shifted_image_dims == (:x, :y)

        # Must reproduce the notebook's hand-rolled single-slice assembly exactly (no extra
        # `fftshift` needed — see the constructor's docstring on the FFT-shift convention).
        function assemble_slice(raw, slice)
            profiles = [
                p for p in raw.profiles if
                    Int(p.head.idx.slice) == slice && Int(p.head.idx.contrast) == 0 &&
                    Int(p.head.idx.repetition) == 0 && Int(p.head.idx.average) == 0
            ]
            nsamples, ncoils = size(profiles[1].data)
            pre, post = Int(profiles[1].head.discard_pre), Int(profiles[1].head.discard_post)
            nkx = nsamples - pre - post
            ksp = zeros(ComplexF32, nkx, 256, ncoils)
            for p in profiles
                ksp[:, Int(p.head.idx.kspace_encode_step_1) + 1, :] .= ComplexF32.(p.data[(pre + 1):(pre + nkx), :])
            end
            return ksp
        end
        manual = assemble_slice(raw, 5)
        @test unname(info.kspace_data)[:, :, :, 6] == manual # slice id 5 -> compact index 6
    end
end

@testitem "AcquisitionInfo(::MRIBase.RawAcquisitionData) — real OCMR data (asymmetric-echo readout)" tags = [:acquisition, :integration] begin
    using MriReconstructionToolbox
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using NamedDims: dimnames
    using MRITestData
    using MRIBase

    if MRITestData.get_download_path() === nothing
        MRITestData.set_download_path!(:cache)
    end
    entry = MRITestData.dataset(MRITestData.OCMR_SOURCE, "fs_0001_1_5T"; offline = true)
    if !MRITestData.is_cached(entry)
        @test_skip "OCMR sample not cached locally; not downloading it for this test"
    else
        raw = MRITestData.load_raw(entry)
        info = AcquisitionInfo(raw)
        @test info isa CartesianAcquisitionInfo
        @test dimnames(info.kspace_data) == (:kx, :ky, :coil, :time)
        @test info.image_size == (512, 208)
        @test size(info.kspace_data) == (404, 208, 15, 19)
        # Asymmetric-echo readout: only a contiguous block of the 512-sample encoded readout was
        # acquired, recentered via `head.center_sample` (148) rather than the naive `row + 1`.
        @test info.subsampling[1] == 109:512
        @test info.subsampling[2] == Colon()
    end
end
