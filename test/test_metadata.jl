using TestItems

@testitem "Header: known keys are fields, any other key is kept" tags = [:acquisition] begin
    using Test
    using Ristretto
    using Ristretto: header, image_size

    h = Header(; fov = (240, 180), TE = 4.2, protocol = "t1_se")
    @test h isa AbstractDict{Symbol, Any}
    @test h[:fov] === (240.0, 180.0)
    @test h.TE === 4.2
    @test h[:protocol] == "t1_se"
    @test isnothing(h.offset)
    @test !haskey(h, :offset)
    @test_throws KeyError h[:offset]
    @test get(h, :offset, 0) == 0
    @test Set(keys(h)) == Set((:fov, :TE, :protocol))
    @test length(h) == 3
    h[:offset] = [1, 2, 3]
    @test h.offset === (1.0, 2.0, 3.0)
    h.TR = 10
    @test h[:TR] === 10.0
    h[:site] = "A"
    @test h.extra[:site] == "A"
    delete!(h, :TR)
    @test isnothing(h.TR)
    @test_throws Exception h.protocl
    @test_throws ArgumentError h.protocol = "x"
    @test_throws ArgumentError Header(; orientation = rand(2, 2))
    @test_throws ArgumentError Header(; offset = (1, 2))
    @test_throws ArgumentError Header(; fov = (1,))
    @test Header(Dict("TE" => 2)) == Header(; TE = 2)
    @test occursin("fov = (240.0, 180.0)", sprint(show, MIME"text/plain"(), h))

    hc = copy(h)
    hc.tags["x"] = 1
    hc[:site] = "B"
    @test isempty(h.tags)
    @test h[:site] == "A"
end

@testitem "Header on an acquisition: optional, stored as given, shared by copies" tags = [:acquisition] begin
    using Test
    using Ristretto
    using Ristretto: header, image_size

    acq = AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false)
    @test image_size(acq) == (16, 12)
    @test header(acq) isa Header
    @test isempty(header(acq))

    acq = AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false, header = (; fov = (240, 180), protocol = "t1_se"))
    @test header(acq).fov == (240.0, 180.0)
    @test header(acq)[:protocol] == "t1_se"

    given = Header(; fov = (160, 120))
    acq = AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false, header = given)
    @test header(acq) === given

    # A mismatch between fov, spacing and the image size is warned about, not rejected.
    @test_logs (:warn, r"fov") AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false, header = (; fov = (1, 2), spacing = (1, 1)))
    @test_logs (:warn, r"entries") AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false, header = (; fov = (1, 2, 3)))

    settag!(acq, :subject, "s1")
    acq2 = AcquisitionInfo(acq; sensitivity_maps = nothing)
    @test header(acq2) === header(acq)
    @test gettag(acq2, :subject) == "s1"

    acq3 = AcquisitionInfo(acq; header = (; TR = 10))
    @test header(acq3).TR == 10
    @test isnothing(header(acq3).fov)
end

@testitem "Tags" tags = [:acquisition] begin
    using Test
    using Ristretto

    acq = AcquisitionInfo(rand(ComplexF32, 8, 8); is3D = false)
    @test isempty(tags(acq))
    @test settag!(acq, :site, "A") === acq
    @test gettag(acq, :site) == "A"
    @test gettag(acq, "site") == "A"
    @test gettag(acq, :missing, 0) == 0
    @test_throws KeyError gettag(acq, :missing)
    @test tags(acq) == Dict("site" => "A")
end

@testitem "reconstruct returns a ReconImage with the acquisition's header" tags = [:reconstruction] begin
    using Test
    using Ristretto
    using Ristretto: header, image_size
    using NamedDims

    ksp = NamedDimsArray{(:kx, :ky, :z)}(rand(ComplexF32, 16, 16, 3))
    acq = AcquisitionInfo(ksp; is3D = false, header = (; fov = (160, 160), slice_spacing = 5))
    settag!(acq, :subject, "s1")
    img = reconstruct(acq)
    @test img isa ReconImage
    @test parent(img) isa NamedDimsArray
    @test dimnames(img) == (:x, :y, :z)
    @test image_size(img) == (16, 16)
    @test header(img).spacing == (10.0, 10.0)
    @test isnothing(header(acq).spacing)   # derived on the image's copy only
    @test gettag(img, :subject) == "s1"

    # The image's header is a copy: tagging it leaves the acquisition alone.
    settag!(img, :subject, "other")
    @test gettag(acq, :subject) == "s1"

    @test Array(img) isa Array{ComplexF32, 3}
    @test unname(img) isa Array{ComplexF32, 3}
    @test img .* 2 ≈ 2 .* Array(img)
end

@testitem "ReconImage: keyword indexing moves the offset" tags = [:reconstruction] begin
    using Test
    using Ristretto
    using Ristretto: header, image_size
    using NamedDims

    R = [0.0 0 1; 1 0 0; 0 1 0]          # x → L-P-S column 1 = (0,1,0), ...
    h = Header(; fov = (16, 12, 8), spacing = (2, 2, 2), orientation = R, offset = (10, 20, 30))
    img = ReconImage(NamedDimsArray{(:x, :y, :z, :time)}(rand(8, 6, 4, 3)), h)

    crop = img[x = 3:6]
    @test crop isa ReconImage
    @test size(crop) == (4, 6, 4, 3)
    @test image_size(crop) == (4, 6, 4)
    @test header(crop).fov == (8.0, 12.0, 8.0)
    @test collect(header(crop).offset) ≈ [10, 20, 30] .+ R[:, 1] .* (2 * 2.0)

    slice = img[z = 3]
    @test image_size(slice) == (8, 6)
    @test header(slice).spacing == (2.0, 2.0)
    @test header(slice).slice_thickness == 2.0
    @test collect(header(slice).offset) ≈ [10, 20, 30] .+ R[:, 3] .* (2 * 2.0)

    # Dropping x keeps the remaining in-plane axes first in the orientation.
    sag = img[x = 2]
    @test header(sag).orientation[:, 1] == R[:, 2]
    @test header(sag).orientation[:, 3] == R[:, 1]

    # Non-spatial axes leave the geometry alone; views work the same way.
    frame = view(img; time = 2)
    @test frame isa ReconImage
    @test header(frame).offset == h.offset
    @test header(img).offset == (10.0, 20.0, 30.0)   # slicing never changes the original
    @test image_size(frame) == (8, 6, 4)

    # A 2D multi-slice image: `z` moves the offset by the slice spacing.
    h2 = Header(; spacing = (1, 1), slice_spacing = 5, orientation = R, offset = (0, 0, 0))
    ms = ReconImage(NamedDimsArray{(:x, :y, :z)}(rand(8, 6, 4)), h2)
    @test image_size(ms) == (8, 6)
    @test collect(header(ms[z = 4]).offset) ≈ R[:, 3] .* 15
    @test image_size(ms[z = 4]) == (8, 6)
end

@testitem "ReconImage moves to a device with its header" tags = [:gpu, :reconstruction] setup = [GpuEnvSetup, GpuHelpers] begin
    using Test
    using Ristretto
    using Ristretto: header
    using NamedDims

    acq = AcquisitionInfo(NamedDimsArray{(:kx, :ky)}(rand(ComplexF32, 16, 16)); is3D = false, header = (; fov = (160, 160)))
    test_on_devices(acq; rtol = 1.0e-4) do a
        img = reconstruct(a)
        @test img isa ReconImage
        @test header(img).fov == (160.0, 160.0)
        img
    end
end
