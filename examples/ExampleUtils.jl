"""
Helpers shared by the scripts under `examples/`.

Every example follows the same shape: pick one dataset from the `MRITestData` catalog,
download it, load it as a `RawAcquisitionData`, build an `AcquisitionInfo`, reconstruct, and
write a magnitude image. The functions here take care of everything that is not specific to
one source, so that each script shows only what is characteristic of its data type.

Images are written as 8-bit binary PGM (`P5`), which every image viewer and `ImageMagick`
reads and which needs no extra dependency.
"""
module ExampleUtils

using MRITestData
using MRIBase
using MriReconstructionToolbox
using Printf
using Statistics

export setup, load_example, first_slab, describe, save_image, report, noise_covariance

"""
    setup()

Accept the terms notice and make sure downloads have a destination. Call it once at the top
of an example. `MRT_EXAMPLES_DOWNLOAD_PATH` overrides the default, which is the package's
own Scratch cache, shared by every project on the machine.
"""
function setup()
    MRITestData.dismiss_terms_notice!()
    dir = get(ENV, "MRT_EXAMPLES_DOWNLOAD_PATH", "")
    isempty(dir) ? MRITestData.set_download_path!(:cache) : MRITestData.set_download_path!(dir)
    return nothing
end

"""
    load_example(source, id; max_bytes = typemax(Int), kwargs...)

Download the catalog entry `id` of `source` (a no-op if it is already cached) and load it.
Remaining keyword arguments (`slice`, `contrast`, `repetition`) are passed to `load_raw`,
which is how a multi-gigabyte file is kept out of memory.

Dwell times are repaired on the way out: a few exporters write `sample_time_us = 0`, which
is only recoverable from the trajectory description, and is needed by anything that works in
physical time (off-resonance correction, EPI ghost correction).
"""
function load_example(source, id; max_bytes = typemax(Int), kwargs...)
    entry = dataset(source, id)
    download_dataset(entry; max_bytes = max_bytes)
    raw = load_raw(entry; kwargs...)
    repair_dwell!(raw)
    return raw
end

"""
    repair_dwell!(raw) -> Union{Float64, Nothing}

Fill in a zero `sample_time_us` from the trajectory description, returning the dwell time in
microseconds that was written, or `nothing` if the profiles already carry one or the header
does not allow it to be recovered.
"""
function repair_dwell!(raw)
    any(p -> iszero(p.head.sample_time_us), raw.profiles) || return nothing
    desc = get(raw.params, "trajectoryDescription", nothing)
    desc isa AbstractDict || return nothing
    nsamples = max(Int(raw.profiles[1].head.number_of_samples), size(raw.profiles[1].data, 1))
    dwell = nothing
    for key in ("readTime_ns", "readTime_us", "readTime")
        value = get(desc, key, nothing)
        if value isa Real && value > 0
            dwell = Float64(value) / nsamples
            break
        end
    end
    (dwell === nothing || !(0.1 <= dwell <= 1000.0)) && return nothing
    for p in raw.profiles
        iszero(p.head.sample_time_us) && (p.head.sample_time_us = Float32(dwell))
    end
    return dwell
end

"""
    first_slab(raw)

A copy of `raw` holding one slab: the first value of every profile counter except the
phase-encoding steps. Sensitivity estimation and the iterative reconstructions are shown on
a single slab throughout, so that an example of a 20-slice cine stays a few seconds long.
"""
function first_slab(raw)
    p1 = raw.profiles[1]
    keep(p) = all(
        (
            p.head.idx.slice == p1.head.idx.slice,
            p.head.idx.contrast == p1.head.idx.contrast,
            p.head.idx.phase == p1.head.idx.phase,
            p.head.idx.repetition == p1.head.idx.repetition,
            p.head.idx.set == p1.head.idx.set,
            p.head.idx.average == p1.head.idx.average,
        )
    )
    return typeof(raw)(raw.params, [p for p in raw.profiles if keep(p)])
end

"""
    noise_covariance(raw)

The coil noise covariance estimated from the noise-adjustment profiles, or `nothing` when
the exporter dropped them. Feed it to [`prewhiten`](@ref).
"""
function noise_covariance(raw)
    noise = [p for p in raw.profiles if (p.head.flags & (UInt64(1) << 18)) != 0]
    isempty(noise) && return nothing
    samples = cat((p.data for p in noise)...; dims = 1)
    return estimate_noise_covariance(samples; coil_dim = 2)
end

"""
    describe(raw)
    describe(acq)

Print the few header fields that decide how the data has to be handled.
"""
function describe(raw::RawAcquisitionData)
    p = raw.profiles[1]
    @printf(
        "  raw: %d profiles, %d samples, %d channels, %s, encoded %s, dwell %s us\n",
        length(raw.profiles), size(p.data, 1), Int(p.head.active_channels),
        get(raw.params, "trajectory", "?"), string(Int.(raw.params["encodedSize"])),
        string(p.head.sample_time_us),
    )
    return nothing
end

function describe(acq::AcquisitionInfo)
    @printf(
        "  acq: %s, dims %s, size %s\n", string(nameof(typeof(acq))),
        string(dimnames(acq.kspace_data)), string(size(acq.kspace_data)),
    )
    return nothing
end

"""
    report(label, x)

Print the size and dynamic range of a reconstruction, and say so loudly when it came out
empty or non-finite — the two ways a real dataset fails silently.
"""
function report(label, x)
    a = abs.(collect(unname(x)))
    if !all(isfinite, a)
        @printf("  %-28s %s  NON-FINITE\n", label, string(size(x)))
    elseif iszero(maximum(a))
        @printf("  %-28s %s  ALL ZERO\n", label, string(size(x)))
    else
        @printf("  %-28s %s  max %.3g, mean %.3g\n", label, string(size(x)), maximum(a), mean(a))
    end
    return nothing
end

"""
    save_image(name, x; dim = index...) -> String

Write the magnitude of `x` as an 8-bit PGM under `examples/output/` and return the path.

Every dimension beyond the first two is selected by name, defaulting to its first entry, so
`save_image("brain", x; z = 9, coil = 3)` writes slice 9 of coil 3. The intensity scale is
per image, clipped at the 99.5th percentile so that a single hot pixel does not wash the
picture out.
"""
function save_image(name, x; kwargs...)
    names = dimnames(x)
    selection = values(kwargs)
    a = abs.(collect(unname(x)))
    for d in 3:ndims(x)
        key = d <= length(names) ? names[d] : Symbol(:dim, d)
        a = selectdim(a, 3, get(selection, key, 1))
    end
    a = Array{Float64}(a)
    hi = quantile(vec(a), 0.995)
    hi = hi > 0 ? hi : maximum(a)
    scaled = hi > 0 ? clamp.(a ./ hi, 0, 1) : zero(a)
    dir = joinpath(@__DIR__, "output")
    mkpath(dir)
    path = joinpath(dir, string(name, ".pgm"))
    open(path, "w") do io
        # A PGM row runs along the image's first axis, so write the transpose.
        write(io, "P5\n$(size(scaled, 1)) $(size(scaled, 2))\n255\n")
        write(io, round.(UInt8, 255 .* scaled'))
    end
    println("  wrote $(relpath(path, dirname(@__DIR__)))")
    return path
end

end # module
