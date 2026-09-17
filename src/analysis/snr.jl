"""
    estimate_snr(image; signal_box = nothing, noise_box = nothing, corners = :all) -> Real

Signal-to-noise ratio of a magnitude image, measured the way a clinical acceptance test measures it
(NEMA MS 1; Dietrich et al., *Measurement of SNR in MR images*, J. Magn. Reson. Imaging 26:375-385,
2007, their method 1): the mean signal in a box at the centre of the image, divided by the standard
deviation of the background in boxes placed in its corners, times the Rayleigh correction factor
`sqrt(2 - π/2) ≈ 0.6551`:

```math
\\mathrm{SNR} = 0.6551 \\; \\frac{\\overline{S}_{\\text{centre box}}}{\\sigma_{\\text{corner boxes}}}
```

The correction is what makes the result an SNR rather than a ratio of two differently-distributed
statistics: a corner box contains no signal, so the magnitude of complex Gaussian noise there is
Rayleigh-distributed and its standard deviation is `0.6551 σ`, not the `σ` the ratio wants.

Both regions are boxes of a size the caller gives in **voxels**. Nothing is segmented and no
threshold is applied: where the two regions sit is a property of the measurement, not of the image
it is measured in. That is what makes the number comparable between images — a threshold moves with
the noise it is supposed to be measuring, and reads high by tens of percent once the noise stops
being small — and it is also what makes it the caller's responsibility to check that the boxes land
where they should. [`snr_masks`](@ref) returns the two regions so they can be displayed.

# Keyword arguments

- `signal_box` — the size of the centred signal box, in voxels: an `Integer` for the same side in
  every dimension, or a `Tuple` of `Integer`s for one side per dimension. `nothing` (the default)
  uses an eighth of each image dimension. Make it small enough to stay inside uniform tissue: the
  mean over a box straddling an edge is not a signal level.
- `noise_box` — the size of each corner box, in voxels, in the same two forms. `nothing` (the
  default) uses an eighth of each image dimension. Boxes are taken in every dimension, so a 3D
  volume is read in its eight corners rather than in the four corners of one slice.
- `corners` — which corners to use. `:all` (the default) pools every corner; an `Integer` or a
  collection of `Integer`s selects corners by index, counted `1:2^ndims(image)` with the first
  dimension varying fastest (in 2D: 1 = low-low, 2 = high-low, 3 = low-high, 4 = high-high). Use it
  when part of the field of view is not signal-free — a wrapped shoulder, a coil close to the edge,
  a flow artefact — since a corner that contains anything but noise makes the estimate read low.

The background must *be* noise. On data from a parallel-imaging reconstruction, or with any
filtering that has touched the air around the object, the corner standard deviation is no longer the
noise level and the number is not an SNR; the multiple-acquisition estimator [`pseudo_replica`](@ref)
is the one to reach for there.

Measured against `add_noise(; snr)` on a uniform box, the estimate is unbiased to well under a
percent from SNR 5 upwards, and reads about 10% high at SNR 2, where the Rician bias on the signal
mean stops being negligible.

A noiseless image has a corner standard deviation of zero, and the ratio is then `Inf`.

# Examples
```julia
noisy = add_noise(phantom; snr = 20)
estimate_snr(noisy)                                        # ≈ 20
estimate_snr(noisy; signal_box = 24, noise_box = 16)
estimate_snr(noisy; noise_box = (16, 16), corners = (1, 2))  # only the two corners at low y
```
"""
function estimate_snr(
        image::AbstractArray;
        signal_box = nothing,
        noise_box = nothing,
        corners = :all,
    )
    mag = _magnitude(image)
    signal, noise = _snr_regions(size(mag), signal_box, noise_box, corners)
    return RAYLEIGH_CORRECTION * mean(mag[signal]) / std(mag[noise])
end

# The standard deviation of a Rayleigh magnitude is `sqrt(2 - π/2)` times the standard deviation of
# the underlying Gaussian noise in each component, so the background standard deviation has to be
# divided by it — equivalently the ratio is multiplied by it — to become a noise level.
const RAYLEIGH_CORRECTION = sqrt(2 - π / 2)

"""
    snr_masks(image; signal_box = nothing, noise_box = nothing, corners = :all) -> (signal, noise)

The two boolean masks [`estimate_snr`](@ref) measures over, returned so that the regions a reported
SNR was taken from can be displayed rather than described. `signal` is the centred signal box;
`noise` is the union of the selected corner boxes. The keyword arguments are [`estimate_snr`](@ref)'s,
and mean the same thing.
"""
function snr_masks(
        image::AbstractArray;
        signal_box = nothing,
        noise_box = nothing,
        corners = :all,
    )
    return _snr_regions(size(_magnitude(image)), signal_box, noise_box, corners)
end

_magnitude(image::AbstractArray) = abs.(image isa NamedDimsArray ? NamedDims.unname(image) : image)

function _snr_regions(sz::Dims, signal_box, noise_box, corners)
    signal = falses(sz)
    signal[_centre_box(sz, signal_box)...] .= true

    noise = falses(sz)
    for ranges in _corner_boxes(sz, noise_box, corners)
        noise[ranges...] .= true
    end

    if any(signal .& noise)
        throw(
            ArgumentError(
                "the signal box overlaps a corner box: the two together do not fit in an image of size $(sz). Reduce `signal_box` or `noise_box`.",
            )
        )
    end
    return signal, noise
end

# A box side per dimension, in voxels, from an `Integer`, a per-dimension tuple, or `nothing` for
# an eighth of each image dimension.
function _box_sides(sz::Dims, box, name::String)
    n = length(sz)
    sides = if box === nothing
        ntuple(d -> max(1, sz[d] ÷ 8), n)
    elseif box isa Integer
        ntuple(_ -> Int(box), n)
    else
        @argcheck length(box) == n "`$(name)` has $(length(box)) entries for a $(n)-dimensional image"
        @argcheck all(b -> b isa Integer, box) "`$(name)` is a size in voxels, so its entries must be `Integer`s"
        ntuple(d -> Int(box[d]), n)
    end
    for d in 1:n
        @argcheck 0 < sides[d] <= sz[d] "`$(name)` asks for $(sides[d]) voxels along a dimension of $(sz[d])"
    end
    return sides
end

# The index ranges of the box centred in the image.
function _centre_box(sz::Dims, box)
    sides = _box_sides(sz, box, "signal_box")
    return ntuple(d -> ((sz[d] - sides[d]) ÷ 2 + 1):((sz[d] - sides[d]) ÷ 2 + sides[d]), length(sz))
end

# The index ranges of the selected corner boxes. Corners are numbered `1:2^n` with the first
# dimension varying fastest, so in 2D: 1 = low-low, 2 = high-low, 3 = low-high, 4 = high-high.
function _corner_boxes(sz::Dims, box, corners)
    n = length(sz)
    sides = _box_sides(sz, box, "noise_box")
    for d in 1:n
        @argcheck sides[d] <= sz[d] ÷ 2 "a corner box of $(sides[d]) voxels does not fit twice along a dimension of $(sz[d]); the boxes at the two ends would meet"
    end
    ncorners = 1 << n
    selected = if corners === :all
        1:ncorners
    elseif corners isa Integer
        (Int(corners),)
    else
        @argcheck !isempty(corners) "`corners` selects no corner to measure the noise in"
        Int.(corners)
    end
    for c in selected
        @argcheck 1 <= c <= ncorners "corner $(c) does not exist: a $(n)-dimensional image has $(ncorners) corners"
    end
    return [
        ntuple(
                d -> ((c - 1) >> (d - 1)) & 1 == 0 ? (1:sides[d]) : (sz[d] - sides[d] + 1):sz[d],
                n,
            ) for c in unique(selected)
    ]
end

# Mean magnitude over the same centred box `estimate_snr` measures the signal in, so
# `add_noise(x; snr = s)` and `estimate_snr` agree by construction.
function signal_box_mean(image::AbstractArray, signal_box)
    mag = _magnitude(image)
    return mean(@view mag[_centre_box(size(mag), signal_box)...])
end
