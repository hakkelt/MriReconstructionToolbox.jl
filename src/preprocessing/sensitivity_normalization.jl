"""
	normalize_sensitivity_maps(acq::AcquisitionInfo; threshold = 1.0e-4, coil_dim = nothing)
	normalize_sensitivity_maps(maps::AbstractArray; threshold = 1.0e-4, coil_dim = nothing)

Rescale sensitivity maps so that `Σ_c |S_c(r)|² = 1` at every voxel that carries signal, returning
a copy; the input is left untouched.

The maps are divided by `sqrt(Σ_c |S_c|²)`, which is a real, positive, per-voxel factor, so the
relative magnitude and phase between coils — the only thing the maps encode — is preserved.

Voxels where `Σ_c |S_c|²` falls below `threshold` times its maximum are set to zero instead of
being divided: outside the object the sum is noise, and dividing by it would amplify that noise
into the maps. The threshold is relative because the overall scale of a sensitivity map set is
arbitrary, so no absolute value could be meaningful.

The coil axis is taken from `coil_dim`, else from the `:coil` dimension of a `NamedDimsArray`,
else from the trailing-dims convention (see [`compress_coils`](@ref)). When `acq` has no
sensitivity maps, `acq` is returned unchanged.

# Why normalize

This is the conventional SENSE scaling (Pruessmann et al., *SENSE: sensitivity encoding for fast
MRI*, Magn. Reson. Med. 42:952-962, 1999; Roemer et al., *The NMR phased array*, Magn. Reson. Med.
16:192-225, 1990), and it buys three things:

 1. The encoding operator becomes a contraction: `‖𝒜‖ ≤ 1` holds by construction, so the operator
    norm behind the step size is known in closed form rather than estimated.
 2. The reconstructed image carries the conventional intensity scale instead of one that depends on
    how the maps happened to be estimated.
 3. Regularization strengths become comparable across datasets, because `λ` no longer competes with
    an arbitrary map scale.

It is not applied automatically: rescaling maps a caller supplied changes the units of the image
they get back, and map sets in the wild are not normalized (the maps shipped with this package have
`Σ_c |S_c|²` ranging over roughly 0.73-2.36).

# Examples

```julia
info = CartesianAcquisitionInfo(kspace; sensitivity_maps = maps)
normalized = normalize_sensitivity_maps(info)
```
"""
function normalize_sensitivity_maps(
        acq::AcquisitionInfo;
        threshold::Real = 1.0e-4,
        coil_dim = nothing,
    )
    isnothing(acq.sensitivity_maps) && return acq
    normalized = normalize_sensitivity_maps(acq.sensitivity_maps; threshold, coil_dim)
    return AcquisitionInfo(acq; sensitivity_maps = normalized)
end

function normalize_sensitivity_maps(
        maps::AbstractArray;
        threshold::Real = 1.0e-4,
        coil_dim = nothing,
    )
    @argcheck 0 <= threshold < 1 "threshold ($threshold) must be in [0, 1)"
    c_idx = _resolve_coil_dim(maps, coil_dim)
    T = real(eltype(maps))
    total = sum(abs2, maps; dims = c_idx)
    cutoff = T(threshold) * maximum(total)
    # A voxel with no signal keeps its zeros rather than being divided by noise.
    scale = map(t -> (t > 0 && t >= cutoff) ? inv(sqrt(t)) : zero(T), total)
    return maps .* scale
end
