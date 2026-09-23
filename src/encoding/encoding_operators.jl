"""
Main encoding operators for MRI reconstruction.

This module provides the primary interface for creating MRI encoding operators
that model the complete data acquisition process, including Fourier transforms,
sensitivity map encoding, and subsampling patterns.
"""

"""
	get_encoding_operator(info::AcquisitionInfo; threaded::Bool=true, fast_planning::Bool=false)
    get_encoding_operator(ksp, is3D::Bool; sensitivity_maps=nothing, image_size=nothing, subsampling=nothing, threaded=true, fast_planning=false)
    get_encoding_operator(ksp::NamedDimsArray; sensitivity_maps=nothing, image_size=nothing, subsampling=nothing, threaded=true, fast_planning=false)

Create the main MRI encoding operator for data acquisition modeling.

# Arguments for AcquisitionInfo method
- `info::AcquisitionInfo`: Contains k-space data, sensitivity maps, image size, subsampling pattern, and other acquisition parameters.

# Arguments for raw k-space method
- `ksp`: K-space data array
- `is3D::Bool`: Whether the acquisition is 3D
- `sensitivity_maps`: Coil sensitivity maps (optional)
- `image_size`: Image size tuple (optional)
- `subsampling`: Subsampling pattern (optional)

# Arguments for NamedDimsArray method
- `ksp::NamedDimsArray`: K-space data with named dimensions
- `sensitivity_maps`: Coil sensitivity maps (optional, can be NamedDimsArray)
- `image_size`: Image size tuple (optional, Tuple{Int,Int} or Tuple{Int,Int,Int})
- `subsampling`: Subsampling pattern (optional, 2D or 3D pattern)

# Common keyword arguments
- `threaded::Bool=true`: Whether to use multi-threading for operator construction and FFTs.
- `fast_planning::Bool=false`: Whether to use fast FFTW planning (reduces setup time, may affect performance).
- `m`, `sigma`, `precompute`: Non-Cartesian only. NFFT gridding operating point, forwarded to
  `get_fourier_operator`/`NFFTOp`; `nothing` (the default) substitutes MRT's own defaults
  (`DEFAULT_NFFT_M`, `DEFAULT_NFFT_SIGMA`, `DEFAULT_NFFT_PRECOMPUTE`),
  which are faster than NFFT.jl's at an accuracy that is still far below the noise floor. See
  "Non-Cartesian accuracy / speed trade-off" in `docs/src/high-level/performance.md`.

# Returns
- Encoding operator modeling the full MRI acquisition process, including Fourier transform, sensitivity map encoding, and subsampling (if present).

# Details
This function constructs the composite encoding operator E that models the MRI data acquisition pipeline:
1. Applies sensitivity map encoding (if provided)
2. Applies Fourier transform (subsampled if a subsampling pattern is present)
3. Returns the composed operator E = F * S or E = F

If no sensitivity maps are provided, only the Fourier/subsampled Fourier operator is returned.
"""
function get_encoding_operator(info::CartesianAcquisitionInfo; threaded::Bool = true, fast_planning::Bool = false)
    @argcheck !isnothing(info.kspace_data) "The provided CartesianAcquisitionInfo does not contain k-space data, which is required to build the encoding operator."
    fused = _coil_fused_encoding_operator(info; threaded, fast_planning)
    isnothing(fused) || return fused
    has_subs = !isnothing(info.subsampling)
    ℱ = has_subs ? get_subsampled_fourier_operator(info; threaded, fast_planning) : get_fourier_operator(info; threaded, fast_planning)
    return _compose_with_sensitivity(ℱ, info; threaded)
end

"""
	_coil_fused_encoding_operator(info::CartesianAcquisitionInfo; threaded, fast_planning)

The Cartesian multicoil encoding operator written as one batch over coils, or `nothing` when
the acquisition is not of the shape this form covers (in which case the caller builds the
generic chain).

`𝒫 ℱ 𝒮 ℬ` — broadcast the image over the coil axis, multiply by the maps, transform, sample —
has a coil axis running through every stage after the broadcast, and no stage mixes coils. The
same operator can therefore be written the other way round: one per-coil operator
`𝒫 ℱ diag(smaps[…, c])`, applied to all coils in a single batch loop. The two forms compute the
same thing to the last bit; what differs is how many parallel regions an apply opens, and how
much memory traffic it does.

The chain form opens one per stage: with a 128×128 image and 8 coils, a forward apply is a
broadcast, a `DiagOp`, a 3-D `DFT`, a `SignAlternation` and a batched `GetIndex`, each threading
(or not) on its own, each reading and writing a full 128×128×8 intermediate. The batch form
opens one region for the whole chain and keeps each coil's 128×128 working set in one worker's
cache from the multiply to the sampling.

Measured on the comparison benchmark's 2×-undersampled phantom (128×128, 8 coils, `ComplexF32`),
AMD EPYC 7352, 8 Julia threads, minimum of 100 applies. The normal operator is the row that
matters: `SqrNormL2 ∘ 𝒜` takes StructuredOptimization's `:normal_op` route, so a CG iteration
applies one `𝒜ᴴ𝒜` pass on the image domain rather than a forward and an adjoint. Both forms
answer `has_optimized_normalop` with `true`, so both get the fused pass and the row below is a
like-for-like comparison.

| apply           |  chain   | coil-fused |
|-----------------|----------|------------|
| forward         |  420.9 µs |  107.6 µs |
| adjoint         |  407.2 µs |  148.3 µs |
| `𝒜ᴴ𝒜` (per CG)  |  873.4 µs |  213.2 µs |

All three are bit-identical to the chain's output.

At one thread the fused form is a wash and is not built — 465.1 µs against 452.6 forward,
710.2 against 688.1 adjoint, 1168.7 against 1141.6 for `𝒜ᴴ𝒜`, i.e. 2-3 % in its favour against
`ncoils` FFT plans instead of one at build time (1.05 ms against 0.21 ms for this shape). The
`Threads.nthreads() > 1` condition below is that measurement, not an assumption: with 8 threads
available but `threaded = false` the fused form is actually the slower of the two (506.2 µs
forward against the chain's 460.1).

`FIXED_OPERATOR` rather than the default `AUTO`: every per-coil operator here is freshly built
and therefore used by exactly one batch item, which is the condition that strategy checks for
and the reason it needs neither the per-item lock nor the per-thread copies the other two
strategies pay for. `AUTO` picks `LOCKING` for this shape (the copies would exceed its 10 MB
budget), which measured 243 µs against 187 µs for this one.

The batch loop itself has no headroom left to find: driving the same per-coil operators from a
bare `Threads.@threads` loop measured 269.9 µs against the batch operator's 287.5 µs, where one
coil alone is 70.7 µs and all eight serially are 542.5 µs. What caps it at ~2x on 8 threads is
the per-coil work, not the loop around it.

The cost is one FFT plan per coil instead of one batched plan, paid once when the operator is
built. `fast_planning` is forwarded unchanged, so a caller that cares keeps its existing control.

Returns `nothing` unless the acquisition is Cartesian with sensitivity maps whose last axis is
`:coil`, nothing after the coil axis in k-space, more than one coil, and threading actually
available: the fused form is not faster serially, and the generic chain stays the only path a
single-threaded run takes.
"""
function _coil_fused_encoding_operator(
        info::CartesianAcquisitionInfo; threaded::Bool, fast_planning::Bool
    )
    smaps = info.sensitivity_maps
    (threaded && !isnothing(smaps) && Threads.nthreads() > 1) || return nothing
    ksp = info.kspace_data
    image_size = info.image_size
    nd = length(image_size)
    ndims(smaps) == nd + 1 || return nothing
    _has_dimnames(smaps) && dimnames(smaps)[end] !== :coil && return nothing
    _has_dimnames(ksp) && dimnames(ksp)[end] !== :coil && return nothing
    # Anything after the coil axis (a time or slab axis) means a per-coil operator is not the
    # whole story: the generic chain, which wraps the result in its own `BatchOp`, stays in
    # charge rather than growing a second batching layer here.
    ndims(ksp) == _get_sample_dims_count(info) + 1 || return nothing
    ncoils = size(smaps, nd + 1)
    ncoils > 1 || return nothing
    _is_cartesian_storage(ksp) || return nothing

    plain_smaps = smaps isa NamedDimsArray ? unname(smaps) : smaps
    ksp_one_coil = ksp[ntuple(_ -> Colon(), ndims(ksp) - 1)..., 1]
    shift_kwargs = (
        shifted_kspace_dims = info.shifted_kspace_dims,
        shifted_image_dims = info.shifted_image_dims,
    )
    # A fresh Fourier operator per coil, never one shared object: each carries its own FFT plan
    # and scratch buffers, which is what makes the batch loop race-free without locking.
    single_coil_fourier() = _unwrap_named(
        if isnothing(info.subsampling)
            ksp_one_coil isa NamedDimsArray ?
                get_fourier_operator(ksp_one_coil; shift_kwargs..., threaded = false, fast_planning) :
                get_fourier_operator(ksp_one_coil, info.is3D; shift_kwargs..., threaded = false, fast_planning)
        else
            get_subsampled_fourier_operator(
                ksp_one_coil, image_size, info.subsampling;
                shift_kwargs..., threaded = false, fast_planning,
            )
        end
    )
    per_coil = [
        single_coil_fourier() * DiagOp(copy(selectdim(plain_smaps, nd + 1, c)); threaded = false)
            for c in 1:ncoils
    ]
    codomain_rank = length(size(first(per_coil), 1))
    mask = (ntuple(_ -> :_, nd)..., :s) => (ntuple(_ -> :_, codomain_rank)..., :s)
    𝒞 = BatchOp(
        per_coil, mask;
        threaded, threading_strategy = ThreadingStrategy.FIXED_OPERATOR,
    )
    ℬ = BroadCast(
        Eye(zeros(domain_type(first(per_coil)), image_size...)),
        (image_size..., ncoils); threaded,
    )
    op = 𝒞 * ℬ
    _has_dimnames(ksp) || return op
    return NamedDimsOp{get_image_dims(info), dimnames(ksp)}(op)
end

_unwrap_named(op::NamedDimsOp) = op.L
_unwrap_named(op::AbstractOperators.AbstractOperator) = op

# `PartitionedKSpace` and friends do not have one array behind them, so `ksp[…, 1]` is not a
# single-coil k-space of the same kind and the fused form does not apply.
_is_cartesian_storage(::AbstractArray) = true
_is_cartesian_storage(::Any) = false

function get_encoding_operator(
        info::NonCartesianAcquisitionInfo;
        threaded::Bool = true,
        fast_planning::Bool = false,
        m::Union{Nothing, Integer} = nothing,
        sigma::Union{Nothing, Real} = nothing,
        precompute = nothing,
    )
    @argcheck !isnothing(info.kspace_data) "The provided NonCartesianAcquisitionInfo does not contain k-space data, which is required to build the encoding operator."
    ℱ = get_fourier_operator(info; threaded, m, sigma, precompute)
    return _compose_with_sensitivity(ℱ, info; threaded)
end

function _compose_with_sensitivity(ℱ, info::AcquisitionInfo; threaded::Bool)
    smaps = info.sensitivity_maps
    # A named acquisition builds a named Fourier operator, and a plain sensitivity operator cannot
    # be composed with it. Plain maps are given the standard names instead — the layout the
    # positional branch below assumes anyway. (A Cartesian acquisition already rejects this
    # combination; a non-Cartesian one, e.g. one `simulate_acquisition` gave named k-space, did
    # not, and failed here with a storage-type `DomainError`.)
    if !isnothing(smaps) && !(smaps isa NamedDimsArray) && _has_dimnames(info.kspace_data)
        smaps = NamedDimsArray{_standard_smaps_dimnames(ndims(smaps), info.is3D)}(smaps)
    end
    return if isnothing(smaps)
        ℱ
    elseif smaps isa NamedDimsArray && _has_dimnames(info.kspace_data)
        # Named maps only give named batch dimensions when the acquisition itself is named:
        # with unnamed k-space `get_image_dims` answers with integer positions, and building a
        # `NamedTuple` from those is a `TypeError`, not a useful error message. Fall through to
        # the positional branch instead, which is what the rest of the model uses in that case.
        image_size = get_image_size(info)
        image_dims = get_image_dims(info)
        # `smaps` has one more dimension than it consumes from the image domain (the :coil axis
        # isn't an image dimension), so the batch dims start right after `ndims(smaps) - 1` image
        # dims, not after `ndims(smaps)`.
        consumed = ndims(smaps) - 1
        batch_dims_size = image_size[(consumed + 1):end]
        batch_dim_names = image_dims[(consumed + 1):end]
        batch_dims = NamedTuple{batch_dim_names}(batch_dims_size)
        𝒮 = get_sensitivity_map_operator(smaps; batch_dims, threaded)
        ℱ * 𝒮
    else
        plain_smaps = smaps isa NamedDimsArray ? unname(smaps) : smaps
        batch_dims_start = ndims(plain_smaps) + 1
        batch_dims = size(ℱ, 2)[batch_dims_start:end]
        𝒮 = get_sensitivity_map_operator(plain_smaps, info.is3D; batch_dims, threaded)
        ℱ * 𝒮
    end
end

# The dimension names sensitivity maps carry by convention: `(:x, :y, :z, :coil)` in 3D,
# `(:x, :y, :coil)` for one 2D slice and `(:x, :y, :coil, :z)` for a 2D multislice stack.
function _standard_smaps_dimnames(nd::Int, is3D::Bool)
    is3D && return (:x, :y, :z, :coil)
    nd == 3 && return (:x, :y, :coil)
    nd == 4 && return (:x, :y, :coil, :z)
    throw(ArgumentError("sensitivity maps of a 2D acquisition have 3 or 4 dimensions, got $nd"))
end

function get_encoding_operator(
        ksp,
        is3D::Bool;
        sensitivity_maps = nothing,
        image_size = nothing,
        subsampling = nothing,
        threaded::Bool = true,
        fast_planning::Bool = false,
    )
    info = CartesianAcquisitionInfo(ksp; is3D, sensitivity_maps, image_size, subsampling)
    return get_encoding_operator(info; threaded, fast_planning)
end

function get_encoding_operator(
        ksp::NamedDimsArray;
        sensitivity_maps::Union{<:NamedDimsArray, Nothing} = nothing,
        image_size::Union{Tuple{Int, Int}, Tuple{Int, Int, Int}, Nothing} = nothing,
        subsampling = nothing,
        threaded::Bool = true,
        fast_planning::Bool = false,
    )
    info = CartesianAcquisitionInfo(ksp; sensitivity_maps, image_size, subsampling)
    return get_encoding_operator(info; threaded, fast_planning)
end
