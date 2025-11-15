"""
        AcquisitionInfo

Configuration container for MRI acquisition and encoding settings.

Use this to centralize validation and pass a single object to
`get_encoding_operator(::AcquisitionInfo)` and related helpers. The
constructor performs consistency checks across k-space layout, coil
maps, subsampling, and image size, and stores threading and FFTW
planning preferences.

Fields
- `kspace_data::K`: K‑space data array or template. Can be a plain
    `AbstractArray` or a `NamedDimsArray` whose first dimensions must
    be `:kx`, `:ky` (and `:kz` for 3D). When sensitivity maps are
    provided and `kspace_data` is named, a `:coil` dimension is required.
- `is3D::Bool`: Whether data is 3D. If `kspace_data` is a
    `NamedDimsArray`, this is inferred from the presence of `:kz`, or
    the length of `image_size` if provided. Otherwise it must be provided.
- `sensitivity_maps::S`: Coil sensitivity maps or `nothing`.
    Must be 3D (2D + coil) or 4D (3D + coil); element type must match
    `kspace_data`.
- `image_size::I`: `(nx, ny)` for 2D or `(nx, ny, nz)` for 3D, or
    `nothing`. Required when `subsampling` is provided and cannot be
    inferred from the pattern.
- `subsampling::Sub`: Subsampling pattern (2D/3D). Supports boolean
    masks and tuples combining `Colon`, boolean masks, and ranges.
- `shifted_kspace_dims::SD`: Dimensions in k‑space where the DC is at the
    first index (useful for pre‑shifted data).
- `shifted_image_dims::ID`: Image dimensions requiring fft shift
    (equivalent to an kspace-domain sign-alternation).

Constructor
```julia
AcquisitionInfo(
        kspace_data;
        is3D::Union{Bool,Nothing}=nothing,
        sensitivity_maps=nothing,
        image_size=nothing,
        subsampling=nothing,
        shifted_kspace_dims::Tuple=(),
        shifted_image_dims::Tuple=(),
)
```
When `kspace_data` is a `NamedDimsArray`, `is3D` is inferred from
the presence of `:kz`. Otherwise, `is3D` must be provided. If a
subsampling pattern is provided, `image_size` is validated or
inferred when possible.

Examples
- 2D, single‑coil, fully sampled Array:
```julia
ksp = rand(ComplexF32, 64, 64)
info = AcquisitionInfo(ksp; is3D=false)
```

- 2D, parallel imaging with NamedDims:
```julia
ksp   = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 64, 64, 8))
smaps = NamedDimsArray{(:x, :y, :coil)}(rand(ComplexF32, 64, 64, 8))
info = AcquisitionInfo(ksp; sensitivity_maps=smaps)
```

- 2D subsampled with mask:
```julia
ksp_full = rand(ComplexF32, 64, 64, 8)
mask = rand(Bool, 64, 64)
ksp_sub = ksp_full[mask, :]
info = AcquisitionInfo(ksp_sub; is3D=false, sensitivity_maps=nothing,
                                             image_size=(64, 64), subsampling=mask)
```

- 3D, parallel imaging Array:
```julia
ksp   = rand(ComplexF32, 64, 64, 32, 8)
smaps = rand(ComplexF32, 64, 64, 32, 8)
info = AcquisitionInfo(ksp; is3D=true, sensitivity_maps=smaps)
```
"""
struct AcquisitionInfo{K,I,S,Sub,SD,ID}
    kspace_data::K
    is3D::Bool
    image_size::I
    sensitivity_maps::S
    subsampling::Sub
    shifted_kspace_dims::SD
    shifted_image_dims::ID

    function AcquisitionInfo(ksp, is3D, img_size, smaps, subs, sK, sI)
        if !isnothing(subs)
            if !isnothing(ksp)
                guessed = MriReconstructionToolbox._get_img_size_from_subsampling(subs, ksp)
                if isnothing(img_size)
                    @argcheck !isnothing(guessed) "image_size must be provided or be inferable from subsampling"
                    img_size = guessed
                else
                    if !isnothing(guessed)
                        @argcheck img_size == guessed "image_size must match subsampling implied size"
                    end
                end
            end
            if !(subs isa Tuple)
                subs = (subs,)
            end
        end

        if isnothing(is3D)
            # Prefer inference from image size if provided
            if !isnothing(img_size)
                @argcheck length(img_size) == 2 || length(img_size) == 3 "image_size must be length 2 or 3"
                is3D = length(img_size) == 3
            else
                is3D = _infer_is3D_from_ksp(ksp)
                @argcheck !isnothing(is3D) "is3D must be provided when non-NamedDimsArray k-space is used and image_size is not given"
            end
        end

        if isnothing(img_size)
            if !isnothing(ksp)
                img_size = size(ksp)[1:(is3D ? 3 : 2)]
            elseif !isnothing(smaps)
                img_size = size(smaps)[1:(is3D ? 3 : 2)]
            end
        end
        @argcheck !isnothing(img_size) "image_size must be provided or inferable from subsampling or sensitivity maps"

        if ksp isa NamedDimsArray
            _check_ksp_dimnames(dimnames(ksp), subs, is3D, img_size)
        end

        if !isnothing(smaps)
            _check_smaps(smaps, ksp, subs, is3D)
        end

        if sK != ()
            if sK isa Integer || sK isa Symbol
                sK = (sK,)
            end
            for d in sK
                @argcheck d isa Integer || d isa Symbol "shifted_kspace_dims must be Integer, Symbol, or Tuple of those"
                if d isa Integer
                    @argcheck d ∈ 1:ndims(ksp) "shifted_kspace_dims out of range"
                else
                    @argcheck ksp isa NamedDimsArray "shifted_kspace_dims as Symbol requires NamedDimsArray k-space"
                    ksp_dimnames = ()
                    @argcheck d ∈ dimnames(ksp) "shifted_kspace_dims Symbol not found in k-space dimnames"
                end
            end
        end
        if sI != ()
            if sI isa Integer || sI isa Symbol
                sI = (sI,)
            end
            for d in sI
                @argcheck d isa Integer || d isa Symbol "shifted_image_dims must be Integer, Symbol, or Tuple of those"
                if d isa Integer
                    @argcheck d ∈ 1:length(img_size) "shifted_image_dims out of range"
                else
                    @argcheck ksp isa NamedDimsArray "shifted_image_dims as Symbol requires NamedDimsArray k-space"
                    img_dimnames = (:x, :y, (is3D ? :z : ()), dimnames(ksp)[(is3D ? 4 : 3):end]...) |> filter(!=(()))
                    @argcheck d ∈ img_dimnames "shifted_image_dims Symbol not found in image dimnames"
                end
            end
        end

        return new{typeof(ksp),typeof(img_size),typeof(smaps),typeof(subs),typeof(sK),typeof(sI)}(
            ksp, is3D, img_size, smaps, subs, sK, sI
        )
    end
end

AcquisitionInfo(
    kspace_data=nothing;
    is3D::Union{Bool,Nothing}=nothing,
    image_size=nothing,
    sensitivity_maps=nothing,
    subsampling=nothing,
    shifted_kspace_dims::Union{Tuple,Integer,Symbol}=(),
    shifted_image_dims::Union{Tuple,Integer,Symbol}=()
) = AcquisitionInfo(kspace_data, is3D, image_size, sensitivity_maps, subsampling, shifted_kspace_dims, shifted_image_dims)

function AcquisitionInfo(config::AcquisitionInfo; kwargs...)
    new_kwargs = Dict{Symbol,Any}()
	for fn in fieldnames(AcquisitionInfo)
		if haskey(kwargs, fn)
			new_kwargs[fn] = kwargs[fn]
		else
			new_kwargs[fn] = getfield(config, fn)
		end
	end
    args = (new_kwargs[fn] for fn in fieldnames(AcquisitionInfo))
	return AcquisitionInfo(args...)
end

function _check_smaps(smaps, ksp, subs, is3D)
    ksp_dims_count = isnothing(subs) ? (is3D ? 3 : 2) : length(subs)
    if ksp isa NamedDimsArray
        @argcheck smaps isa NamedDimsArray "sensitivity maps must be NamedDimsArray when k-space is NamedDimsArray"
        @argcheck :coil ∈ dimnames(ksp) ":coil dimension required in k-space when sensitivity maps are provided"
        if is3D
            @argcheck :z ∉ dimnames(ksp) "2D k-space must not have :z dimension"
            @argcheck :kz ∈ dimnames(ksp) "3D k-space must have :kz dimension"
            @argcheck dimnames(smaps) == (:x, :y, :z, :coil) "sensitivity maps dimnames must be (:x, :y, :z, :coil) for 3D acquisition"
        elseif ndims(smaps) == 4 # 2D multislice
            @argcheck dimnames(smaps) == (:x, :y, :coil, :z) "sensitivity maps dimnames must be (:x, :y, :coil, :z) for 2D acquisition"
            @argcheck dimnames(ksp, ksp_dims_count + 1) == :coil "k-space coil dimension must be right after k-space dimensions"
            @argcheck dimnames(ksp, ksp_dims_count + 2) == :z "k-space slice dimension must be right after coil dimension for 2D multislice acquisition"
            @argcheck :kz ∉ dimnames(ksp) "2D k-space must not have :kz dimension"
        else
            @argcheck dimnames(smaps) == (:x, :y, :coil) "sensitivity maps dimnames must be (:x, :y, :coil) for 2D acquisition"
            @argcheck dimnames(ksp, ksp_dims_count + 1) == :coil "k-space coil dimension must be right after k-space dimensions"
            @argcheck :kz ∉ dimnames(ksp) "2D k-space must not have :kz dimension"
            @argcheck :z ∉ dimnames(ksp) "2D k-space must not have :z dimension if sensitivity maps only 3-dimensional (2D+coil)"
        end
    end
    if !isnothing(ksp)
        @argcheck eltype(ksp) == eltype(smaps) "k-space and sensitivity maps eltype mismatch"
        if is3D
            @argcheck ndims(smaps) == 4 "sensitivity maps must be 4D for 3D acquisition"
        elseif ndims(smaps) == 4 # 2D multislice
            @argcheck ndims(ksp) >= ksp_dims_count + 2 "k-space must have slice dimension when sensitivity maps are 4D for 2D acquisition"
            @argcheck size(ksp, ksp_dims_count + 1) == size(smaps, 3) "k-space and sensitivity maps coil dimension size mismatch"
            @argcheck size(ksp, ksp_dims_count + 2) == size(smaps, 4) "k-space and sensitivity maps slice dimension size mismatch"
        else
            @argcheck ndims(smaps) == 3 "sensitivity maps must be 3D for 2D acquisition"
            @argcheck ndims(ksp) >= ksp_dims_count + 1 "k-space must have coil dimension when sensitivity maps are 3D for 2D acquisition"
            @argcheck size(ksp, ksp_dims_count + 1) == size(smaps, 3) "k-space and sensitivity maps coil dimension size mismatch"
        end
    end
end

function _subsample_item_to_str(item)
    if item isa Integer || item isa AbstractRange
        return string(item)
    elseif item isa AbstractVector
        return "Vector{$(eltype(item))}<$(join(size(item), "×"))>"
    elseif item isa AbstractArray
        return "Array{$(eltype(item))}<$(join(size(item), "×"))>"
    elseif item isa Colon
        return ":"
    else
        return "?"
    end
end

function _get_acq_info_meta(info::AcquisitionInfo)
    meta = String[]
    if !isnothing(info.kspace_data)
        if info.kspace_data isa NamedDimsArray
            dims = ["$dim: $s" for (dim, s) in zip(dimnames(info.kspace_data), size(info.kspace_data))]
            size_str = join(dims, ", ")
        else
            size_str = join(size(info.kspace_data), "×")
        end
        push!(meta, "kspace_data=Array{$(eltype(info.kspace_data))}<$size_str>")
    end
    push!(meta, "encoding=" * (info.is3D ? "3D" : "2D"))
    push!(meta, "image_size=$(join(info.image_size, "×"))")
    if !isnothing(info.sensitivity_maps)
        push!(meta, "sensitivity_maps=$(eltype(info.sensitivity_maps))<$(join(size(info.sensitivity_maps), "×"))>")
    end
    if !isnothing(info.subsampling)
        subsampling = [_subsample_item_to_str(subs) for subs in info.subsampling]
        if length(subsampling)  == 1
            subsampling = subsampling[1]
        else
            subsampling = "($(join(subsampling, ", ")))"
        end
        push!(meta, "subsampling=$subsampling")
    end
    if !isempty(info.shifted_kspace_dims)
        if length(info.shifted_kspace_dims) == 1
            shifted_kspace_dims = string(info.shifted_kspace_dims[1])
        else
            shifted_kspace_dims = "($(join(info.shifted_kspace_dims, ",")))"
        end
        push!(meta, "shifted_kspace_dims=$shifted_kspace_dims")
    end
    if !isempty(info.shifted_image_dims)
        if length(info.shifted_image_dims) == 1
            shifted_image_dims = string(info.shifted_image_dims[1])
        else
            shifted_image_dims = "($(join(info.shifted_image_dims, ",")))"
        end
        push!(meta, "shifted_image_dims=$shifted_image_dims")
    end
    return meta
end

function Base.show(io::IO, info::AcquisitionInfo)
    # Single line summary
    meta = _get_acq_info_meta(info)
    print(io, "AcquisitionInfo(", join(meta, ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", info::AcquisitionInfo)
    meta = _get_acq_info_meta(info)
    println(io, "AcquisitionInfo:")
    for (i, m) in enumerate(meta)
        m = replace(m, "=" => " = ", "_" => " ")
        print(io, "  - $m")
        if i < length(meta)
            println(io)
        end
    end
end

function _infer_is3D_from_ksp(ksp)
    if ksp isa NamedDimsArray
        return :kz ∈ dimnames(ksp)
    else
        return nothing
    end
end
