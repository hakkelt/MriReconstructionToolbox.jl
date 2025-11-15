struct ProblemDecompositionPlan{N,M,K,L}
	image_size::NTuple{N,Int}
	image_batch_dims::NTuple{M,Int}
	kspace_size::NTuple{K,Int}
	kspace_batch_dims::NTuple{L,Int}
	slices_sensitivity_maps::Bool
end

abstract type ReconstructionExecutor end
struct SequentialExecutor <: ReconstructionExecutor end
struct MultiThreadingExecutor <: ReconstructionExecutor end

function get_problem_decomposition_plan(acq_data, regularization, config)
	if config.disable_problem_decomposition || regularization == ()
		return nothing
	end

	# Determine which image dimensions can be used for problem decomposition
	image_dims = get_image_dims(acq_data)
	image_batch_dims = collect(get_nonfourier_image_dims(acq_data))
	for reg in regularization
		affected_dims = get_affected_dims(reg, acq_data, image_dims)
		image_batch_dims = setdiff(image_batch_dims, affected_dims)
	end
	image_batch_dims = tuple(image_batch_dims...)

	if image_batch_dims == () # no batch dimensions, no decomposition
		return nothing
	end

	image_size = get_image_size(acq_data)
	if image_batch_dims[1] isa Symbol # convert to indices
		image_batch_dims = findall(d -> d in image_batch_dims, eachindex(image_dims))
	end

	kspace_size = size(acq_data.kspace_data)

	# Calculate how the image batch dimensions map to k-space batch dimensions
	kspace_fourier_dims = get_fourier_kspace_dims(acq_data)
	image_fourier_dims = get_fourier_image_dims(acq_data)
	dim_index_offset = length(image_fourier_dims) - length(kspace_fourier_dims)
	if !isnothing(acq_data.sensitivity_maps)
		dim_index_offset -= 1 # account for coil dimension in k-space
	end
	kspace_batch_dims = tuple((d - dim_index_offset for d in image_batch_dims)...)

	slices_sensitivity_maps = (
		!isnothing(acq_data.sensitivity_maps) &&
		ndims(acq_data.sensitivity_maps) == 4 && # if true, the third dimension of the image must be the slice dimension
		3 ∈ image_batch_dims
	)

	return ProblemDecompositionPlan(
		image_size,
		image_batch_dims,
		kspace_size,
		kspace_batch_dims,
		slices_sensitivity_maps,
	)
end

function execute(f::Function, plan, acq_data, config)
	executor = suggest_executor(plan, config)
	return execute(f, plan, acq_data, config, executor)
end

function execute(f::Function, plan, acq_data, config, ::SequentialExecutor)
	maybe_print_decomposition_info(plan, config)
	batch_sizes = plan.image_size[collect(plan.image_batch_dims)]
	results = Array{AbstractArray}(undef, batch_sizes)
	scales = Array{real(eltype(acq_data.kspace_data))}(undef, batch_sizes)
	threaded = config.threaded
	slices = get_slices(plan, acq_data)
	@conditionally_enable_threading threaded for (idx, id, local_acq) in slices
		r, s = execute_single_slice(f, id, local_acq, config; threaded=threaded)
		results[idx] = r
		scales[idx] = s
	end
	maybe_rescale_results!(results, scales, config)
	return stack_image_slices(results, plan, Val(config.threaded))
end

function execute(f::Function, plan, acq_data, config, ::MultiThreadingExecutor)
	maybe_print_decomposition_info(plan, config)
	batch_sizes = plan.image_size[collect(plan.image_batch_dims)]
	results = Array{AbstractArray}(undef, batch_sizes)
	scales = Array{real(eltype(acq_data.kspace_data))}(undef, batch_sizes)
	slices = collect(get_slices(plan, acq_data))
	@restrict_threading @threads for (idx, id, local_acq) in slices
		r, s = execute_single_slice(f, id, local_acq, config; threaded=false)
		results[idx] = r
		scales[idx] = s
	end
	maybe_rescale_results!(results, scales, config)
	return stack_image_slices(results, plan, Val(config.threaded))
end

# Helper functions

function map_dims_to_strs(sizes, batch_dims)
	return map(d -> d[1] ∈ batch_dims ? "_$(d[2])_" : string(d[2]), enumerate(sizes))
end

function Base.show(io::IO, plan::ProblemDecompositionPlan)
	print(io, "ProblemDecompositionPlan{")
	img_size_strs = map_dims_to_strs(plan.image_size, plan.image_batch_dims)
	print(io, "image_size=(", join(img_size_strs, ", "), "), ")
	ksp_size_strs = map_dims_to_strs(plan.kspace_size, plan.kspace_batch_dims)
	print(io, "kspace_size=(", join(ksp_size_strs, ", "), ")}")
end

function Base.length(plan::ProblemDecompositionPlan)
	prod(plan.image_size[collect(plan.image_batch_dims)])
end

function maybe_print_decomposition_info(plan, config)
	if config.verbose
		batch_dims = plan.image_batch_dims
		batch_size = plan.image_size[collect(batch_dims)]
		if length(batch_dims) == 1
			msg_part = "dimension $(batch_dims[1]) with size $(batch_size[1])"
		else
			msg_part = "dimensions $batch_dims with sizes $batch_size"
		end
		config.printfunc("Decomposing problem over $msg_part")
	end
end

function get_slice_id(plan, idx, slice_idx_widths)
	id_parts = []
	counter = 1
	for d in eachindex(plan.image_size)
		if d in plan.image_batch_dims
			idx_str = @sprintf("%*s", slice_idx_widths[counter], string(idx[counter]))
			push!(id_parts, idx_str)
			counter += 1
		else
			push!(id_parts, ":")
		end
	end
	return "[" * join(id_parts, ", ") * "]"
end

function get_slices(plan, acq_data)
	indices = CartesianIndices(plan.kspace_size[collect(plan.kspace_batch_dims)])
	get_index_max_width = d -> length(string(size(acq_data.kspace_data, d)))
	slice_idx_widths = map(get_index_max_width, plan.kspace_batch_dims)
	slice_ids = map(idx -> get_slice_id(plan, idx, slice_idx_widths), indices)
	slices = eachslice(acq_data.kspace_data; dims=plan.kspace_batch_dims)
	ssm = plan.slices_sensitivity_maps
	local_acq = (
		get_acquisition_info_slice(acq_data, idx, ksp_slice, ssm) for
		(idx, ksp_slice) in zip(indices, slices)
	)
	return zip(indices, slice_ids, local_acq)
end

function get_acquisition_info_slice(acq_info, idx, kspace_data, slice_sensitivity_maps)
	if slice_sensitivity_maps
		sensitivity_maps = @view acq_info.sensitivity_maps[:,:,:,idx[1]]
		return AcquisitionInfo(acq_info; kspace_data, sensitivity_maps)
	else
		return AcquisitionInfo(acq_info; kspace_data)
	end
end

function stack_image_slices(results, plan, ::Val{false})
	full_image = similar(results[1], plan.image_size)
	for (output_slice, result) in
		zip(eachslice(full_image; dims=plan.image_batch_dims), results)
		output_slice .= result
	end
	return full_image
end

function stack_image_slices(results, plan, ::Val{true})
	full_image = similar(results[1], plan.image_size)
	extended_results = collect(
		zip(eachslice(full_image; dims=plan.image_batch_dims), results)
	)
	@threads for (output_slice, result) in extended_results
		output_slice .= result
	end
	return full_image
end

function execute_single_slice(f::Function, id, local_acq, config; kwargs...)
	if config.verbose
		freq = isnothing(config.freq) ? 0 : config.freq
	else
		freq = -1
	end
	printfunc = (s...) -> config.printfunc("[$id] ", s...)
	local_conf = Config(config; verbose=false, printfunc, freq, kwargs...)
	return f(local_acq, local_conf)
end

function maybe_rescale_results!(results, scales, config)
	if !config.disable_inverse_scale_output
		median_scale = median(scales)
		@threads for i in eachindex(results)
			results[i] .*= scales[i]
		end
		config.verbose &&
			config.printfunc("Rescaled output by median scale factor $median_scale")
	end
end

function suggest_executor(plan, config)
	if !isnothing(config.decomposition_executor)
		return config.decomposition_executor
	elseif config.threaded && length(plan) > nthreads()
		return MultiThreadingExecutor()
	else
		return SequentialExecutor()
	end
end
