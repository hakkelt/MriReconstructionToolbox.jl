function get_full_kspace(acq_info::CartesianAcquisitionInfo)
    @argcheck !isnothing(acq_info.kspace_data) "AcquisitionInfo must include k-space data"
    𝒫 = get_subsampling_operator(acq_info)
    return 𝒫' * acq_info.kspace_data
end

function normalize_op(A::AbstractOperator, exact_opnorm::Bool = false)
    if exact_opnorm
        L = LinearAlgebra.opnorm(A)
    else
        L = AbstractOperators.estimate_opnorm(A)
    end
    @argcheck L != 0 "Cannot normalize operator with zero norm"
    return 1 / L * A
end

ensure_tuple(x::Tuple) = x
ensure_tuple(x) = (x,)


macro conditionally_enable_threading(threaded, expr)
    return quote
        if $(esc(threaded))
            with_full_threads() do
                $(esc(expr))
            end
        else
            with_restricted_threads() do
                $(esc(expr))
            end
        end
    end
end
