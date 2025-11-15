function get_full_kspace(acq_info::AcquisitionInfo)
    @argcheck !isnothing(acq_info.kspace_data) "AcquisitionInfo must include k-space data"
    Γ = get_subsampling_operator(acq_info)
    return Γ' * acq_info.kspace_data
end

function normalize_op(A::AbstractOperator, exact_opnorm::Bool=false)
    if exact_opnorm
        L = LinearAlgebra.opnorm(A)
    else
        L = AbstractOperators.estimate_opnorm(A)
    end
    @argcheck L != 0 "Cannot normalize operator with zero norm"
    return 1/L * A
end

ensure_tuple(x::Tuple) = x
ensure_tuple(x) = (x,)


macro conditionally_enable_threading(threaded, expr)
    return quote
        if $(esc(threaded))
            @enable_full_threading $(esc(expr))
        else
            @restrict_threading $(esc(expr))
        end
    end
end
