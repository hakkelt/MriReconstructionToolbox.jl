import Base: size
import LinearAlgebra: mul!
import AbstractOperators:
	fun_name,
	domain_type,
	codomain_type,
	domain_storage_type,
	codomain_storage_type,
	allocate_in_domain,
	allocate_in_codomain,
	can_be_combined,
	combine,
	remove_displacement,
	get_normal_op,
	ndoms,
	domain_type,
	codomain_type,
	is_linear,
	is_eye,
	is_null,
	is_diagonal,
	is_AcA_diagonal,
	is_AAc_diagonal,
	is_orthogonal,
	is_invertible,
	is_full_row_rank,
	is_full_column_rank,
	is_sliced,
	diag_AcA,
	diag_AAc,
	displacement,
	is_thread_safe,
	has_optimized_normalop

import NamedDims: dimnames, parent, unname

struct NamedDimsOp{D,C} <: AbstractOperators.AbstractOperator
	L::AbstractOperators.AbstractOperator
end

mul!(y::AbstractArray, L::NamedDimsOp, b::AbstractArray) = mul!(unname(y), L.L, b)
function mul!(y::AbstractArray, L::NamedDimsOp{D,C}, b::NamedDimsArray) where {C,D}
	if dimnames(b) != D
        msg = "dimnames of b must match domain of operator L: $(dimnames(b)) != $D"
		throw(ArgumentError(msg))
	elseif y isa NamedDimsArray && dimnames(y) != C
        msg = "dimnames of y must match codomain of operator L: $(dimnames(y)) != $C"
		throw(ArgumentError(msg))
	end
	mul!(unname(y), L.L, unname(b))
end
function mul!(y::AbstractArray, L::AdjointOperator{<:NamedDimsOp}, b::AbstractArray)
	mul!(unname(y), L.A.L', b)
end
function mul!(
	y::AbstractArray, L::AdjointOperator{<:NamedDimsOp{D,C}}, b::NamedDimsArray
) where {C,D}
	if dimnames(b) != C
        msg = "dimnames of b must match codomain of operator L: $(dimnames(b)) != $C"
		throw(ArgumentError(msg))
	elseif y isa NamedDimsArray && dimnames(y) != D
        msg = "dimnames of y must match domain of operator L: $(dimnames(y)) != $D"
		throw(ArgumentError(msg))
	end
	mul!(unname(y), L.A.L', unname(b))
end

function domain_storage_type(L::NamedDimsOp{D,C}) where {D,C}
	NamedDimsArray{D,domain_type(L.L),domain_storage_type(L.L)}
end
function codomain_storage_type(L::NamedDimsOp{D,C}) where {D,C}
	NamedDimsArray{C,codomain_type(L.L),codomain_storage_type(L.L)}
end
function allocate_in_domain(L::NamedDimsOp{D,C}, dims...) where {D,C}
	NamedDimsArray{D}(allocate_in_domain(L.L, dims...))
end
allocate_in_domain(L::AdjointOperator{<:NamedDimsOp}, dims...) = allocate_in_codomain(L.A, dims...)
function allocate_in_codomain(L::NamedDimsOp{D,C}, dims...) where {D,C}
	NamedDimsArray{C}(allocate_in_codomain(L.L, dims...))
end
allocate_in_codomain(L::AdjointOperator{<:NamedDimsOp}, dims...) = allocate_in_domain(L.A, dims...)

function remove_displacement(L::NamedDimsOp{D,C}) where {D,C}
	NamedDimsOp{D,C}(remove_displacement(L.L))
end
get_normal_op(L::NamedDimsOp{D,C}) where {D,C} = NamedDimsOp{D,D}(get_normal_op(L.L))

can_be_combined(::NamedDimsOp, ::NamedDimsOp) = true
function combine(L::NamedDimsOp{D1,C1}, R::NamedDimsOp{D2,C2}) where {D1,D2,C1,C2}
	@assert D1 == C2 "Domain of $L must match codomain of $R"
    if can_be_combined(L.L, R.L)
	    return NamedDimsOp{D2,C1}(combine(L.L, R.L))
    else
        return NamedDimsOp{D2,C1}(L.L * R.L)
    end
end

fun_name(L::NamedDimsOp) = fun_name(L.L)

size(L::NamedDimsOp) = size(L.L)
ndoms(L::NamedDimsOp) = ndoms(L.L)
domain_type(L::NamedDimsOp) = domain_type(L.L)
codomain_type(L::NamedDimsOp) = codomain_type(L.L)
is_linear(L::NamedDimsOp) = is_linear(L.L)
is_eye(L::NamedDimsOp) = is_eye(L.L)
is_null(L::NamedDimsOp) = is_null(L.L)
is_diagonal(L::NamedDimsOp) = is_diagonal(L.L)
is_AcA_diagonal(L::NamedDimsOp) = is_AcA_diagonal(L.L)
is_AAc_diagonal(L::NamedDimsOp) = is_AAc_diagonal(L.L)
is_orthogonal(L::NamedDimsOp) = is_orthogonal(L.L)
is_invertible(L::NamedDimsOp) = is_invertible(L.L)
is_full_row_rank(L::NamedDimsOp) = is_full_row_rank(L.L)
is_full_column_rank(L::NamedDimsOp) = is_full_column_rank(L.L)
is_sliced(L::NamedDimsOp) = is_sliced(L.L)
diag_AcA(L::NamedDimsOp) = diag_AcA(L.L)
diag_AAc(L::NamedDimsOp) = diag_AAc(L.L)
displacement(L::NamedDimsOp) = displacement(L.L)
is_thread_safe(L::NamedDimsOp) = is_thread_safe(L.L)
has_optimized_normalop(L::NamedDimsOp) = has_optimized_normalop(L.L)

dimnames(::NamedDimsOp{D,C}) where {D,C} = (C, D)
dimnames(::NamedDimsOp{D,C}, i::Int) where {D,C} = i == 1 ? C : D
dimnames(::AdjointOperator{<:NamedDimsOp{D,C}}) where {D,C} = (D, C)
dimnames(::AdjointOperator{<:NamedDimsOp{D,C}}, i::Int) where {D,C} = i == 1 ? D : C

parent(L::NamedDimsOp) = L.L
parent(L::AdjointOperator{<:NamedDimsOp}) = L.A.L'
unname(L::NamedDimsOp) = L.L
unname(L::AdjointOperator{<:NamedDimsOp}) = L.A.L'

Base.:*(L::NamedDimsOp, R::NamedDimsOp) =
	NamedDimsOp{dimnames(R,2),dimnames(L,1)}(unname(L) * unname(R))
Base.:*(L::AdjointOperator{<:NamedDimsOp}, R::AdjointOperator{<:NamedDimsOp}) =
	NamedDimsOp{dimnames(R,2),dimnames(L,1)}(unname(L) * unname(R))
Base.:*(L::NamedDimsOp, R::AdjointOperator{<:NamedDimsOp}) =
	NamedDimsOp{dimnames(R,2),dimnames(L,1)}(unname(L) * unname(R))
Base.:*(L::AdjointOperator{<:NamedDimsOp}, R::NamedDimsOp) =
	NamedDimsOp{dimnames(R,2),dimnames(L,1)}(unname(L) * unname(R))
AbstractOperators.Scale(α::Number, L::NamedDimsOp) = NamedDimsOp{dimnames(L,2),dimnames(L,1)}(α * unname(L))
AbstractOperators.Scale(α::Number, L::AdjointOperator{<:NamedDimsOp}) =
	NamedDimsOp{dimnames(L,2),dimnames(L,1)}(α * unname(L))
