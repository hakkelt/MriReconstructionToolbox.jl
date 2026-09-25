module OperatorCore

"""
	is_linear(A)

Returns true if `A` is a linear operator.
Operator `A` is linear if `A * (x+y) = A * x + A * y` and `A * (α * x) = α * A * x` for all `x` and `y` in the domain of `A` and all scalars `α`.
In particular `A * 0 = 0`: an affine operator `x ↦ Aₗ * x + d` with `d ≠ 0` is not linear (see [`is_affine`](@ref)).
"""
is_linear(L) = false

"""
	is_affine(A)

Returns true if `A` is an affine operator.
Operator `A` is affine if `A * x = Aₗ * x + d` for all `x` in the domain of `A`, where `Aₗ` is a linear operator and `d` is a fixed element of the codomain of `A` (the displacement).
Every linear operator is affine, with `d = 0`.
"""
is_affine(L) = is_linear(L)

"""
	is_null(A)

Returns true if `A` is a null operator.
Operator `A` is null if `A * x = 0` for all `x` in the domain of `A`, where `0` is the zero element of the codomain of `A` (e.g. zero vector).
"""
is_null(L) = false

"""
	is_eye(A)

Returns true if `A` is an identity operator.
Operator `A` is an identity operator if `A * x = x` for all `x` in the domain of `A`.
"""
is_eye(L) = false

"""
	is_symmetric(A)

Returns true if `A` is a symmetric operator.
Operator `A` is symmetric if `A * x = A' * x` for all `x` in the domain of `A`, where `A'` is the adjoint of `A`.
In other words, `A` is symmetric if it is equal to its adjoint.
"""
is_symmetric(L) = is_diagonal(L)

"""
	is_diagonal(A)

Returns true if `A` is a diagonal operator.
Operator `A` is diagonal if `(A * x)[i] = (A * eᵢ)[i]` for all `x` in the domain of `A` and all `i`, where `eᵢ` is the `i`-th canonical basis vector.
In other words, `(A * x)[i]` depends only on `x[i]`.
"""
is_diagonal(L) = is_eye(L)

"""
	is_AcA_diagonal(A)

Returns true if `A*A'` is diagonal, where `A'` is the adjoint of `A`.
Compounds `A*A'` is diagonal if `((A*A') * x)[i] = (A*A') * eᵢ)[i]` for all `x` in the domain of `A` and all `i`, where `eᵢ` is the `i`-th canonical basis vector.
In other words, `((A*A') * x)[i]` depends only on `x[i]`.
"""
is_AcA_diagonal(L) = is_diagonal(L)

"""
	is_AAc_diagonal(A)

Returns true if `A'*A` is diagonal, where `A'` is the adjoint of `A`.
Compounds `A'*A` is diagonal if `((A'*A) * x)[i] = (A'*A) * eᵢ)[i]` for all `x` in the domain of `A` and all `i`, where `eᵢ` is the `i`-th canonical basis vector.
In other words, `((A'*A) * x)[i]` depends only on `x[i]`.
"""
is_AAc_diagonal(L) = is_diagonal(L)

"""
	diag_AcA(A)

Returns the diagonal of `A*A'`, where `A'` is the adjoint of `A`.
If `A*A'` is not diagonal, an error is thrown.
"""
diag_AcA(L) = error("is_AcA_diagonal($L) == false")

"""
	diag_AAc(A)

Returns the diagonal of `A'*A`, where `A'` is the adjoint of `A`.
If `A'*A` is not diagonal, an error is thrown.
"""
diag_AAc(L) = error("is_AAc_diagonal($L) == false")

"""
	is_orthogonal(A)

Returns true if `A` is an orthogonal operator.
Operator `A` is orthogonal if `A * A' = A' * A = I`, where `A'` is the adjoint of `A` and `I` is the identity operator.
"""
is_orthogonal(L) = false

"""
	is_invertible(A)

Returns true if `A` is an invertible operator.
Operator `A` is invertible if there exists an operator `B` such that `A * B = B * A = I`, where `I` is the identity operator.
"""
is_invertible(L) = is_orthogonal(L) || (is_full_row_rank(L) && is_full_column_rank(L))

"""
	is_full_row_rank(A)

Returns true if `A` has full row rank.
Operator `A` has full row rank if the rows of `A` are linearly independent.
In other words, the number of linearly independent rows of `A` is equal to the number of rows of `A`.
"""
is_full_row_rank(L) = false

"""
	is_full_row_rank(A)

Returns true if `A` has full column rank.
Operator `A` has full column rank if the columns of `A` are linearly independent.
In other words, the number of linearly independent columns of `A` is equal to the number of columns of `A`.
"""
is_full_column_rank(L) = false

"""
	is_positive_definite(A)

Returns true if `A` is a positive definite operator.
Operator `A` is positive definite if `x' * A * x > 0` for all non-zero vectors `x`, where `x'` is the conjugate transpose of `x`.
Equivalently, `A` is positive definite if it is symmetric and all its eigenvalues are positive.
"""
is_positive_definite(L) = false

"""
	is_positive_semidefinite(A)

Returns true if `A` is a positive semidefinite operator.
Operator `A` is positive semidefinite if `x' * A * x >= 0` for all vectors `x`, where `x'` is the conjugate transpose of `x`.
Equivalently, `A` is positive semidefinite if it is symmetric and all its eigenvalues are non-negative.
"""
is_positive_semidefinite(L) = is_positive_definite(L)

end # module OperatorCore
