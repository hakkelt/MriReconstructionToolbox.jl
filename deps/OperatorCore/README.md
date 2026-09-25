# OperatorCore.jl

A lightweight Julia package providing a trait-based interface for defining properties of linear operators.

## Overview

OperatorCore.jl defines a minimal set of traits (predicate functions) that can be used to query mathematical properties of operators. This package is designed to be extended by concrete operator implementations to enable optimized algorithms based on operator properties.

## Features

OperatorCore provides the following property queries:

### Linearity and Basic Properties

- **`is_linear(A)`** - Check if `A` is a linear operator (so `A * 0 = 0`)
- **`is_affine(A)`** - Check if `A` is an affine operator, `A * x = Aₗ * x + d` with `Aₗ` linear (defaults to `is_linear(A)`)
- **`is_null(A)`** - Check if `A` is a null operator (always returns zero)
- **`is_eye(A)`** - Check if `A` is an identity operator
- **`is_symmetric(A)`** - Check if `A` equals its adjoint

### Structural Properties

- **`is_diagonal(A)`** - Check if `A` is a diagonal operator
- **`is_AcA_diagonal(A)`** - Check if `A'*A` is diagonal
- **`is_AAc_diagonal(A)`** - Check if `A*A'` is diagonal
- **`diag_AcA(A)`** - Get the diagonal of `A'*A` (if diagonal)
- **`diag_AAc(A)`** - Get the diagonal of `A*A'` (if diagonal)

### Matrix Properties

- **`is_orthogonal(A)`** - Check if `A*A' = A'*A = I`
- **`is_invertible(A)`** - Check if `A` is invertible
- **`is_full_row_rank(A)`** - Check if rows are linearly independent
- **`is_full_column_rank(A)`** - Check if columns are linearly independent

### Definiteness Properties

- **`is_positive_definite(A)`** - Check if `A` is positive definite
- **`is_positive_semidefinite(A)`** - Check if `A` is positive semidefinite

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hakkelt/OperatorCore.jl")
```

Or in development mode:

```julia
using Pkg
Pkg.develop(path="/path/to/OperatorCore")
```

## Usage

OperatorCore is meant to be extended by concrete operator types. All trait functions return `false` by default.

### Defining a Custom Operator Type

```julia
using OperatorCore

# Define your operator type
struct MyDiagonalOperator
    diagonal::Vector{Float64}
end

# Implement relevant traits
OperatorCore.is_linear(::MyDiagonalOperator) = true
OperatorCore.is_diagonal(::MyDiagonalOperator) = true
OperatorCore.is_symmetric(::MyDiagonalOperator) = true

# Use the traits for optimized algorithms
function solve(A, b)
    if is_diagonal(A)
        # Use efficient diagonal solve
        return diagonal_solve(A, b)
    else
        # Fall back to general solver
        return general_solve(A, b)
    end
end
```

### Using with AbstractOperators.jl

OperatorCore is designed to work seamlessly with the [AbstractOperators.jl](https://github.com/kul-forbes/AbstractOperators.jl) ecosystem:

```julia
using AbstractOperators
using OperatorCore

# AbstractOperators types already implement OperatorCore traits
A = Eye(10)
@assert is_linear(A) == true
@assert is_eye(A) == true
@assert is_diagonal(A) == true
```

## Related Packages

- **[AbstractOperators.jl](https://github.com/kul-forbes/AbstractOperators.jl)** - A comprehensive operator algebra package
- **[ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)** - Proximal algorithms using operator traits

## License

This project is licensed under the MIT License - see the LICENSE file for details.
