# [AbstractOperators.jl: Matrix-Free Linear Operators](@id abstract_operators)

## Why Matrix-Free Operators?

### The Matrix Representation Problem

In image processing and MRI reconstruction, operators are typically linear maps that can be theoretically represented as matrices. For example:

- **Fourier Transform**: $\mathcal{F}: \mathbb{C}^{N \times M} \to \mathbb{C}^{N \times M}$
- **Subsampling**: $\Gamma: \mathbb{C}^{N \times M} \to \mathbb{C}^{K}$ (where $K < N \times M$)
- **Sensitivity Encoding**: $\mathcal{S}: \mathbb{C}^{N \times M} \to \mathbb{C}^{N \times M \times N_c}$

In traditional linear algebra, these would be represented as matrices $A$ where:

```math
y = A x
```

where $x$ is the **flattened** input image and $y$ is the flattened output.

### The Good: Rich Algorithm Ecosystem

Matrix representations have significant advantages:

✅ **Wide range of algorithms**: Conjugate Gradient, LSQR, GMRES, and many other iterative solvers work with matrices.

✅ **Theoretical foundation**: Linear algebra theory provides convergence guarantees, preconditioning strategies, and optimality conditions.

✅ **Composability**: Operators can be easily composed ($C = A \cdot B$), added ($D = A + B$), scaled, transposed, etc.

✅ **Standard interface**: Most numerical libraries understand matrices.

### The Bad: Computational Inefficiency

However, explicit matrix representations have critical drawbacks for image processing:

❌ **Memory explosion**: An $N \times M$ image treated as a vector of length $N \cdot M$ requires a matrix of size $(N \cdot M) \times (N \cdot M)$ to represent an operator.

**Example**: For a modest $256 \times 256$ image:
- Flattened vector: $256 \times 256 = 65{,}536$ elements
- Matrix representation: $65{,}536 \times 65{,}536 = 4{,}294{,}967{,}296$ elements
- Memory (Float64): ~34 GB just for **one** operator!

❌ **Loss of structure**: Flattening an image discards its natural 2D/3D structure, making many operations less efficient.

❌ **Slow operations**: Matrix-vector multiplication is $\mathcal{O}(N^2)$, while FFT on structured data is $\mathcal{O}(N \log N)$.

### The Solution: Matrix-Free Operators

**Matrix-free operators** are objects that:

- Act like matrices (support `*`, `'`, composition, addition)  
- Work directly on multi-dimensional arrays (no flattening needed)  
- Implement efficient algorithms internally (FFT, convolution, sparse indexing)  
- Use minimal memory (store only parameters, not full matrix)

This is exactly what [AbstractOperators.jl](https://github.com/kul-forbes/AbstractOperators.jl) provides.

## Introducing AbstractOperators.jl

[AbstractOperators.jl](https://github.com/kul-forbes/AbstractOperators.jl) is a Julia package that provides a comprehensive framework for matrix-free linear operators with:

- **Rich operator library**: Fourier transforms, convolutions, finite differences, wavelets, and more
- **Matrix-like interface**: Supports `*`, `'`, `+`, `∘`, and composition
- **Multi-dimensional arrays**: Works directly on images without flattening
- **Performance optimizations**: Multi-threading, SIMD, operator fusion
- **Extensibility**: Easy to define custom operators

### Key Features

| Feature | Benefit |
|---------|---------|
| **Matrix-free** | Minimal memory footprint |
| **Lazy composition** | Combines operators without intermediate allocations |
| **Multi-threading** | Automatic parallelization where supported |
| **Type stability** | Full Julia type inference for performance |

## Basic Usage

Note: Code blocks below are illustrative and not executed during docs build. To run them locally, activate a Julia environment and install the required packages:

```julia
using Pkg

# Add the package from GitHub
Pkg.add(url="https://github.com/hakkelt/AbstractOperators.jl")
Pkg.add(url="https://github.com/hakkelt/AbstractOperators.jl", subdir="FFTWOperators")
Pkg.add(url="https://github.com/hakkelt/AbstractOperators.jl", subdir="WaveletOperators")
```

### Simple Operator Example

```@setup imports
using AbstractOperators, FFTW, FFTWOperators, LinearAlgebra
```

```@repl imports
using AbstractOperators, FFTW, FFTWOperators, LinearAlgebra

nx, ny = 64, 64
F = ℱ = DFT(ComplexF32, (nx, ny)) # A 2D DFT operator for 64x64 images

x = randn(ComplexF32, nx, ny);
y = ℱ * x;  # Apply forward transform
y == fft(x)  # Verify correctness

x_back = F' * y;
x_back == bfft(x)  # Verify correctness

scale = 1 / (nx * ny) # Normalization factor that makes F unitary
norm(x - scale * x_back) / norm(x) < 1e-6 # Verify it's an approximate inverse
```

### Operator Composition

Operators can be composed naturally:

```@repl imports
mask = rand(Bool, nx, ny); mask[32-5:32+5, 32-5:32+5] .= true;  # Fully sample center
sum(mask) # Count of sampled points

Γ = GetIndex(zeros(ComplexF32, nx, ny), (mask,))

ℱ_sub = Γ * ℱ # Compose: subsampled Fourier transform

x = randn(ComplexF32, nx, ny);
y = ℱ_sub * x; # Apply the composition
y == fft(x)[mask]  # Verify correctness
```

### Operator Algebra

AbstractOperators supports rich algebraic operations:

```@repl imports
I_op = Eye(ComplexF32, (nx, ny)) # Identity operator

A = 2 * ℱ # Scaling

B = A + I_op # Addition of outputs

d = randn(ComplexF32, nx, ny);
D = DiagOp(d) # Diagonal scaling, i.e., element-wise multiplication with d

C = D * B  # Composition / chaining of operators

x = randn(ComplexF32, nx, ny);
y = C * x; # Apply composed operator
y == d .* (2 * fft(x) + x)  # Verify correctness
```

### Batch Operations

Operators can work on batched inputs (e.g., multi-coil data):

```julia
# Sensitivity map operator (per-coil scaling)
nc = 8  # 8 coils
smaps = randn(ComplexF32, nx, ny, nc)
𝒮 = DiagOp(smaps)

# Encoding operator: sensitivity * FFT
𝒜 = ℱ * 𝒮

# Apply to multi-coil image
x = randn(ComplexF32, nx, ny, nc)
ksp = 𝒜 * x
println("Image size: $(size(x)), k-space size: $(size(ksp))")
```

## Creating Custom Operators

You can easily define custom operators for domain-specific operations. Here's a minimal example:

```julia
using AbstractOperators

# Define the operator struct
struct MyCustomLinOp{N,M,D,C} <: LinearOperator
    dim_in::NTuple{M,Int}
    dim_out::NTuple{N,Int}
    # Add any additional fields needed for your operator
end

# Mandatory functions

# 1. size: Return (codomain_size, domain_size)
Base.size(L::MyCustomLinOp) = (L.dim_out, L.dim_in)

# 2. domain_type: Return the element type of the input
AbstractOperators.domain_type(::MyCustomLinOp{N,M,D,C}) where {N,M,D,C} = D

# 3. codomain_type: Return the element type of the output
AbstractOperators.codomain_type(::MyCustomLinOp{N,M,D,C}) where {N,M,D,C} = C

# 4. fun_name: Return a string/symbol for display purposes
AbstractOperators.fun_name(::MyCustomLinOp) = "MyOp"

# 5. mul!: Forward operator (output, operator, input)
function LinearAlgebra.mul!(y::AbstractArray, L::MyCustomLinOp, x::AbstractArray)
    # Utility function to check if
    #   - eltype(x) == domain_type(L)
    #   - eltype(y) == codomain_type(L)
    #   - size(x) == size(L, 2)
    #   - size(y) == size(L, 1)
    #   - x isa domain_storage_type(L)
    #   - y isa codomain_storage_type(L)
    AbstractOperators.check(y, L, x)
    # Implement your forward operation here
    # Example: y .= some_function(x)
    return y
end

# 6. mul! for adjoint: Adjoint operator (output, adjoint_operator, input)
function LinearAlgebra.mul!(y::AbstractArray, L::AdjointOperator{<:MyCustomLinOp}, x::AbstractArray)
    AbstractOperators.check(y, L, x) # Utility function to check types and sizes
    # Implement your adjoint operation here
    # Example: y .= adjoint_function(x)
    return y
end
```

**Key points for custom operators:**

1. **Inherit from the right type**: `LinearOperator` for linear maps, `AbstractOperator` for nonlinear
2. **Implement required interface**: `size`, `domain_type`, `codomain_type`, `fun_name`, `mul!`
3. **Implement adjoint**: For `LinearOperator`, also implement `mul!` for `AdjointOperator`
4. **Test correctness**: Verify `⟨Lx, y⟩ = ⟨x, L'y⟩` for random inputs

For complete details on implementing custom operators, see the [AbstractOperators.jl documentation](https://hakkelt.github.io/AbstractOperators.jl/stable/custom/).

## Operators in AbstractOperators.jl

AbstractOperators.jl provides a rich library of pre-built operators, from which many are directly useful in MRI reconstruction:

### Transform Operators
- **`DFT`**: Discrete Fourier Transform (via FFTWOperators.jl)
- **`DCT`**: Discrete Cosine Transform (via FFTWOperators.jl)
- **`RDFT`**: Real-to-complex FFT (via FFTWOperators.jl)
- **`WaveletOp`**: Wavelet transforms (via WaveletOperators.jl)
- **`NFFTOp`**: Non-uniform FFT (via NFFTOperators.jl)

### Linear Operators
- **`Eye`**: Identity operator
- **`DiagOp`**: Diagonal operator (element-wise multiplication)
- **`MatrixOp`**: Wraps a regular matrix
- **`FiniteDiff`**: Finite difference (gradients)
- **`Variation`**: Total variation operator
- **`GetIndex`**: Subsampling/indexing operator
- **`ZeroPad`**: Zero-padding operator

### Calculus
- **`Scale`**: Scalar multiplication
- **`Compose`**: Operator composition
- **`Sum`**: Sum outputs of different operators applying to the same input
- **`BroadCast`**: Broadcasting operator -- repeats the output of an operator across specified dimensions
- **`Reshape`**: Reshape arrays

### Nonlinear Operators
- **`Sigmoid`**, **`Tanh`**, **`Exp`**, **`Sin`**, **`Cos`**: Activation functions
- **`SoftMax`**, **`SoftPlus`**: Softmax and softplus

### Signal Processing
- **`Conv`**: Convolution (via DSPOperators.jl)
- **`Filt`**: Filtering (via DSPOperators.jl)
- **`Xcorr`**: Cross-correlation (via DSPOperators.jl)

## Integration with MriReconstructionToolbox

MriReconstructionToolbox.jl is built entirely on AbstractOperators.jl. All encoding operators (`get_encoding_operator`, `get_fourier_operator`, etc.) return AbstractOperators objects, giving you:

- **Seamless integration**: Use operators with any algorithm that supports the interface
- **Flexibility**: Extract and manipulate individual components of the encoding chain
- **Performance**: Benefit from all AbstractOperators optimizations
- **Easy debugging**: Inspect and visualize operators at any stage
- **Extensibility**: Add custom operators to the reconstruction pipeline

### Extracting Operators

```julia
# Get the full encoding operator
𝒜 = get_encoding_operator(acq_info)

# Or get individual components
ℱ = get_fourier_operator(acq_info)
𝒮 = get_sensitivity_map_operator(acq_info)
ℳ = get_subsampling_operator(acq_info)

# Manually compose them
𝒜_manual = ℳ * ℱ * 𝒮

# Both are equivalent
@assert 𝒜 * x ≈ 𝒜_manual * x
```

### Custom Reconstruction Pipeline

```julia
𝒮 = MyCustomSensitivityMapOperator(...)
ℱ = get_fourier_operator(acq_info)
ℳ = get_subsampling_operator(acq_info)
𝒜 = ℳ * ℱ * 𝒮

# Add custom regularization operator
𝒲 = MyCustomWavelet(...)  # Your custom operator

# Build optimization problem
using StructuredOptimization
v = Variable(𝒜' * ksp)  # Initial guess

# Solve custom problem
x̂, _ = @minimize ls(𝒜 * v - ksp) + 0.01 * norm(𝒲 * v, 1)
```

## Further Reading

- **AbstractOperators.jl Documentation**: [https://hakkelt.github.io/AbstractOperators.jl/stable/](https://hakkelt.github.io/AbstractOperators.jl/stable/)
- **Custom Operators Guide**: [https://hakkelt.github.io/AbstractOperators.jl/stable/custom/](https://hakkelt.github.io/AbstractOperators.jl/stable/custom/)
- **Operator Properties**: Learn about [operator traits and properties](https://hakkelt.github.io/AbstractOperators.jl/stable/properties/)
