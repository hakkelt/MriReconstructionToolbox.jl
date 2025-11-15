# MRI Forward Model

Understanding the MRI forward model is crucial for effective reconstruction. This page explains the mathematical foundation and implementation details.

## Mathematical Foundation

The MRI forward model describes how an image transforms into observed k-space data:

```math
y = \Gamma \mathcal{F} S x + n
```

Where:
- ``x \in \mathbb{C}^{N_x \times N_y \times [N_z]}`` is the image to be reconstructed
- ``S \in \mathbb{C}^{N_x \times N_y \times [N_z] \times N_c}`` represents coil sensitivity maps
- ``\mathcal{F}`` is the Fourier transform operator
- ``\Gamma`` is the subsampling operator
- ``y`` is the observed k-space data
- ``n`` is measurement noise

## Operator Components

### 1. Sensitivity Map Operator (S)

Models the spatial sensitivity of receiver coils in parallel imaging:

```math
S: \mathbb{C}^{N_x \times N_y \times [N_z]} \rightarrow \mathbb{C}^{N_x \times N_y \times [N_z] \times N_c}
```

**Forward operation**: ``(Sx)_c = s_c \odot x`` (element-wise multiplication)

**Adjoint operation**: ``S^H y = \sum_{c=1}^{N_c} \bar{s}_c \odot y_c`` (coil combination)

Where ``s_c`` is the sensitivity map for coil ``c`` and ``\odot`` denotes element-wise multiplication.

### 2. Fourier Transform Operator (F)

Transforms between image and k-space:

```math
\mathcal{F}: \mathbb{C}^{N_x \times N_y \times [N_z]} \rightarrow \mathbb{C}^{N_x \times N_y \times [N_z]}
```

**Forward operation**: Discrete Fourier Transform (DFT)
**Adjoint operation**: Inverse DFT (scaled appropriately)

### 3. Subsampling Operator (Γ)

Selects observed k-space locations according to an undersampling pattern:

```math
\Gamma: \mathbb{C}^{N_x \times N_y \times [N_z]} \rightarrow \mathbb{C}^{|\Omega|}
```

Where ``\Omega`` is the set of sampled k-space locations.

## Implementation Details

### Operator Composition

The encoding operator is implemented as a composition:

```julia
E = Γ * F * S  # For undersampled parallel imaging
E = F * S      # For fully sampled parallel imaging  
E = Γ * F      # For undersampled single-coil
E = F          # For fully sampled single-coil
```

### Memory Efficiency

The operators are designed for memory efficiency:

- **Lazy evaluation**: Operations are composed without intermediate allocations
- **In-place operations**: When possible, computations modify existing arrays
- **Batch processing**: Multiple time points or contrasts processed efficiently

### Adjoint Operations

All operators have properly defined adjoints for reconstruction:

```julia
# Forward model
y = E * x

# Adjoint (often used as initial estimate)
x_init = E' * y

# The adjoint satisfies: ⟨E*x, y⟩ = ⟨x, E'*y⟩ for all x, y
```

## Dimension Handling

### Spatial Dimensions

- **2D**: ``(N_x, N_y)`` spatial dimensions
- **3D**: ``(N_x, N_y, N_z)`` spatial dimensions

### Batch Dimensions

Additional dimensions are treated as batch dimensions:

- **Time**: Dynamic imaging with temporal dimension
- **Contrast**: Multiple contrast weightings
- **Slice**: Multi-slice 2D imaging

Example with batch dimensions:
```julia
# Multi-slice, multi-contrast data
ksp = rand(ComplexF32, 64, 64, 8, 20, 4)  # (kx, ky, coil, slice, contrast)
smaps = rand(ComplexF32, 64, 64, 8)       # (x, y, coil)

E = get_encoding_operator(ksp, false; smaps=smaps)
# Automatically handles batch dimensions
```

## Physical Interpretation

### Sensitivity Maps

Sensitivity maps encode the spatial response of each coil:

- **High sensitivity**: Strong signal reception in that region
- **Low sensitivity**: Weak signal reception
- **Phase variations**: Account for coil positioning and B₀ field inhomogeneities

### K-space Sampling

Different sampling strategies have different properties:

- **Cartesian**: Regular grid sampling, simple reconstruction
- **Radial**: Spoke-like sampling, robust to motion
- **Spiral**: Efficient coverage, requires gridding
- **Random**: Incoherent sampling for compressed sensing

### Undersampling Effects

Undersampling in k-space leads to artifacts in image space:

- **Aliasing**: Spatial wraparound artifacts
- **Incoherent artifacts**: Noise-like artifacts (with random sampling)
- **Coherent artifacts**: Structured artifacts (with regular sampling)

## Reconstruction as Inverse Problem

The reconstruction problem is formulated as:

```math
\hat{x} = \arg\min_x \frac{1}{2}\|Ex - y\|_2^2 + \lambda R(x)
```

Where:
- ``\frac{1}{2}\|Ex - y\|_2^2`` is the data fidelity term
- ``R(x)`` is a regularization term (e.g., sparsity, total variation)
- ``\lambda`` controls the regularization strength

The encoding operator ``E`` makes this optimization problem well-defined and computationally tractable.
