# Theoretical Background

## MRI Forward Model

The MRI forward model describes how an image transforms into observed k-space data:

```math
y = \Gamma \mathcal{F} \mathcal{S} x + n
```

Where:
- ``x \in \mathbb{C}^{N_x \times N_y \times [N_z]}`` is the image to be reconstructed
- ``\mathcal{S}`` represents coil sensitivity weighting operator
- ``\mathcal{F}`` is the Fourier transform operator
- ``\Gamma`` is the subsampling operator
- ``y`` is the observed k-space data
- ``n`` is measurement noise

These operators are all linear maps, and they are usually represented as complex matrices in the literature. Even though, in practice, we implement them as efficient computational operators without explicitly forming large matrices, it gives theoretical background for defining adjoint operations (complex conjugate of transpose for matrices) that are essential for iterative reconstruction algorithms.

### Operator Components

#### 1. Sensitivity Map Operator (S)

Models the spatial sensitivity of receiver coils in parallel imaging:

```math
S: \mathbb{C}^{N_x \times N_y \times [N_z]} \rightarrow \mathbb{C}^{N_x \times N_y \times [N_z] \times N_c}
```

**Forward operation**: ``(Sx)_c = s_c \odot x`` (element-wise multiplication)

**Adjoint operation**: ``S^H y = \sum_{c=1}^{N_c} \bar{s}_c \odot y_c`` (coil combination)

Where ``s_c`` is the sensitivity map for coil ``c``, ``\odot`` denotes element-wise multiplication, ``N_c`` is the number of coils, and ``\bar{s}_c`` is the complex conjugate of ``s_c``.

#### 2. Fourier Transform Operator (ℱ)

Transforms between image and k-space:

```math
\mathcal{F}: \mathbb{C}^{N_x \times N_y \times [N_z]} \rightarrow \mathbb{C}^{N_x \times N_y \times [N_z]}
```

**Forward operation**: Discrete Fourier Transform (DFT)
**Adjoint operation**: Inverse DFT (scaled appropriately)

#### 3. Subsampling Operator (Γ)

Selects observed k-space locations according to an undersampling pattern:

```math
\Gamma: \mathbb{C}^{N_x \times N_y \times [N_z]} \rightarrow \mathbb{C}^{|\Omega|}
```

Where ``\Omega`` is the set of sampled k-space locations.

## Reconstruction as Inverse Problem

The most simple way of reconstructing the image ``x`` from observed data ``y`` is to apply the adjoint of the encoding operator:

```math
E^H y = \mathcal{S}^H \mathcal{F}^H \Gamma^H y
```

For fully sampled data without noise, this gives the least-squares solution. However, in practice, data is often undersampled and noisy, making direct inversion ill-posed. To address this, we formulate the reconstruction as a regularized inverse problem, formulated as:

```math
\hat{x} = \arg\min_x \frac{1}{2}\|Ex - y\|_2^2 + \lambda R(x)
```

Where:
- ``\frac{1}{2}\|Ex - y\|_2^2`` is the data fidelity term
- ``R(x)`` is a regularization term (e.g., sparsity, total variation)
- ``\lambda`` controls the regularization strength

The encoding operator ``E`` makes this optimization problem well-defined and computationally tractable.

### Regularization Techniques

The regularization term ``R(x)`` can take various forms depending on the desired image properties:
- **L2 Regularization**: ``R(x) = \|x\|_2^2`` -> Promotes smoothness
- **L1 Wavelet Regularization**: ``R(x) = \|\Psi x\|_1`` where ``\Psi`` is a wavelet transform -> Promotes sparsity in the wavelet domain
- **Total Variation (TV)**: ``R(x) = \|\nabla x\|_1`` -> Preserves edges while reducing noise

The regularization terms are represented by a (linear) operator and a norm function in general:
```math
R(x) = f(\mathcal{T} x)
```
where ``\mathcal{T}`` is some transform operator and ``f`` is a norm function (e.g., L1, L2, nuclear norm, etc.).

### Optimization Algorithms

Various iterative algorithms can be employed to solve the regularized inverse problem, most commonly:
- **Gradient Descent** (usually conjugate gradient): If ``R(x)`` is differentiable
- **Proximal Gradient Methods**: For non-differentiable ``R(x)``

Proximal gradient methods alternate between gradient descent on the data fidelity term and applying the proximal operator of the regularization term. The proximal operator is defined as:
```math
\text{prox}_{\alpha R}(v) = \arg\min_x \frac{1}{2}\|x - v\|_2^2 + \alpha R(x)
```
While it is inefficient to solve this minimization directly, many common regularization terms have closed-form proximal operators. For example, the proximal operator for L1 regularization is soft-thresholding:
```math
\text{prox}_{\alpha \|\cdot\|_1}(v) = \text{sign}(v) \odot \max(|v| - \alpha, 0)
```
