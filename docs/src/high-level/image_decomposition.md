# Image Decomposition

Image decomposition models the reconstructed image as a sum of additive
components, each with its own regularizer — the canonical example being
low-rank + sparse (L+S) decomposition of dynamic MRI. This is a different
concept from [Problem Decomposition](decomposition.md), which splits a
*single-image* problem over independent batch dimensions (e.g. slices); the
two can be combined (see [Interaction with Problem Decomposition](@ref
image-decomposition-problem-decomposition) below).

See [Theoretical Background](../theory.md#Additive-Image-Decomposition) for the
underlying optimization model.

## API Reference

```@docs
Component
DecomposedImage
components
total
```

## Basic Usage

```@setup imgdecomp
using MriReconstructionToolbox
using Random
Random.seed!(123)
```

Declare each component with a name and one or more regularizations, then pass
the tuple of components as the regularization argument to `reconstruct`:

```@example imgdecomp
using MriReconstructionToolbox

ksp = rand(ComplexF32, 64, 64, 4)
acq = AcquisitionInfo(ksp; is3D = false)

img = reconstruct(
    acq,
    (Component(:smooth, Tikhonov(0.01)), Component(:sparse, L1Image(0.05)));
    maxit = 30, verbose = false,
)

println(typeof(img))
println("Components: ", keys(components(img)))
```

The result is a `DecomposedImage`, which behaves as an `AbstractArray` equal
to the sum of the components:

```@example imgdecomp
using LinearAlgebra

sum(values(components(img))) ≈ total(img)
```

Individual components stay accessible via `.components`:

```@example imgdecomp
img.components.smooth isa AbstractArray
```

To get a plain, mutable array of the sum (rather than the read-only
`DecomposedImage`), use `Array`:

```@example imgdecomp
x = Array(img)
println(typeof(x))
```

## Low-Rank + Sparse (L+S)

The model image decomposition was built for is the L+S decomposition of dynamic
MRI (Otazo, Candès & Sodickson, *Magn Reson Med* 2015): a low-rank component
`L` carrying the temporally correlated background, plus a sparse component `S`
carrying the dynamic foreground.

```julia
img = reconstruct(
    acq_dynamic,
    (
        Component(:lowrank, LowRank(5e-2; time_dim = 3)),
        Component(:sparse, TemporalTotalVariation(2e-2; time_dim = 3)),
    );
    maxit = 100,
)

background = img.components.lowrank   # e.g. static anatomy
dynamics   = img.components.sparse    # e.g. contrast uptake, motion
```

Common choices for the sparse component are [`TemporalTotalVariation`](@ref)
(irregular dynamics), [`TemporalFourier`](@ref) (periodic dynamics, the
original k-t SPARSE transform) or [`L1Image`](@ref); the low-rank component is
[`LowRank`](@ref), or [`LocallyLowRank`](@ref) when the dynamics vary across
the field of view.

!!! note "Solvable combinations"
    Two components whose regularizers *both* use a non-tight operator cannot
    currently be prepared for any of the available algorithms — for example
    [`LowRank`](@ref) (whose Casorati reshape is not `is_AAc_diagonal`)
    together with [`TotalVariation2D`](@ref) or
    [`TemporalTotalVariation`](@ref) (finite differences). Working L+S pairs
    include `LowRank` + [`L1Image`](@ref)/[`TemporalFourier`](@ref)/[`L1Wavelet2D`](@ref),
    and [`LocallyLowRank`](@ref) + any of the above or `TemporalTotalVariation`
    (its proximal operator acts on the identity, so it composes freely).

## What Additive Components Are Not

Every component is an image that is *summed into the data term*: the model is
`‖E(x₁ + x₂ + …) - y‖²`. This is the right structure for L+S, for
infimal-convolution-style splittings of an image into parts with different
regularity, and for background/foreground separation.

It is *not* a general auxiliary-variable mechanism. Regularizers such as total
generalized variation (TGV) introduce an auxiliary variable that is coupled to
the image through a term like `‖∇x - w‖`, while being absent from the data term
entirely. That variable is not an additive image component, so TGV does not
follow from the `Component` API — it additionally needs a symmetrized-gradient
operator, which the operator library does not currently provide. See
[Regularizers Not Currently Available](regularization.md#Regularizers-Not-Currently-Available).

## At Least Two Components

Image decomposition requires **at least two** components — a single component
is rejected, since a one-component reconstruction is just the plain
regularization API:

```@example imgdecomp
try
    reconstruct(acq, (Component(:only, L1Image(0.05)),); verbose = false)
catch e
    println(e)
end
```

Component names must also be unique.

## Multiple Regularizations per Component

A `Component` can combine several regularizations, exactly like the plain
regularization API:

```@example imgdecomp
img_multi = reconstruct(
    acq,
    (
        Component(:structured, L1Wavelet2D(0.01), TotalVariation2D(0.005)),
        Component(:sparse, L1Image(0.05)),
    );
    maxit = 20, verbose = false,
)
nothing # hide
```

## Choosing λ per Component

Each component's regularization strength is set independently, exactly as for
the plain regularization API — there is no automatic balancing between
components. As a starting point, scale `λ` for each regularizer the same way
you would if that component were reconstructed on its own (see
[Regularization](regularization.md)), then adjust based on how much of the
signal each component should absorb.

## Initial Guess

By default, the first component is initialized with the direct (adjoint)
reconstruction and the remaining components start at zero — the standard
L+S/RPCA warm start. Override this with `x₀` as a `Tuple` (component order) or
`NamedTuple` (by component name):

```@example imgdecomp
x̂ = reconstruct(acq; verbose = false)
img_warm = reconstruct(
    acq,
    (Component(:smooth, Tikhonov(0.01)), Component(:sparse, L1Image(0.05)));
    x₀ = (smooth = x̂, sparse = zero(x̂)),
    maxit = 30, verbose = false,
)
nothing # hide
```

## Solver Applicability

The same rules that govern algorithm selection for a single image apply here,
term by term: a component with one regularization whose operator is
`is_AAc_diagonal` (e.g. `L1Image`, `L1Wavelet2D/3D`) can be solved with
FISTA/PANOC-family algorithms; a component with multiple regularizations, or a
non-tight operator, falls back to ADMM — exactly as for multiple
regularizations on a single image. Use `StructuredOptimization.print_diagnostics`
or `suggest_algorithm` to see which condition failed if a forced algorithm
errors.

## Performance Notes

- The data term applies the encoding operator `𝒜` to the *sum* of the
  component variables (`𝒜*(x₁ + x₂ + …)`), not once per component, so its
  cost matches a single-image reconstruction with the same `𝒜`.
- `disable_normalop_optimization` has no effect for image decomposition: the
  fast normal-operator path (`normalop_ls`) requires the encoding operator's
  normal operator to fuse across components (e.g. via the Toeplitz-embedded
  NFFT normal operator), which does not currently happen for a sum of shared
  operators — plain `ls` is always used instead.
- The Lipschitz constant of the data term scales with the number of
  components (for `n` components sharing a unit-norm operator `𝒜`,
  `‖[𝒜 … 𝒜]‖ = √n‖𝒜‖`), so `reconstruct` defaults `Lf = n_components` for
  FISTA/PANOC-family algorithms when reconstructing with components (instead
  of `Lf = 1` for a single image). Overriding `Lf` explicitly on the algorithm
  bypasses this default.

## [Interaction with Problem Decomposition](@id image-decomposition-problem-decomposition)

Image decomposition composes with [Problem Decomposition](decomposition.md):
if the data has batch dimensions (e.g. slices) that none of the components'
regularizations couple, `reconstruct` still decomposes the problem over those
dimensions automatically, solving each slice's image-decomposition problem
independently and stacking both the total image and each component:

```@example imgdecomp
nx, ny, nslices, nc = 32, 32, 3, 2
ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
acq_ms = AcquisitionInfo(ksp_ms; is3D = false)

img_ms = reconstruct(
    acq_ms,
    (Component(:smooth, Tikhonov(0.01)), Component(:sparse, L1Image(0.05)));
    maxit = 10, verbose = false,
)
println(size(img_ms))
println(size(img_ms.components.smooth))
```
