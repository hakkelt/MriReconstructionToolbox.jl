# Matrix-free operators

Most modeling languages *stuff* your problem: they flatten every expression into one sparse
matrix and hand it to a solver that only knows how to multiply by matrices. This package
does not. The operator you wrote stays the operator that is applied.

## No matrix stuffing

When you write `fft(x)`, StructuredOptimization holds a `DFT` operator: an object that knows
how to apply an FFT and its adjoint, and — crucially — knows a few things *about itself*.
Stuffing would replace it with a dense ``N \times N`` matrix of complex exponentials:
``O(N^2)`` storage where the operator needs ``O(N)``, and ``O(N^2)`` per application where
the FFT costs ``O(N \log N)``.

The gap is not a constant factor. For an image of a million pixels the DFT matrix does not
exist on any machine you own, while the operator applies in milliseconds. The same holds for
convolution, finite differences, wavelet transforms, and every subsampling or padding
operator — which is to say, for most of what makes an inverse problem an inverse problem.

## The operator calculus

`AbstractOperators` closes under composition, so expressions compose into operators rather
than into matrices:

| You write | You get | Applied as |
|---|---|---|
| `A * x` | the operator itself | one application |
| `A * (B * x)` | `Compose(A, B)` | `B` then `A` |
| `A * x + B * y` | `HCAT(A, B)` | one block per variable, summed |
| `ls(A*x) + ls(B*x)` | `VCAT(A, B)` | stacked codomains |
| `A * x - b` | `AffineAdd(A, -b)` | apply `A`, then subtract `b` |
| `x[1:4]` | `GetIndex` | a view, not a copy |

Each of these carries the same self-knowledge as its parts. That is what the parser queries:
`is_linear`, `is_diagonal`, `is_AAc_diagonal`, `is_full_column_rank`, the domain and codomain
sizes. None of those questions require touching an array, which is why the parse of a problem
costs microseconds regardless of how large the problem is — see the `parse/` group of
`benchmark/benchmarks.jl`.

## The normal-operator trick

For a least-squares term ``\tfrac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2``, the
gradient is

```math
\nabla f(\mathbf{x}) = \mathbf{A}^{\mathsf{H}}(\mathbf{A}\mathbf{x} - \mathbf{b})
= (\mathbf{A}^{\mathsf{H}}\mathbf{A})\mathbf{x} - \mathbf{A}^{\mathsf{H}}\mathbf{b}.
```

Written the first way, every iteration applies ``\mathbf{A}`` and then
``\mathbf{A}^{\mathsf{H}}``. Written the second, it applies the single *normal operator*
``\mathbf{A}^{\mathsf{H}}\mathbf{A}`` — and for many operators that product collapses into
something cheaper than either factor. A `MatrixOp` becomes its Gram matrix; a `DiagOp`
becomes the squared diagonal; an FFT-based convolution becomes one multiplication in the
frequency domain. [`StructuredOptimization.SqrNormL2WithNormalOp`](@ref) is that
formulation, and the parser selects it — see
[`StructuredOptimization.best_formulation`](@ref).

Two caveats, both real.

**The objective value.** The formulation computes the gradient in one pass and recovers
``f(\mathbf{x})`` *from that gradient* rather than by applying ``\mathbf{A}`` again. That is
the whole point, but it means the value depends on ``\mathbf{A}^{\mathsf{H}}`` being the true
adjoint of ``\mathbf{A}``. It is not always: a `BACKWARD`-normalized DFT has
``\mathbf{A}' = \mathbf{A}^{-1} = \mathbf{A}^{\mathsf{H}}/N``. The constructor measures that
scaling once, with a single probe, and applies it to the returned value — so anything reading
both the value and the gradient (a backtracking line search, a printed objective) sees a
consistent pair. Without that correction the two would disagree by a constant factor and the
line search would be meaningless.

**The condition number.** ``\mathbf{A}^{\mathsf{H}}\mathbf{A}`` has the *square* of
``\mathbf{A}``'s condition number. On a badly conditioned problem that is a genuine loss of
accuracy, not a bookkeeping detail, and no amount of speed compensates for it.

## When matrix-free wins

`StructuredOptimization.normal_op_worthwhile` requires the codomain to be at least as large
as the domain, which rules out an underdetermined operator. `benchmark/benchmarks.jl`
measures where that threshold sits; the numbers are recorded in the function's docstring.
The shape of the answer:

* **Tall and square operators** (``n \le m``): the fused normal operator is 2–5× faster per
  gradient, and the one-off cost of forming it is repaid in roughly fifty iterations.
* **Wide operators** (``n > m``): the per-iteration saving collapses to a few percent, within
  noise, while the construction cost keeps growing — break-even moves out to several hundred
  iterations. Add the squared condition number and it is not worth it.
* **Non-fusing operators**: when ``\mathbf{A}^{\mathsf{H}}\mathbf{A}`` stays a `Compose`, the
  formulation saves no pass at all and is never selected.

A least-squares term over several variables is the usual way to end up wide: its domain is
the sum of the blocks' domains while its codomain is shared, so two variables of size ``n``
against ``m`` residuals need ``2n \le m``.

The broader rule: matrix-free wins whenever the operator has structure a matrix would throw
away. FFTs, convolutions, finite differences and subsampling all do. A genuinely dense,
unstructured, small ``\mathbf{A}`` is the one case where stuffing would lose nothing — and
there `MatrixOp` already *is* the matrix, so nothing is lost either.
