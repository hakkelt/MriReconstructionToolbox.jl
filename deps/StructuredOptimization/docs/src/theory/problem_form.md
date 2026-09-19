# Problem form & algorithms

Every problem StructuredOptimization can solve is a sum of terms of the form

```math
\lambda \, f(\mathbf{A}\mathbf{x} + \mathbf{d})
```

where ``f`` is a function from [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl),
``\mathbf{A}`` is an [AbstractOperators.jl](https://github.com/JuliaFirstOrder/AbstractOperators.jl)
operator (linear or not), ``\mathbf{d}`` an additive displacement and ``\lambda`` a scalar
weight. A constraint is a term whose ``f`` is the *indicator* of a set. Writing
`ls(A*x - b) + 1e-2*norm(x, 1)` builds two such terms; `problem(...)` collects them into a
[`TermSet`](@ref StructuredOptimization.TermSet).

What the solvers actually accept is narrower. The classical composite form is

```math
\operatorname*{minimize}_{\mathbf{x}} \quad f(\mathbf{A}\mathbf{x}) + g(\mathbf{x}),
```

with ``f`` smooth and ``g`` proximable. Which of your terms can play the part of ``f``, and
which of ``g``, is decided by the properties below. [How problems are parsed](@ref) explains
the rewriting that gets you from the first form to the second; this page defines the
vocabulary that rewriting uses.

## The properties that matter

These are the predicates the parser queries. They are *structural* — derived from how a term
was built, never from sampling the function — which is what makes a "cannot parse this
problem" answer trustworthy.

**Convex.** ``f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \le \theta f(\mathbf{x}) +
(1-\theta) f(\mathbf{y})``. Convexity survives composition with a *linear* operator, so
`norm(A*x, 1)` is convex; it does not survive composition with a nonlinear one, so
`ls(sin(x) - b)` is not. Algorithms differ in whether they need it:
`FastForwardBackward` does, `PANOCplus` and `ZeroFPR` do not.

**Strongly convex.** Convex with a quadratic margin: ``f - \tfrac{\mu}{2}\|\cdot\|^2`` is
still convex for some ``\mu > 0``. For ``\tfrac{1}{2}\|\mathbf{A}\mathbf{x}\|^2`` this needs
``\mathbf{A}`` to have full column rank — otherwise the function is flat along the null
space, and no positive ``\mu`` works.

**Smooth.** Differentiable with a Lipschitz gradient, which is what a first-order method
needs to take a gradient step. Every algorithm here has at most one smooth slot.

**Proximable.** The proximal operator

```math
\operatorname{prox}_{\gamma f}(\mathbf{v}) = \operatorname*{arg\,min}_{\mathbf{x}}
\Big\{ f(\mathbf{x}) + \tfrac{1}{2\gamma}\|\mathbf{x} - \mathbf{v}\|^2 \Big\}
```

is available in closed form and cheap. This is the property that most often blocks a parse,
because *composition destroys it*: knowing ``\operatorname{prox}_f`` tells you nothing about
``\operatorname{prox}_{f \circ A}`` in general. The exceptions are exactly the absorptions
in [How problems are parsed](@ref).

**Generalized quadratic.** ``f`` is a quadratic plus a linear term, so its gradient is
affine. Some line searches exploit this to avoid re-evaluating the objective.

**Set indicator.** ``f = \delta_C``, zero on ``C`` and ``+\infty`` off it. A constraint is a
term of this shape, and its prox is the projection onto ``C`` — which is why
`norm(x) <= 1` costs no more than a norm evaluation.

## Three proximal operators worked out

The closed forms below are what "proximable" buys, and each is the engine of a constraint or
regularizer you are likely to write.

**``\ell_1`` norm**, ``f(\mathbf{x}) = \lambda\|\mathbf{x}\|_1`` — soft thresholding,
elementwise:

```math
[\operatorname{prox}_{\gamma f}(\mathbf{v})]_i =
\operatorname{sign}(v_i)\,\max(|v_i| - \gamma\lambda,\, 0).
```

This is `norm(x, 1)`, and the shrinkage toward zero is why an ``\ell_1`` penalty produces
exactly-zero coefficients rather than merely small ones.

**Euclidean ball**, ``C = \{\mathbf{x} : \|\mathbf{x}\|_2 \le r\}`` — projection by
rescaling:

```math
\operatorname{prox}_{\gamma \delta_C}(\mathbf{v}) =
\begin{cases}
\mathbf{v}, & \|\mathbf{v}\|_2 \le r,\\[2pt]
r\,\mathbf{v}/\|\mathbf{v}\|_2, & \text{otherwise.}
\end{cases}
```

This is `norm(x, 2) <= r`. Note it does not depend on ``\gamma``: the prox of any indicator
is a projection, and projections do not have a step size.

**Squared Euclidean norm**, ``f(\mathbf{x}) = \tfrac{\lambda}{2}\|\mathbf{x}\|^2`` — uniform
shrinkage:

```math
\operatorname{prox}_{\gamma f}(\mathbf{v}) = \frac{\mathbf{v}}{1 + \gamma\lambda}.
```

This is `ls(x)`, and being both smooth *and* proximable is what lets a least-squares term be
placed in either slot — the choice the parser makes by cost.

## The algorithm classes

Each algorithm in ProximalAlgorithms advertises the problem shape it assumes, and
StructuredOptimization matches your terms against it. The table below is generated from
those declarations at documentation build time, so it cannot drift from the code:

```@example assumptions
using ProximalAlgorithms, Markdown

rows = String[]
for alg in ProximalAlgorithms.get_algorithms()
    push!(rows, "| `$(typeof(alg).name.name)` | " *
                replace(sprint(show, ProximalAlgorithms.get_assumptions(alg)), "|" => "\\|") * " |")
end
Markdown.parse(join(vcat("| Algorithm | Assumed problem form |", "|---|---|", rows), "\n"))
```

Reading the table: `f is_smooth; and A is_linear; and g is_proximable` is the composite form
above — a smooth function of a linear map, plus a proximable function of the variable
itself. An algorithm with only a proximable slot cannot take a least-squares term unless its
operator absorbs; one with only a smooth slot cannot take an ``\ell_1`` penalty at all.

[`suggest_algorithm`](@ref) reports which of these your problem satisfies, and
[`print_diagnostics`](@ref StructuredOptimization.print_diagnostics) reports, term by term,
which property blocked the rest.

## Where to go next

* [How problems are parsed](@ref) — the rewriting from `λ·f(Ax+d)` triples to a solver call,
  and the operator absorptions that decide proximability.
* [Matrix-free operators](@ref) — why ``\mathbf{A}`` is never stuffed into a matrix, and when
  that wins.
* [FAQ & Troubleshooting](@ref) — what to do when nothing parses.
