# Functions and constraints

Once an expression is created it is possible to create the `Term`s defining the optimization problem.
These can consists of either [Smooth functions](@ref),  [Nonsmooth functions](@ref), [Inequality constraints](@ref)
or [Equality constraints](@ref).

## Smooth functions

```@docs
ls
huberloss
sqrhingeloss
crossentropy
logisticloss
dot
```

## Nonsmooth functions

```@docs
norm
maximum
sumpositive
hingeloss
logbarrier
^
```

## Inequality constraints

```@docs
<=
rank
```

## Equality constraints

```@docs
==
```

## Smoothing

Sometimes the optimization problem might involve non-smooth terms which
do not have efficiently computable proximal mappings.
It is possible to *smoothen* these terms by means of the *Moreau envelope*.

```@docs
smooth
```

## Duality

In some cases it is more convenient to solve the *dual problem* instead
of the primal problem. It is possible to convert a problem into its dual
by means of the *convex conjugate*.

See the [Total Variation demo](https://github.com/hakkelt/StructuredOptimization.jl/blob/master/demos/TotalVariationDenoising.ipynb) for an example of such procedure.

```@docs
conj
```

## Collecting terms

Terms add up into a `TermSet`, which is what [`problem`](@ref) returns and what
[`solve`](@ref) takes.

```@docs
StructuredOptimization.TermSet
```

## Recording terms for diagnostics

The [`@term`](@ref) macro captures the source text of a term so it can be shown
later (e.g. by [`print_diagnostics`](@ref StructuredOptimization.print_diagnostics))
when a problem cannot be parsed.

```@docs
@term
```
