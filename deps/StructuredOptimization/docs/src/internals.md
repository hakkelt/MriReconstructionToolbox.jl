# Internals

These functions are not exported and carry no compatibility promise. They are documented
because they are where the interesting decisions are made: the reference below is the
companion to [How problems are parsed](@ref) and [Matrix-free operators](@ref), and the
place to look when a problem parses into a formulation you did not expect.

## Choosing a formulation

A term is `λ · f(A·x + d)`. Turning that into a single function the solver can call is an
*absorption*, and there are several ways to do it — with different costs, and not all of
them keeping an exact proximal operator. The choice is scored, not fixed.

```@docs
StructuredOptimization.best_formulation
StructuredOptimization.merge_function_with_operator
StructuredOptimization.keeps_exact_prox
StructuredOptimization.is_aac_diagonal
```

## Choosing an algorithm and a term split

The formulation choice and the algorithm choice are scored together: a cheaper formulation is
only an improvement if the algorithm that ends up selected can use it.

```@docs
StructuredOptimization.match_assumption
StructuredOptimization.parse_failure_message
```

## The normal-operator formulation

```@docs
StructuredOptimization.SqrNormL2WithNormalOp
StructuredOptimization.with_normal_op
StructuredOptimization.fused_normal_op
StructuredOptimization.normal_op_fuses
StructuredOptimization.normal_op_applicable
StructuredOptimization.normal_op_worthwhile
```

## Non-linear compositions

```@docs
PrecomposeNonlinear
```
