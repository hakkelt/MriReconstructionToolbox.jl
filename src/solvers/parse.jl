function add_to_incompatibilities(incompatibilities, t1, t2)
    if haskey(incompatibilities, t1)
        push!(incompatibilities[t1], t2)
    else
        incompatibilities[t1] = Set([t2])
    end
    return if haskey(incompatibilities, t2)
        push!(incompatibilities[t2], t1)
    else
        incompatibilities[t2] = Set([t1])
    end
end

function group_by_variables(terms)
    variable_bags = Dict{Variable, Vector{Any}}()
    for term in terms
        for var in variables(term)
            if haskey(variable_bags, var)
                push!(variable_bags[var], term)
            else
                variable_bags[var] = [term]
            end
        end
    end
    return variable_bags
end

function can_be_separable_sum(variable_bags)
    for (var, term_list) in variable_bags
        if length(term_list) > 1 # more than one term for this variable
            all(splits_per_variable, term_list) || return false
            # Check if any of the terms are sliced
            operators = [get_operators_for_var(term, var) for term in term_list]
            slicing_masks = [is_sliced(op) ? AbstractOperators.get_slicing_mask(op) : nothing for op in operators]
            for i in eachindex(operators)
                if is_sliced(operators[i])
                    # This operator is sliced, check if it is overlapping with any other sliced operator
                    for j in (i + 1):length(operators)
                        if is_sliced(operators[j]) && any(slicing_masks[i] .&& slicing_masks[j])
                            return false
                        end
                    end
                else # no slicing -> this term is incompatible with all others
                    return false
                end
            end
        end
    end
    return true
end

function get_unseparable_pairs(variable_bags)
    incompatibilities = Dict{Term, Set{Term}}()
    for (var, term_list) in variable_bags
        if length(term_list) > 1 # more than one term for this variable
            # A term that does not split per variable is incompatible with every other term
            # sharing this variable, regardless of slicing (see `splits_per_variable`).
            for t in term_list, other in term_list
                (t === other || splits_per_variable(t)) && continue
                add_to_incompatibilities(incompatibilities, t, other)
            end
            # Check if any of the terms are sliced
            operators = [get_operators_for_var(term, var) for term in term_list]
            slicing_masks = [is_sliced(op) ? AbstractOperators.get_slicing_mask(op) : nothing for op in operators]
            for i in eachindex(operators)
                if is_sliced(operators[i])
                    # This operator is sliced, check if it is overlapping with any other sliced operator
                    for j in (i + 1):length(operators)
                        if is_sliced(operators[j]) && any(slicing_masks[i] .&& slicing_masks[j])
                            add_to_incompatibilities(incompatibilities, term_list[i], term_list[j])
                        end
                    end
                else # no slicing -> this term is incompatible with all others
                    for j in (i + 1):length(operators)
                        add_to_incompatibilities(incompatibilities, term_list[i], term_list[j])
                    end
                end
            end
        end
    end
    return incompatibilities
end

# The dense matrix behind `op`, or `nothing` when `op` is not a plain `MatrixOp`. Only a
# stored matrix can be handed to `IndAffine`, which needs to factorise it.
_matrix_of(op) = nothing
_matrix_of(op::MatrixOp) = op.A
_matrix_of(op::AbstractOperators.AffineAdd) = _matrix_of(AbstractOperators.remove_displacement(op))

"""
    is_aac_diagonal(op)

`AbstractOperators.is_AAc_diagonal(op)`, answered cheaply where that matters.

For every structured operator the upstream predicate is a type-level trait and costs
nothing. For a `MatrixOp` it is `isdiag(A * Aᴴ)`: `O(m²n)` work and an `m × m` temporary,
more than several iterations of the solver being selected — unaffordable in a predicate the
parser consults for every candidate formulation of every term subset.

`A * Aᴴ` is diagonal exactly when the rows of `A` are pairwise orthogonal, and a *single*
non-orthogonal pair disproves it. A handful of pairs are therefore tested first, in `O(n)`
each; anything that is not genuinely AAᴴ-diagonal — the overwhelmingly common case — is
rejected there. Only a matrix that survives the sample falls through to the full check, so
the answer is identical to the upstream one, never merely an approximation of it.
"""
is_aac_diagonal(op) = is_AAc_diagonal(op)
is_aac_diagonal(op::AbstractOperators.AffineAdd) = is_aac_diagonal(AbstractOperators.remove_displacement(op))
function is_aac_diagonal(op::MatrixOp)
    A = op.A
    m = size(A, 1)
    m <= 1 && return is_AAc_diagonal(op)
    # `isdiag` compares against exact zero, so use the same test here — the sample must
    # reject only matrices the full check would reject too.
    for i in 1:min(m - 1, 4), j in (i + 1):min(m, i + 4)
        iszero(dot(view(A, i, :), view(A, j, :))) || return false
    end
    return is_AAc_diagonal(op)
end

"""
    keeps_exact_prox(op, f)

Whether absorbing `op` into `f` (see [`merge_function_with_operator`](@ref)) leaves a
function whose `prox!` is still the exact proximal operator of the composition.

This mirrors the branch table of `merge_function_with_operator`: the identity, diagonal and
AAᴴ-diagonal absorptions all have a closed-form prox (the "prox trick" — `is_aac_diagonal`
covers the first two, since `Eye` and `DiagOp` are both AAᴴ-diagonal), and so does the
`IndPoint` + `MatrixOp` rewrite into `IndAffine`. Everything below that — the normal-operator
formulation, `Precompose` with a general linear operator, `PrecomposeNonlinear` — implements
only a gradient, or a `prox!` that is not the prox of the composed function; a solver picked
on the strength of a prox it does not have would fail at the first iteration.

`op` may carry a displacement (`affine(term)`); it does not affect the answer.
"""
keeps_exact_prox(op, f) = is_aac_diagonal(op) || (f isa IndPoint && _matrix_of(op) !== nothing)

# The cost of the `:normal_op` candidate in `best_formulation` below. `n / m` prices it as a
# dense-matrix Gram construction, which is the honest estimate for everything except a
# multi-variable `HCAT` whose blocks share one encoding operator: there the shared operator's
# own fast normal operator is reused rather than reconstructed, so forming it is cheap
# regardless of shape. See [`reuses_optimized_normalop`](@ref) for why the bypass is not taken
# on the strength of `has_optimized_normalop` alone.
normal_op_cost(op, n, m) = n / m
function normal_op_cost(op::AbstractOperators.HCAT, n, m)
    reuses_optimized_normalop(op) && return 1.0
    inner = _drop_zero_blocks(op)
    return inner === op ? n / m : normal_op_cost(inner, _total_length(size(inner, 2)), m)
end

"""
    best_formulation(op, f, disp, λ, needs = :any) -> (kind::Symbol, cost::Float64)

Score every way this package can express `λ · f(op·x + disp)` as a single function and
return the winner. `needs === :prox` restricts the search to formulations whose `prox!` is
the exact proximal operator of the composition; `:any` accepts a gradient-only one as well.
`(:none, Inf)` means no formulation qualifies, which only happens under `needs === :prox`.

# The cost model

Costs are in units of *one application of `op` plus one of `opᴴ`* — the work a first-order
method does for this term in one iteration — normalised so that the generic formulation,
`Precompose(f, op, 1, disp)`, costs `2`. `n = prod(domain)` and `m = prod(codomain)`:

| kind | applies when | keeps prox | cost | why |
|---|---|---|---|---|
| `:eye` | `is_eye(op)` | yes | `0` | no operator is applied at all |
| `:diagonal_weight` | diagonal `op`, `f::SqrNormL2`, no displacement | yes | `0` | `½‖diag(a)x‖²` *is* the weighted `½∑aᵢ²xᵢ²`; the operator disappears |
| `:diagonal` | `is_diagonal(op)` | yes | `1` | one elementwise pass, no adjoint |
| `:aac_diagonal` | `is_aac_diagonal(op)` | yes | `2` | the "prox trick": `op` and `opᴴ` once each |
| `:ind_affine` | `f::IndPoint`, `op` a `MatrixOp` | yes | `2.5` | a QR factorisation amortised over a triangular solve per prox |
| `:normal_op` | `f::SqrNormL2`, `opᴴop` fuses and is worthwhile | no | `n/m` (`1` for a shared-operator `HCAT`) | one fused `opᴴop` pass on the domain instead of two passes through `op` |
| `:precompose` | `is_linear(op)` | no | `2` | `op` then `opᴴ`, the generic linear case |
| `:nonlinear` | always | no | `2` | `op` then its Jacobian adjoint |

A formulation that keeps an exact prox is preferred over a cheaper one that does not, which
is why the key is `(keeps_prox ? 0 : 1, cost)` rather than the cost alone. That is a real
preference, not an artefact: the exact prox is what makes the term usable by the proximal
algorithms at all, and the algorithm layer scores the two choices together (see
[`match_assumption`](@ref)). Within each class the cost decides, and the table's order
breaks exact ties — so the ranking reproduces the fixed `if`-chain this replaced.

# Cost of scoring

Scoring must be negligible next to the optimization pass it selects, even a pass of a few
iterations, so it reads **only static operator metadata**: the trait predicates
(`is_eye`/`is_diagonal`/[`is_aac_diagonal`](@ref)/`is_linear`), the two size tuples, and the
*type-level* [`normal_op_fuses`](@ref). No operator is built and no array is touched. In
particular `fused_normal_op`, which answers the same question by constructing `opᴴ*op` (for
a `MatrixOp` that is the Gram matrix — `O(n²m)`, more than several solver iterations), is
called only for the candidate that actually wins.
"""
function best_formulation(op, f, disp, λ, needs::Symbol = :any)
    want_prox = needs === :prox
    n = _total_length(size(op, 2))
    m = _total_length(size(op, 1))
    diagonal = is_diagonal(op)
    linear = is_linear(op)

    best = (:none, 2, Inf)
    best = _consider(best, want_prox, :eye, is_eye(op), true, 0.0)
    best = _consider(best, want_prox, :diagonal_weight, diagonal && f isa SqrNormL2 && iszero(disp), true, 0.0)
    best = _consider(best, want_prox, :diagonal, diagonal, true, 1.0)
    # `is_aac_diagonal` is the only predicate here that is not a type-level trait for every
    # operator, so it is asked last and only when its answer can still change the winner:
    # any prox-keeping candidate already found with cost ≤ 2 beats it outright.
    best = _consider(best, want_prox, :aac_diagonal, (best[2], best[3]) > (0, 2.0) && is_aac_diagonal(op), true, 2.0)
    best = _consider(best, want_prox, :ind_affine, f isa IndPoint && _matrix_of(op) !== nothing, true, 2.5)
    best = _consider(best, want_prox, :normal_op, linear && normal_op_applicable(f, op, disp, λ), false, normal_op_cost(op, n, m))
    best = _consider(best, want_prox, :precompose, linear, false, 2.0)
    best = _consider(best, want_prox, :nonlinear, !linear, false, 2.0)

    return best[1], best[3]
end

# One step of the ranking above, written as a pure function of the incumbent so that no
# variable is captured and mutated (a closure over a mutated binding would box it and
# allocate, which is exactly what the scoring budget forbids).
#
# `best` is `(kind, prox class, cost)`; the comparison is strict, so a candidate that ties
# with the incumbent loses and the table order in `best_formulation` is the tiebreak.
@inline function _consider(best, want_prox::Bool, kind::Symbol, applicable::Bool, keeps_prox::Bool, cost::Float64)
    (applicable && (keeps_prox || !want_prox)) || return best
    class = keeps_prox ? 0 : 1
    return (class, cost) < (best[2], best[3]) ? (kind, class, cost) : best
end

"""
    merge_function_with_operator(op, f, disp, λ; needs = :any)

Build the formulation of `λ · f(op·x + disp)` that [`best_formulation`](@ref) selects.
`needs === :prox` demands one whose `prox!` is exact; passing it is how a caller states
what the selected algorithm will ask of the term.

This is the one place in the package where a function and its operator are combined — the
syntax layer builds `λ · f(A·x + d)` triples and nothing else (PLAN.md 2.6).
"""
function merge_function_with_operator(op, f, disp, λ; needs::Symbol = :any)
    kind, _ = best_formulation(op, f, disp, λ, needs)
    # A dry run only asks whether a problem parses, which the formulation cannot change, so
    # the two formulations that build something expensive (a normal operator, a
    # factorization) are left unbuilt.
    _is_dry_run() && (kind === :normal_op || kind === :ind_affine) && return f
    if kind === :normal_op
        # Scoring used the type-level fuse predicate, which is deliberately conservative but
        # can still be optimistic where inference sees a fusing product that the operator's
        # own `*` declines to build. Fall back to the generic linear formulation then.
        f_normal = with_normal_op(f, op, disp, λ)
        f_normal === nothing || return f_normal
        kind = :precompose
    end
    if kind === :eye
        f = disp == 0 ? f : PrecomposeDiagonal(f, 1.0, disp)
        if size(op, 1) != size(op, 2)
            f = ReshapeInput(f, size(op, 1))
        end
    elseif kind === :diagonal_weight
        # ½‖diag(a)·x‖² is the same function as the weighted ½∑ aᵢ²xᵢ², so a diagonal
        # operator can be folded into the weight — but only without a displacement, since
        # the weighted form has nowhere to put one.
        f = SqrNormL2(f.lambda .* diag(op) .^ 2)
    elseif kind === :diagonal
        f = PrecomposeDiagonal(f, diag(op), disp)
    elseif kind === :aac_diagonal
        f = Precompose(f, op, diag_AAc(op), disp)
    elseif kind === :ind_affine
        # `IndPoint(p)(A·x + d)` is the indicator of `{x : A·x = p - d}`, which `IndAffine`
        # solves exactly (it factorises `A` once and projects). This is the formulation
        # `==(ex, b)` used to build in the syntax layer.
        f = IndAffine(_matrix_of(op), f.p .- disp)
    elseif kind === :precompose
        # Only the gradient is ever asked of this formulation; its `prox!` is not the prox
        # of the composition, which is why `needs === :prox` rules it out.
        f = Precompose(f, op, 1, disp)
    elseif kind === :nonlinear
        if disp != 0
            op = AbstractOperators.AffineAdd(op, disp)
        end
        f = PrecomposeNonlinear(f, op)
    else
        error(
            "no formulation of this term keeps an exact prox: " *
                "$(typeof(f)) composed with $(typeof(op))"
        )
    end
    return λ == 1 ? f : Postcompose(f, λ)
end

unsatisfied_properties(term, assumptions::ProximalAlgorithms.AssumptionItem) = [property_func for property_func in assumptions.second if !property_func(term)]
does_satisfy(term, assumptions::ProximalAlgorithms.AssumptionItem) = all(property_func(term) for property_func in assumptions.second)

# Whether an assumption asks the term for a proximal operator. This is what decides the
# `needs` a formulation has to satisfy (see `best_formulation`): it is the same question the
# `keeps_exact_prox` gate asks, so the gate and the candidate filter cannot disagree.
# Assumptions without a function side (`LeastSquaresTerm`, `SquaredL2Term`) and the
# infimal-convolution ones (which recurse through `SimpleTerm`) answer `false`.
needs_prox(assumption) = hasproperty(assumption, :func) && _item_needs_prox(assumption.func)
_item_needs_prox(item::ProximalAlgorithms.AssumptionItem) = ProximalCore.is_proximable in item.second

function prepare(term::Term, assumption::ProximalAlgorithms.SimpleTerm, variables::NTuple{N, Variable}) where {N}
    needs = needs_prox(assumption) ? :prox : :any
    if does_satisfy(term, assumption.func) && (needs === :any || keeps_exact_prox(affine(term), term.f))
        op = extract_operators(variables, term)
        disp = displacement(term)
        return (assumption.func.first => merge_function_with_operator(op, term.f, disp, term.lambda; needs),)
    else
        return nothing
    end
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.SimpleTerm, ::NTuple{N, Variable}) where {N}
    repr = term.repr !== nothing ? term.repr : string(term)
    problematic_properties = unsatisfied_properties(term, assumption.func)
    return if length(problematic_properties) == 0
        println(
            "Term $repr satisfies all required properties, but absorbing the following operator ",
            "would not keep an exact prox: ", affine(term)
        )
    else
        println("Term $repr does not satisfy required property: $(join(problematic_properties, ", "))")
    end
end

# One absorbed function per variable, in `variables` order, for the case where every
# variable is mentioned by exactly one term — which is what the caller has already
# established. A variable no term mentions contributes `IndFree()`, the indicator of the
# whole space, so the `SeparableSum` still covers the full domain.
#
# The multiple-terms-per-variable case is *not* handled here: it is unreachable from the
# only caller (which enters this function only when every bag holds one term), and the
# sliced case it would have covered is handled by the `PrecomposedSlicedSeparableSum`
# branch alongside it.
function prepare_proximable_single_var_per_term(variable_bags, variables::NTuple{N, Variable}) where {N}
    fs = ()
    for var in variables
        if haskey(variable_bags, var)
            term = only(variable_bags[var])
            op = operator(term)
            disp = displacement(term)
            fs = (fs..., merge_function_with_operator(op, term.f, disp, term.lambda; needs = :prox))
        else
            fs = (fs..., IndFree())
        end
    end
    return SeparableSum(fs)
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.SimpleTerm, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        return prepare(terms[1], assumption, variables)
    end
    if any(term -> !does_satisfy(term, assumption.func), terms)
        return nothing
    end
    if needs_prox(assumption)
        if any(!keeps_exact_prox(affine(term), term.f) for term in terms)
            return nothing
        end
        variable_bags = group_by_variables(terms)
        if !can_be_separable_sum(variable_bags)
            return nothing
        end
        if all(length.(values(variable_bags)) .== 1)
            # all terms references only one variable
            return (assumption.func.first => prepare_proximable_single_var_per_term(variable_bags, variables),)
        else
            op = extract_operators(variables, terms)
            idxs = AbstractOperators.get_slicing_expr(op)
            op = remove_slicing(op)
            hcat_ops = op.A
            μs = Tuple(AbstractOperators.diag_AAc(op_i) for op_i in op.A)
            # This is the one site that wants the displacement inside the function rather
            # than in the operator: `PrecomposedSlicedSeparableSum` is handed the *linear*
            # blocks `hcat_ops` (displacement removed) and precomposes each `fᵢ` with them
            # itself, so a displacement left in the operator would simply be dropped.
            function fold_displacement(t::Term)
                disp = displacement(t)
                f = disp == 0 ? t.f : PrecomposeDiagonal(t.f, one(t.lambda), disp)
                return t.lambda == 1 ? f : Postcompose(f, t.lambda)
            end
            f = Tuple(fold_displacement(t) for t in terms)
            return (assumption.func.first => PrecomposedSlicedSeparableSum(f, idxs, hcat_ops, μs),)
        end
    else
        fs = ()
        for term in terms
            if is_linear(term)
                f = merge_function_with_operator(extract_operators(variables, term), term.f, displacement(term), term.lambda)
            else
                # Displacement is carried once by the affine operator; use the raw
                # `term.f` (no displacement-folding) and apply λ exactly once.
                op = extract_affines(variables, term)
                f = PrecomposeNonlinear(term.f, op)
                f = term.lambda == 1 ? f : Postcompose(f, term.lambda)
            end
            fs = (fs..., f)
        end
        return (assumption.func.first => ProximalOperators.Sum(fs),)
    end
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.SimpleTerm, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        print_diagnostics(terms[1], assumption, variables)
        return
    end
    # `TermSet` supports iteration and integer indexing but not `findfirst` directly,
    # so search the collected vector; its order matches `terms[i]`.
    problematic_term_index = findfirst(term -> !does_satisfy(term, assumption.func), collect(terms))
    return if problematic_term_index !== nothing
        problematic_term = terms[problematic_term_index]
        repr = problematic_term.repr !== nothing ? problematic_term.repr : string(problematic_term)
        problematic_properties = unsatisfied_properties(problematic_term, assumption.func)
        println("Term $repr does not satisfy required property: $(join(problematic_properties, ", "))")
    elseif any(term -> !keeps_exact_prox(affine(term), term.f), terms)
        println("The following terms have operators whose absorption would not keep an exact prox:")
        for term in terms
            if !keeps_exact_prox(affine(term), term.f)
                repr = term.repr !== nothing ? term.repr : string(term)
                println(" - $repr")
            end
        end
    else
        variable_bags = group_by_variables(terms)
        incompatibilities = get_unseparable_pairs(variable_bags)
        println("The following terms are incompatible with each other:")
        for (term, incompatible_terms) in incompatibilities
            println(" - $term: $(join(incompatible_terms, ", "))")
        end
    end
end

# `λ · h(L·x + d)` split as `(λ · h(· + d), L)`: the linear part of the term's operator, with the
# displacement moved into the function. An algorithm's operator slot is handed `L` alone, so
# every algorithm gets an operator it may treat as linear -- adjoint, normal operator, operator
# norm, linear combinations of `L·x` -- whatever the term's displacement.
function split_affine(variables, term::Term)
    return translated_function(term), extract_operators(variables, term)
end
function split_affine(variables, terms::TermSet)
    return SeparableSum(translated_function.(terms)...), extract_operators(variables, terms)
end
function translated_function(t::Term)
    d = displacement(t)
    f = weighted_function(t)
    return iszero(d) ? f : Translate(f, d)
end

function prepare(term::Term, assumption::ProximalAlgorithms.OperatorTerm, variables::NTuple{N, Variable}) where {N}
    f, op = split_affine(variables, term)
    if does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func)
        return (
            assumption.func.first => f,
            assumption.operator.first => op,
        )
    else # try preparing as a simple term
        tup = prepare(term, ProximalAlgorithms.SimpleTerm(assumption.func), variables)
        if tup !== nothing && length(variables) > 1
            example_input = ArrayPartition(Tuple(~var for var in variables))
            tup = (tup..., assumption.operator.first => AbstractOperators.Eye(example_input))
        end
        return tup
    end
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.OperatorTerm, variables::NTuple{N, Variable}) where {N}
    # The decomposition shown is the one `prepare` would hand over (see `split_affine`).
    f, op = split_affine(variables, term)
    repr = term.repr !== nothing ? term.repr : string(term)
    if is_eye(op)
        problematic_properties = unsatisfied_properties(term.f, assumption.func)
        println("Term $repr does not satisfy required properties: $(join(problematic_properties, ", "))")
    else
        println("A possible decomposition of term $repr:")
        print(" - ", assumption.func.first, " = ", f)
        if !does_satisfy(f, assumption.func)
            problematic_properties = unsatisfied_properties(f, assumption.func)
            println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        else
            println()
        end
        print(" - ", assumption.operator.first, " = ", op)
        if !does_satisfy(op, assumption.operator)
            problematic_properties = unsatisfied_properties(op, assumption.operator)
            println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        else
            println()
        end
    end
    println("When trying to prepare the term as a simple term:")
    return print_diagnostics(term, ProximalAlgorithms.SimpleTerm(assumption.func), variables)
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.OperatorTerm, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        return prepare(terms[1], assumption, variables)
    end
    f, op = split_affine(variables, terms)
    if does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func)
        return (
            assumption.func.first => f,
            assumption.operator.first => op,
        )
    else # try preparing as a simple term
        return prepare(terms, ProximalAlgorithms.SimpleTerm(assumption.func), variables)
    end
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.OperatorTerm, variables::NTuple{N, Variable}) where {N}
    # The decomposition shown is the one `prepare` would hand over (see `split_affine`).
    f, op = split_affine(variables, terms)
    repr = string(terms)
    if is_eye(op)
        for term in terms
            problematic_properties = unsatisfied_properties(term.f, assumption.func)
            println("Term $repr does not satisfy required properties: $(join(problematic_properties, ", "))")
        end
    else
        println("A possible decomposition of terms $repr:")
        print(" - ", assumption.func.first, " = ", f)
        if !does_satisfy(f, assumption.func)
            problematic_properties = unsatisfied_properties(f, assumption.func)
            println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        else
            println()
        end
        print(" - ", assumption.operator.first, " = ", op)
        if !does_satisfy(op, assumption.operator)
            problematic_properties = unsatisfied_properties(op, assumption.operator)
            println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        else
            println()
        end
    end
    println("When trying to prepare terms as a simple function:")
    return print_diagnostics(terms, ProximalAlgorithms.SimpleTerm(assumption.func), variables)
end

function prepare(term::Term, assumption::ProximalAlgorithms.OperatorTermWithInfimalConvolution, variables::NTuple{N, Variable}) where {N}
    f, op = split_affine(variables, term)
    if does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func₁)
        return (
            assumption.func₁.first => f,
            assumption.operator.first => op,
        )
    elseif does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func₂)
        return (
            assumption.func₂.first => f,
            assumption.operator.first => op,
        )
    else
        # try preparing as a simple term
        tup = prepare(term, ProximalAlgorithms.SimpleTerm(assumption.func₁), variables)
        if tup !== nothing && length(variables) > 1
            example_input = ArrayPartition(tuple([~var for var in variables]...))
            tup = (tup..., assumption.operator.first => AbstractOperators.Eye(example_input))
        end
        return tup
    end
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.OperatorTermWithInfimalConvolution, variables::NTuple{N, Variable}) where {N}
    f, op = split_affine(variables, term)
    repr = term.repr !== nothing ? term.repr : string(term)
    if is_eye(op)
        problematic_properties = unsatisfied_properties(term.f, assumption.func₁)
        println("Term $repr does not satisfy required properties: $(join(problematic_properties, ", "))")
    else
        println("A possible decomposition of term $repr:")
        print(" - ", assumption.func₁.first, " = ", f)
        if !does_satisfy(f, assumption.func₁)
            problematic_properties = unsatisfied_properties(f, assumption.func₁)
            println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        else
            println()
        end
        print(" - ", assumption.operator.first, " = ", op)
        if !does_satisfy(op, assumption.operator)
            problematic_properties = unsatisfied_properties(op, assumption.operator)
            println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        else
            println()
        end
    end
    println("When trying to prepare the term as a simple term:")
    return print_diagnostics(term, ProximalAlgorithms.SimpleTerm(assumption.func₁), variables)
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.OperatorTermWithInfimalConvolution, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        return prepare(terms[1], assumption, variables)
    end
    f, op = split_affine(variables, terms)
    if does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func₁)
        return (
            assumption.func₁.first => f,
            assumption.operator.first => op,
        )
    elseif does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func₂)
        return (
            assumption.func₂.first => f,
            assumption.operator.first => op,
        )
    else
        # try preparing as a simple term
        tup = prepare(terms, ProximalAlgorithms.SimpleTerm(assumption.func₁), variables)
        if tup === nothing
            tup = prepare(terms, ProximalAlgorithms.SimpleTerm(assumption.func₂), variables)
        end
        if tup !== nothing && length(variables) > 1
            example_input = ArrayPartition(tuple([~var for var in variables]...))
            tup = (tup..., assumption.operator.first => AbstractOperators.Eye(example_input))
        end
        return tup
    end
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.OperatorTermWithInfimalConvolution, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        print_diagnostics(terms[1], assumption, variables)
        return
    end
    f, op = split_affine(variables, terms)
    repr = string(terms)
    if is_eye(op)
        for term in terms
            problematic_properties = unsatisfied_properties(term.f, assumption.func₁)
            println("Term $repr does not satisfy required properties: $(join(problematic_properties, ", "))")
        end
    else
        println("A possible decomposition of terms $repr:")
        print(" - ", assumption.func₁.first, " = ", f)
        if !does_satisfy(f, assumption.func₁)
            problematic_properties = unsatisfied_properties(f, assumption.func₁)
            println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        else
            println()
        end
        print(" - ", assumption.operator.first, " = ", op)
        if !does_satisfy(op, assumption.operator)
            problematic_properties = unsatisfied_properties(op, assumption.operator)
            println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        else
            println()
            println("Alteratively, one can try to prepare the function part as:")
            print(" - ", assumption.func₂.first, " = ", f)
            if !does_satisfy(f, assumption.func₂)
                problematic_properties = unsatisfied_properties(f, assumption.func₂)
                println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
            else
                println()
            end
        end
    end
    println("When trying to prepare the term as a simple term:")
    return print_diagnostics(terms, ProximalAlgorithms.SimpleTerm(assumption.func₁), variables)
end

function prepare(term::Term, assumption::ProximalAlgorithms.LeastSquaresTerm, variables::NTuple{N, Variable}) where {N}
    f = term.f
    # The CG-family objective is ‖A x - b‖² but StructuredOptimization stores the
    # displacement `d` of `A x + d`, so the least-squares target is b = -d.
    aha = nothing
    if f isa SqrNormL2WithNormalOp
        lambda = term.lambda * f.lambda
        op = f.A
        b = -displacement(op)
        op = remove_displacement(op)
        # `f` already built AᴴA in its constructor; an algorithm that asks for it
        # (`assumption.AHA`) gets this one instead of forming an identical second copy.
        aha = remove_displacement(f.AᴴA)
    elseif f isa ProximalOperators.SqrNormL2
        # Fold the function's own weight f.lambda in as well (it was ignored before).
        lambda = term.lambda * f.lambda
        op = extract_operators(variables, term)
        b = -displacement(term)
    else
        # ProximalOperators.LeastSquares carries its own embedded operator and vector
        # that this path does not read; reject rather than silently mis-scale it.
        return nothing
    end
    # Only scalar weights can be folded into the operator; array weights would need a
    # diagonal reweighting the CG-family objective does not model here.
    if lambda isa AbstractArray
        return nothing
    end
    if !does_satisfy(op, assumption.operator)
        return nothing
    end
    # CG-family objective is ‖A x - b‖² + λ_reg‖x‖², where SquaredL2Term maps the
    # regularizer to λ_reg = term.lambda*f.lambda (no ½). To keep the data term at the
    # correct *relative* weight, scale the residual by √λ, not by λ.
    c = sqrt(lambda)
    # `AᴴA` is only worth having when the algorithm asks for it and the operator is not about
    # to be rescaled — `(cA)ᴴ(cA) ≠ AᴴA`, so a cached one would no longer match — and this is
    # not a dry run. The tests are free, and they come first so that the `SqrNormL2` branch
    # below never pays to build a normal operator it would immediately discard.
    want_aha = assumption.AHA !== nothing && c == 1 && !_is_dry_run()
    if want_aha && f isa ProximalOperators.SqrNormL2
        # Absorption to the normal-op formulation happens at `merge_function_with_operator`
        # time, not when `ls` builds the term, so `f` is always the plain `SqrNormL2` here.
        # Ask the formulation layer's question at the metadata level, then build only the one
        # operator wanted: going through `merge_function_with_operator` would construct a whole
        # `SqrNormL2WithNormalOp` and tilt its normal operator by the displacement, only for
        # this to strip the tilt straight back off.
        if best_formulation(op, f, displacement(term), term.lambda)[1] === :normal_op
            aha = fused_normal_op(op)
        end
    end
    if c != 1
        op = c * op
        b = c * b
    end
    result = (assumption.operator.first => op, assumption.b => b)
    return if want_aha && aha !== nothing
        (result..., assumption.AHA => aha)
    else
        result
    end
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.LeastSquaresTerm, variables::NTuple{N, Variable}) where {N}
    op = extract_operators(variables, term)
    b = -displacement(term)
    f = term.f
    repr = term.repr !== nothing ? term.repr : string(term)
    return if !(f isa ProximalOperators.LeastSquares || f isa ProximalOperators.SqrNormL2)
        println("Term $repr does not satisfy required property: it is not a least squares function")
    else
        println("A possible decomposition of term $repr:")
        print(" - ", assumption.operator.first, " = ", op)
        problematic_properties = unsatisfied_properties(op, assumption.operator)
        println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        print(" - ", assumption.b, " = ", b)
    end
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.LeastSquaresTerm, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        return prepare(terms[1], assumption, variables)
    end
    return nothing
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.LeastSquaresTerm, variables::NTuple{N, Variable}) where {N}
    return if length(terms) == 1
        print_diagnostics(terms[1], assumption, variables)
    else
        println("Cannot prepare terms $terms as a least squares term: only a single term can be prepared as such.")
    end
end

function prepare(term::Term, assumption::ProximalAlgorithms.SquaredL2Term, variables::NTuple{N, Variable}) where {N}
    f = term.f
    if displacement(term) != 0 || !(f isa ProximalOperators.SqrNormL2)
        return nothing
    end
    λ = term.lambda * f.lambda
    op = extract_affines(variables, term)
    if is_eye(op)
        return (assumption.λ => λ,)
    elseif is_diagonal(op)
        return (assumption.λ => λ * diag(op),)
    else
        return nothing
    end
end

function print_diagnostics(term::Term, ::ProximalAlgorithms.SquaredL2Term, variables::NTuple{N, Variable}) where {N}
    repr = term.repr !== nothing ? term.repr : string(term)
    return if displacement(term) != 0
        println("Term $repr does not satisfy required property: it has non-zero displacement")
    elseif !(term.f isa ProximalOperators.SqrNormL2)
        println("Term $repr does not satisfy required property: it is not a squared L2 function")
    else
        println("Term $repr does not satisfy required property: the operator is not an identity or diagonal")
    end
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.SquaredL2Term, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        return prepare(terms[1], assumption, variables)
    end
    return nothing
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.SquaredL2Term, variables::NTuple{N, Variable}) where {N}
    return if length(terms) == 1
        print_diagnostics(terms[1], assumption, variables)
    else
        println("Cannot prepare terms $terms as a squared L2 term: only a single term can be prepared as such.")
    end
end

function prepare(term::Term, assumption::ProximalAlgorithms.RepeatedSimpleTerm, variables::NTuple{N, Variable}) where {N}
    simple_assumption = ProximalAlgorithms.SimpleTerm(assumption.func)
    return prepare(term, simple_assumption, variables)
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.RepeatedSimpleTerm, variables::NTuple{N, Variable}) where {N}
    simple_assumption = ProximalAlgorithms.SimpleTerm(assumption.func)
    return print_diagnostics(term, simple_assumption, variables)
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.RepeatedSimpleTerm, variables::NTuple{N, Variable}) where {N}
    simple_assumption = ProximalAlgorithms.SimpleTerm(assumption.func)
    results = ()
    for term in terms
        result = prepare(term, simple_assumption, variables)
        if isnothing(result)
            return nothing
        end
        results = (results..., result[1].second)
    end
    return (assumption.func.first => results,)
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.RepeatedSimpleTerm, variables::NTuple{N, Variable}) where {N}
    simple_assumption = ProximalAlgorithms.SimpleTerm(assumption.func)
    for term in terms
        if prepare(term, simple_assumption, variables) === nothing
            print_diagnostics(term, simple_assumption, variables)
        end
    end
    return
end

function prepare(term::Term, assumption::ProximalAlgorithms.RepeatedOperatorTerm, variables::NTuple{N, Variable}) where {N}
    operator_term_assumption = ProximalAlgorithms.OperatorTerm(assumption.func, assumption.operator)
    return prepare(term, operator_term_assumption, variables)
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.RepeatedOperatorTerm, variables::NTuple{N, Variable}) where {N}
    operator_term_assumption = ProximalAlgorithms.OperatorTerm(assumption.func, assumption.operator)
    return print_diagnostics(term, operator_term_assumption, variables)
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.RepeatedOperatorTerm, variables::NTuple{N, Variable}) where {N}
    operator_term_assumption = ProximalAlgorithms.OperatorTerm(assumption.func, assumption.operator)
    function_results = ()
    operator_results = ()
    for term in terms
        result = prepare(term, operator_term_assumption, variables)
        if isnothing(result)
            return nothing
        end
        function_results = (function_results..., result[1].second)
        operator_results = (operator_results..., result[2].second)
    end
    return (
        assumption.func.first => function_results,
        assumption.operator.first => operator_results,
    )
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.RepeatedOperatorTerm, variables::NTuple{N, Variable}) where {N}
    operator_term_assumption = ProximalAlgorithms.OperatorTerm(assumption.func, assumption.operator)
    for term in terms
        if prepare(term, operator_term_assumption, variables) === nothing
            print_diagnostics(term, operator_term_assumption, variables)
        end
    end
    return
end
