is_proximable(term::Term) = is_proximable(typeof(term.f)) && keeps_exact_prox(affine(term), term.f)

function get_operators_for_var(term, var)
    full_operator = affine(term)
    if AbstractOperators.ndoms(full_operator, 2) == 1
        return full_operator
    else
        return full_operator[findfirst(==(var), variables(term))]
    end
end

"""
    splits_per_variable(term)

Whether `term` can be handed to a per-variable treatment at all.

A single-variable term trivially can. A multi-variable one can only if its function is
itself separable — otherwise the slicing structure of its operators says nothing about
whether the sum decomposes, because the function couples the blocks regardless. Shared by
[`is_separable_sum`](@ref), `can_be_separable_sum` and `get_unseparable_pairs`, which must
agree on this rule: the first two are predicates and the third names the offending pairs.
"""
splits_per_variable(term) = length(variables(term)) == 1 || is_separable(term.f)

function is_separable_sum(terms::TermSet)
    # Construct the set of occurring variables
    vars = Set()
    for term in terms
        union!(vars, variables(term))
    end
    # Check that each variable occurs in only one term
    for var in vars
        terms_with_var = [t for t in terms if var in variables(t)]
        if length(terms_with_var) != 1
            # All terms must be either separable or have a single variable
            if !all(splits_per_variable, terms_with_var)
                return false
            end
            # All terms must be sliced for this variable
            operators = [get_operators_for_var(term, var) for term in terms_with_var]
            if !all(is_sliced(op) for op in operators)
                return false
            end
            # The sliced operators must not overlap
            slicing_masks = [AbstractOperators.get_slicing_mask(op) for op in operators]
            for i in eachindex(operators), j in (i + 1):length(operators)
                if any(slicing_masks[i] .&& slicing_masks[j])
                    return false
                end
            end
        end
    end
    return true
end

function is_proximable(terms::TermSet)
    return all(is_proximable.(terms)) && is_separable_sum(terms)
end
