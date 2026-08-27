is_proximable(term::Term) = is_proximable(typeof(term.f)) && is_AAc_diagonal(term.A.L)

function get_operators_for_var(term, var)
    full_operator = affine(term)
    if AbstractOperators.ndoms(full_operator, 2) == 1
        return full_operator
    else
        return full_operator[findfirst(==(var), variables(term))]
    end
end

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
			# All terms must be either  or have a single variable
			if ! all( length(variables(term)) == 1 || is_separable(term.f) for term in terms_with_var )
				return false
			end
            # All terms must be sliced for this variable
            operators = [get_operators_for_var(term, var) for term in terms_with_var]
			if !all(is_sliced(op) for op in operators)
				return false
			end
			# The sliced operators must not overlap
            slicing_masks = [AbstractOperators.get_slicing_mask(op) for op in operators]
            for i in eachindex(operators), j in i+1:length(operators)
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
