function add_to_incompatibilities(incompatibilities, t1, t2)
    if haskey(incompatibilities, t1)
        push!(incompatibilities[t1], t2)
    else
        incompatibilities[t1] = Set([t2])
    end
    if haskey(incompatibilities, t2)
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
            # Check if any of the terms are sliced
            operators = [get_operators_for_var(term, var) for term in term_list]
            slicing_masks = [is_sliced(op) ? AbstractOperators.get_slicing_mask(op) : nothing for op in operators]
            for i in eachindex(operators)
                if is_sliced(operators[i])
                    # This operator is sliced, check if it is overlapping with any other sliced operator
                    for j in i+1:length(operators)
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
            # Check if any of the terms are sliced
            operators = [get_operators_for_var(term, var) for term in term_list]
            slicing_masks = [is_sliced(op) ? AbstractOperators.get_slicing_mask(op) : nothing for op in operators]
            for i in eachindex(operators)
                if is_sliced(operators[i])
                    # This operator is sliced, check if it is overlapping with any other sliced operator
                    for j in i+1:length(operators)
                        if is_sliced(operators[j]) && any(slicing_masks[i] .&& slicing_masks[j])
                            add_to_incompatibilities(incompatibilities, term_list[i], term_list[j])
                        end
                    end
                else # no slicing -> this term is incompatible with all others
                    for j in i+1:length(operators)
                        add_to_incompatibilities(incompatibilities, term_list[i], term_list[j])
                    end
                end
            end
        end
    end
    return incompatibilities
end

function merge_function_with_operator(op, f, disp, λ)
    if is_eye(op)
        f = disp == 0 ? f : PrecomposeDiagonal(f, 1.0, disp)
        if size(op, 1) != size(op, 2)
            f = ReshapeInput(f, size(op, 1))
        end
    elseif is_diagonal(op)
        if f isa SqrNormL2
            f = SqrNormL2(f.lambda .* diag(op) .^ 2)
        else
            f = PrecomposeDiagonal(f, diag(op), disp)
        end
    elseif is_AAc_diagonal(op)
        f = Precompose(f, op, diag_AAc(op), disp)
    elseif is_linear(op)
        # we assume that prox will not be called on this term because it will not give a valid result
        f = Precompose(f, op, 1, disp)
    else
        # we assume that prox will not be called on this term because it will not give a valid result
        if disp != 0
            op = AbstractOperators.AffineAdd(op, disp)
        end
        f = PrecomposeNonlinear(f, op)
    end
    return λ == 1 ? f : Postcompose(f, λ)
end

unsatisfied_properties(term, assumptions::ProximalAlgorithms.AssumptionItem) = [property_func for property_func in assumptions.second if !property_func(term)]
does_satisfy(term, assumptions::ProximalAlgorithms.AssumptionItem) = all(property_func(term) for property_func in assumptions.second)

function prepare(term::Term, assumption::ProximalAlgorithms.SimpleTerm, variables::NTuple{N, Variable}) where N
    if does_satisfy(term, assumption.func) && (!(ProximalCore.is_proximable in assumption.func.second) || is_AAc_diagonal(term.A.L))
        op = extract_operators(variables, term)
        disp = displacement(term)
        return (assumption.func.first => merge_function_with_operator(op, term.f, disp, term.lambda),)
    else
        return nothing
    end
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.SimpleTerm, ::NTuple{N, Variable}) where N
    repr = term.repr !== nothing ? term.repr : string(term)
    problematic_properties = unsatisfied_properties(term, assumption.func)
    if length(problematic_properties) == 0
        println("Term $repr satisfies all required properties, but the following operator is not AAc diagonal: ", term.A.L)
    else
        println("Term $repr does not satisfy required property: $(join(problematic_properties, ", "))")
    end
end

function prepare_proximable_single_var_per_term(variable_bags, variables::NTuple{N, Variable}) where {N}
    fs = ()
    for var in variables
        if haskey(variable_bags, var)
            term_list = variable_bags[var]
            if length(term_list) > 1
                #multiple terms per variable
                #currently this happens only with GetIndex
                fxi,idxs = (),()
                for ti in term_list
                    op = operator(ti)
                    fxi  = (fxi..., merge_function_with_operator(op, ti.f, displacement(ti), ti.lambda))
                    if AbstractOperators.ndoms(op, 2) > 1
                        op = op[findfirst(==(var), variables(ti))]
                    end
                    if typeof(op) <: Compose
                        idx = op.A[1].idx
                    else
                        idx = op.idx
                    end
                    idxs = (idxs..., AbstractOperators.get_slicing_mask(op))
                end
                fs = (fs..., SlicedSeparableSum(fxi,idxs))
            else
                op = operator(term_list[1])
                disp = displacement(term_list[1])
                fs = (fs..., merge_function_with_operator(op, term_list[1].f, disp, term_list[1].lambda))
            end
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
    if ProximalCore.is_proximable in assumption.func.second
        if any(!is_AAc_diagonal(affine(term)) for term in terms)
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
            f = extract_functions(terms)
            return (assumption.func.first => PrecomposedSlicedSeparableSum(f.fs, idxs, hcat_ops, μs),)
        end
    else
        fs = ()
        for term in terms
            if is_linear(term)
                f = merge_function_with_operator(extract_operators(variables, term), term.f, displacement(term), term.lambda)
            else
                f = extract_functions(term)
                op = extract_affines(variables, term)
                f = PrecomposeNonlinear(f, op)
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
    problematic_term_index = findfirst(term -> !does_satisfy(term, assumption.func), terms)
    if problematic_term_index !== nothing
        problematic_term = terms[problematic_term_index]
        repr = problematic_term.repr !== nothing ? problematic_term.repr : string(problematic_term)
        problematic_properties = unsatisfied_properties(problematic_term, assumption.func)
        println("Term $repr does not satisfy required property: $(join(problematic_properties, ", "))")
    elseif any(term -> !is_AAc_diagonal(affine(term)), terms)
        println("The following terms contains operators that are not AAc diagonal:")
        for term in terms
            if !is_AAc_diagonal(affine(term))
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

function prepare(term::Term, assumption::ProximalAlgorithms.OperatorTerm, variables::NTuple{N, Variable}) where N
    op = extract_affines(variables, term)
    if does_satisfy(op, assumption.operator) && does_satisfy(term.f, assumption.func)
        return (
            assumption.func.first => term.lambda == 1 ? term.f : Postcompose(term.f, term.lambda),
            assumption.operator.first => op
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

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.OperatorTerm, variables::NTuple{N, Variable}) where N
    op = affine(term)
    repr = term.repr !== nothing ? term.repr : string(term)
    if is_eye(op)
        problematic_properties = unsatisfied_properties(term.f, assumption.func)
        println("Term $repr does not satisfy required properties: $(join(problematic_properties, ", "))")
    else
        println("A possible decomposition of term $repr:")
        f = term.lambda == 1 ? term.f : Postcompose(term.f, term.lambda)
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
    print_diagnostics(term, ProximalAlgorithms.SimpleTerm(assumption.func), variables)
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.OperatorTerm, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        return prepare(terms[1], assumption, variables)
    end
    op = extract_affines(variables, terms)
    f = extract_functions(terms)
    if does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func)
        return (
            assumption.func.first => f,
            assumption.operator.first => op
        )
    else # try preparing as a simple term
        return prepare(terms, ProximalAlgorithms.SimpleTerm(assumption.func), variables)
    end
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.OperatorTerm, variables::NTuple{N, Variable}) where {N}
    op = extract_affines(variables, terms)
    f = extract_functions(terms)
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
    print_diagnostics(terms, ProximalAlgorithms.SimpleTerm(assumption.func), variables)
end

function prepare(term::Term, assumption::ProximalAlgorithms.OperatorTermWithInfimalConvolution, variables::NTuple{N, Variable}) where {N}
    op = extract_affines(variables, term)
    f = extract_functions(term)
    if does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func₁)
        return (
            assumption.func₁.first => f,
            assumption.operator.first => op
        )
    elseif does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func₂)
        return (
            assumption.func₂.first => f,
            assumption.operator.first => affine(term)
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
    op = affine(term)
    f = extract_functions(term)
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
    print_diagnostics(term, ProximalAlgorithms.SimpleTerm(assumption.func₁), variables)
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.OperatorTermWithInfimalConvolution, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        return prepare(terms[1], assumption, variables)
    end
    op = extract_affines(variables, terms)
    f = extract_functions(terms)
    if does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func₁)
        return (
            assumption.func₁.first => f,
            assumption.operator.first => op
        )
    elseif does_satisfy(op, assumption.operator) && does_satisfy(f, assumption.func₂)
        return (
            assumption.func₂.first => f,
            assumption.operator.first => affine(terms[1].A)
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
    op = affine(terms[1].A)
    f = extract_functions(terms)
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
    print_diagnostics(terms, ProximalAlgorithms.SimpleTerm(assumption.func₁), variables)
end

function prepare(term::Term, assumption::ProximalAlgorithms.LeastSquaresTerm, variables::NTuple{N, Variable}) where N
    f = term.f
    f_is_ls = f isa ProximalOperators.LeastSquares || f isa ProximalOperators.SqrNormL2 || f isa SqrNormL2WithNormalOp
    if !f_is_ls
        return nothing
    end
    if f isa SqrNormL2WithNormalOp
        lambda = term.lambda * f.lambda
        op = term.f.A
        b = displacement(op)
        op = remove_displacement(op)
    else
        lambda = term.lambda
        op = extract_operators(variables, term)
        b = displacement(term)
    end
    if !does_satisfy(op, assumption.operator)
        return nothing
    end
    if lambda != 1
        op = lambda * op
        b = lambda * b
    end
    return (
        assumption.operator.first => op,
        assumption.b => b,
    )
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.LeastSquaresTerm, variables::NTuple{N, Variable}) where N
    op = extract_operators(variables, term)
    b = displacement(term)
    f = term.f
    repr = term.repr !== nothing ? term.repr : string(term)
    if !(f isa ProximalOperators.LeastSquares || f isa ProximalOperators.SqrNormL2)
        println("Term $repr does not satisfy required property: it is not a least squares function")
    else
        println("A possible decomposition of term $repr:")
        print(" - ", assumption.operator.first, " = ", op)
        problematic_properties = unsatisfied_properties(op, assumption.operator)
        println(" -> $(join(problematic_properties, ", ")) $(length(problematic_properties) == 1 ? "property is" : "properties are") not satisfied")
        print(" - ", assumption.b.first, " = ", b)
    end
end

function prepare(terms::TermSet, assumption::ProximalAlgorithms.LeastSquaresTerm, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        return prepare(terms[1], assumption, variables)
    end
    return nothing
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.LeastSquaresTerm, variables::NTuple{N, Variable}) where {N}
    if length(terms) == 1
        print_diagnostics(terms[1], assumption, variables)
    else
        println("Cannot prepare terms $terms as a least squares term: only a single term can be prepared as such.")
    end
end

function prepare(term::Term, assumption::ProximalAlgorithms.SquaredL2Term, variables::NTuple{N, Variable}) where N
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

function print_diagnostics(term::Term, ::ProximalAlgorithms.SquaredL2Term, variables::NTuple{N, Variable}) where N
    repr = term.repr !== nothing ? term.repr : string(term)
    if displacement(term) != 0
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
    if length(terms) == 1
        print_diagnostics(terms[1], assumption, variables)
    else
        println("Cannot prepare terms $terms as a squared L2 term: only a single term can be prepared as such.")
    end
end

function prepare(term::Term, assumption::ProximalAlgorithms.RepeatedSimpleTerm, variables::NTuple{N, Variable}) where N
    simple_assumption = ProximalAlgorithms.SimpleTerm(assumption.func)
    return prepare(term, simple_assumption, variables)
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.RepeatedSimpleTerm, variables::NTuple{N, Variable}) where N
    simple_assumption = ProximalAlgorithms.SimpleTerm(assumption.func)
    print_diagnostics(term, simple_assumption, variables)
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
end

function prepare(term::Term, assumption::ProximalAlgorithms.RepeatedOperatorTerm, variables::NTuple{N, Variable}) where N
    operator_term_assumption = ProximalAlgorithms.OperatorTerm(assumption.func, assumption.operator)
    return prepare(term, operator_term_assumption, variables)
end

function print_diagnostics(term::Term, assumption::ProximalAlgorithms.RepeatedOperatorTerm, variables::NTuple{N, Variable}) where N
    operator_term_assumption = ProximalAlgorithms.OperatorTerm(assumption.func, assumption.operator)
    print_diagnostics(term, operator_term_assumption, variables)
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
        assumption.operator.first => operator_results
    )
end

function print_diagnostics(terms::TermSet, assumption::ProximalAlgorithms.RepeatedOperatorTerm, variables::NTuple{N, Variable}) where {N}
    operator_term_assumption = ProximalAlgorithms.OperatorTerm(assumption.func, assumption.operator)
    for term in terms
        if prepare(term, operator_term_assumption, variables) === nothing
            print_diagnostics(term, operator_term_assumption, variables)
        end
    end
end
