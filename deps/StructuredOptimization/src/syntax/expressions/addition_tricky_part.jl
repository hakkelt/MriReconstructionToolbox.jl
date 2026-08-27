using Base.Iterators: flatten
abstract type OpStructure end

struct HCatStructure{N} <: OpStructure
    op::AbstractOperators.AbstractOperator
    structure::NTuple{N,Any}
end

struct SumStructure{N} <: OpStructure
    op::AbstractOperators.AbstractOperator
    structure::NTuple{N,Any}
end

function get_structure(op::AbstractOperators.HCAT, vars)
    if length(op.A) == AbstractOperators.ndoms(op, 2) # this is the deepest or only HCAT operator
        return HCatStructure(op, vars)
    else # there are more nested HCAT operators, let's recurse!
        result = ()
        var_group_counter = 1
        for suboperator in op.A
            subvars = vars[var_group_counter:var_group_counter+AbstractOperators.ndoms(suboperator, 2)-1]
            if AbstractOperators.ndoms(suboperator, 2) == 1
                returned = subvars
            else
                returned = get_structure(suboperator, subvars)
                @assert returned !== nothing
            end
            if returned isa Tuple
                result = (result..., returned...)
            else
                result = (result..., returned)
            end
            var_group_counter += AbstractOperators.ndoms(suboperator, 2)
        end
        return HCatStructure(op, result)
    end
end

function get_structure(op::AbstractOperators.Sum, vars)
    return SumStructure(op, tuple((get_structure(suboperator, vars) for suboperator in op.A)...))
end

function get_structure(op, vars)
    if op isa AbstractOperators.AbstractOperator && AbstractOperators.ndoms(op, 2) == 1
        return SumStructure(op, vars)
    else
        for k in 1:fieldcount(typeof(op))
            value = getfield(op, k)
            if value isa AbstractOperators.AbstractOperator
                return get_structure(value, vars)
            elseif value isa Tuple
                for v in value
                    return get_structure(v, vars)
                end
            end
        end
        @assert false "This should never happen"
    end
end

function deep_flatten(structure::HCatStructure)
    result = ()
    for item in structure.structure
        if isa(item, OpStructure)
            sub_flattened = deep_flatten(item)
            if sub_flattened === nothing
                return nothing
            end
            result = tuple(result..., sub_flattened...)
        else
            result = tuple(result..., item)
        end
    end
    return result
end

function deep_flatten(structure::SumStructure)
    nested_structures = tuple((deep_flatten(item) for item in structure.structure)...)
    if all(==(nested_structures[1]), nested_structures)
        return nested_structures[1]
    else
        return nothing
    end
end

struct UnregularIndex{N}
    max::NTuple{N, Int}
    UnregularIndex(max) = any(max .< 1) ? error("max must be >= 1") : new{length(max)}(tuple(max...))
end

Base.first(iter::UnregularIndex) = tuple(fill(1, length(iter.max))...)
Base.length(iter::UnregularIndex) = sum(iter.max)

function Base.iterate(iter::UnregularIndex)
    state = first(iter)
    return state, state
end

function Base.iterate(iter::UnregularIndex{N}, state::NTuple{N, Int}) where {N}
    if state == iter.max
        return nothing
    end
    currentdim = findfirst(i -> state[i] != iter.max[i], 1:N)
    nextstate = tuple((j < currentdim ? 1 : (j == currentdim ? state[j]+1 : state[j]) for j in 1:N)...)
    return nextstate, nextstate
end

get_structure_only(str) = str isa OpStructure ? tuple((get_structure_only(item) for item in str.structure)...) : str

Base.length(str::OpStructure) = length(str.structure)
Base.getindex(str::OpStructure, i) = str.structure[i]

permute_structure(str, perm) = tuple((str[i][perm[i]] for i in eachindex(str))...)

function compute_permutations(st)
    result = ()
    for perm in UnregularIndex(length.(st))
        result = (result..., permute_structure(st, perm))
    end
    return result
end

function get_all_permutations(structure::SumStructure)
    product = [get_all_permutations(item) for item in structure.structure]
    return tuple((SumStructure(structure.op, st) for st in compute_permutations(product))...)
end

function get_all_permutations(structure::HCatStructure)
    nested_perms = [isa(item, Int) ? (item,) : get_all_permutations(item) for item in structure.structure]
    product = compute_permutations(nested_perms)
    combinations = flatten(permutations(p) for p in product)
    return tuple((HCatStructure(structure.op, tuple(p...)) for p in combinations)...)
end

function find_feasible_permutation(vars, stA, stB)
    stA_perms = get_all_permutations(stA)
    stB_perms = get_all_permutations(stB)
    stA_pairs = filter(pair -> pair[2] !== nothing, [(s, deep_flatten(s)) for s in stA_perms])
    stB_pairs = filter(pair -> pair[2] !== nothing, [(s, deep_flatten(s)) for s in stB_perms])
    for vars_perm in permutations(vars)
        vars_perm = tuple(vars_perm...)
        stA_perm = findfirst(pair -> pair[2] == vars_perm, stA_pairs)
        if stA_perm === nothing
            continue
        end
        stB_perm = findfirst(pair -> pair[2] == vars_perm, stB_pairs)
        if stB_perm === nothing
            continue
        end
        return vars_perm
    end
    return nothing
end

function add_missing_vars(old_vars, op, vars)
    missing_vars = setdiff(vars, old_vars)
    if isempty(missing_vars)
        return old_vars, op
    end
    dummy_ops = [AbstractOperators.Zeros(eltype(~var), size(~var), AbstractOperators.codomain_type(op), size(op, 1)) for var in missing_vars]
    new_vars = (old_vars..., missing_vars...)
    new_op = AbstractOperators.HCAT(op, dummy_ops...)
    return new_vars, new_op
end

function Usum_op(
	xA::NTuple{N,Variable}, xB::NTuple{M,Variable}, A::AbstractOperator, B::AbstractOperator, sign::Bool
) where {N,M}
    xNew  = tuple(unique((xA...,xB...))...)
    xA, A = add_missing_vars(xA, A, xNew)
    xB, B = add_missing_vars(xB, B, xNew)
    vars_index = tuple((i for i in eachindex(xNew))...)
    xA_index = tuple((findfirst(==(x), xNew) for x in xA)...)
    xB_index = tuple((findfirst(==(x), xNew) for x in xB)...)
    structureA = get_structure(A, xA_index)
    structureB = get_structure(B, xB_index)
    var_perm = find_feasible_permutation(vars_index, structureA, structureB)
    if var_perm === nothing
        error("No feasible permutation found")
    end
    if var_perm != xA_index
        A = AbstractOperators.permute(A, invperm([xA_index...]))
    end
    if var_perm != xB_index
        B = AbstractOperators.permute(B, invperm([xB_index...]))
    end
    opNew = sign ? A+B : A-B
	return xNew, opNew
end
