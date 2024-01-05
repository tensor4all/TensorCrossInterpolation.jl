"""
This file contains functions for evaluating a function on a batch of indices mainly for TensorCI2.
If the function supports batch evaluation, then it should implement the `BatchEvaluator` interface.
Otherwise, the function is evaluated on each index individually using the usual function call syntax and loops.
"""
function _batchevaluate_dispatch(
    ::Type{V},
    f,
    localdims::Vector{Int},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M})::Array{V,M + 2} where {V,M}

    if length(leftindexset) * length(rightindexset) == 0
        return Array{V,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    nl = length(first(leftindexset))
    nr = length(first(rightindexset))
    L = M + nl + nr

    indexset = MultiIndex(undef, L)
    result = Array{V,3}(undef, length(leftindexset), prod(localdims[nl+1:L-nr]), length(rightindexset))
    for (i, lindex) in enumerate(leftindexset)
        for (c, cindex) in enumerate(Iterators.product(ntuple(x -> 1:localdims[nl+x], M)...))
            for (j, rindex) in enumerate(rightindexset)
                indexset[1:nl] .= lindex
                indexset[nl+1:nl+M] .= cindex
                indexset[nl+M+1:end] .= rindex
                result[i, c, j] = f(indexset)
            end
        end
    end
    return reshape(result, length(leftindexset), localdims[nl+1:L-nr]..., length(rightindexset))
end


function _batchevaluate_dispatch(
    ::Type{V},
    f::BatchEvaluator{V},
    localdims::Vector{Int},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
    ::Val{M}
)::Array{V,M + 2} where {V,M}
    if length(Iset) * length(Jset) == 0
        return Array{V,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    N = length(localdims)
    nl = length(first(Iset))
    nr = length(first(Jset))
    ncent = N - nl - nr
    return f(Iset, Jset, Val(ncent))
end