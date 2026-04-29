struct _BatchedFunction{F,B} <: Function
    f::F
    batchedf::B
    localdims::Vector{Int}
end

function (bf::_BatchedFunction)(indexset::MultiIndex)
    bf.f(indexset)
end

"""
This file contains functions for evaluating a function on a batch of indices mainly for TensorCI2.
If `batchedf` is supplied, it is called with an integer matrix whose columns are
global index sets. Otherwise, the function is evaluated on each index
individually using the usual function call syntax and loops.
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
    bf::_BatchedFunction,
    localdims::Vector{Int},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M})::Array{V,M + 2} where {V,M}

    return _batchevaluate_dispatch(V, bf.f, bf.batchedf, localdims, leftindexset, rightindexset, Val(M))
end

function _batchevaluate_dispatch(
    ::Type{V},
    f,
    batchedf,
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
    center_dims = localdims[nl+1:L-nr]
    result_dims = (length(leftindexset), center_dims..., length(rightindexset))
    npoints = prod(result_dims)
    indices = Matrix{Int}(undef, L, npoints)

    for (p, cartesian_index) in enumerate(CartesianIndices(result_dims))
        lindex = leftindexset[cartesian_index[1]]
        cindex = ntuple(i -> cartesian_index[i+1], M)
        rindex = rightindexset[cartesian_index[M+2]]
        indices[1:nl, p] .= lindex
        indices[nl+1:nl+M, p] .= cindex
        indices[nl+M+1:end, p] .= rindex
    end

    values = batchedf(indices)
    if length(values) != npoints
        throw(DimensionMismatch("batchedf returned $(length(values)) values for $npoints points"))
    end
    return Array{V,M + 2}(reshape(values, result_dims))
end
