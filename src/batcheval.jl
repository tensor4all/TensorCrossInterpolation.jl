"""
Wrap any function to support batch evaluation.
"""
struct BatchEvaluatorAdapter{T} <: BatchEvaluator{T}
    f::Function
    localdims::Vector{Int}
end

makebatchevaluatable(::Type{T}, f, localdims) where {T} = BatchEvaluatorAdapter{T}(f, localdims)

function (bf::BatchEvaluatorAdapter{T})(indexset::MultiIndex)::T where T
    bf.f(indexset)
end

function (bf::BatchEvaluatorAdapter{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(d -> 0, M + 2)...)
    end
    return _batchevaluate_dispatch(T, bf.f, bf.localdims, leftindexset, rightindexset, Val(M))
end


"""
This file contains functions for evaluating a function on a batch of indices mainly for TensorCI2.
If the function supports batch evaluation, then it should implement the `BatchEvaluator` interface.
Otherwise, the function is evaluated on each index individually using the usual function call syntax and loops.
"""
function _batchevaluate_dispatch(
    ::Type{V},
    f::F,
    localdims::Vector{Int},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M})::Array{V,M + 2} where {V,F,M}

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


"""
Calling `batchevaluate` on a function that supports batch evaluation will dispatch to this function.
"""
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


@doc """
ThreadedBatchEvaluator{T} is a wrapper for a function that supports batch evaluation parallelized over the index sets using threads.
The function to be wrapped must be thread-safe.
"""
struct ThreadedBatchEvaluator{T} <: BatchEvaluator{T}
    f::Function
    localdims::Vector{Int}
    function ThreadedBatchEvaluator{T}(f, localdims) where {T}
        new{T}(f, localdims)
    end
end

function (obj::ThreadedBatchEvaluator{T})(indexset::Vector{Int})::T where {T}
    return obj.f(indexset)
end

# Batch evaluation (loop over all index sets)
function (obj::ThreadedBatchEvaluator{T})(leftindexset::Vector{Vector{Int}}, rightindexset::Vector{Vector{Int}}, ::Val{M})::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M+2}(undef, ntuple(i->0, M+2)...)
    end

    nl = length(first(leftindexset))

    cindexset = vec(collect(Iterators.product(ntuple(i->1:obj.localdims[nl+i], M)...)))
    elements = collect(Iterators.product(1:length(leftindexset), 1:length(cindexset), 1:length(rightindexset)))
    result = Array{T,3}(undef, length(leftindexset), length(cindexset), length(rightindexset))

    Threads.@threads for indices in elements
        l, c, r = leftindexset[indices[1]], cindexset[indices[2]], rightindexset[indices[3]]
        result[indices...] = obj.f(vcat(l, c..., r))
    end
    return reshape(result, length(leftindexset), obj.localdims[nl+1:nl+M]..., length(rightindexset))
end