abstract type BatchEvaluator{V} <: AbstractTensorTrain{V} end


"""
    struct TTCache{ValueType, N}

Cached evalulation of TT
"""
struct TTCache{ValueType} <: BatchEvaluator{ValueType}
    T::Vector{Array{ValueType}}
    cacheleft::Vector{Dict{MultiIndex,Vector{ValueType}}}
    cacheright::Vector{Dict{MultiIndex,Vector{ValueType}}}

    function TTCache(T::AbstractVector{<:AbstractArray{ValueType,3}}) where {ValueType}
        new{ValueType}(
            T,
            [Dict{MultiIndex,Vector{ValueType}}() for _ in T],
            [Dict{MultiIndex,Vector{ValueType}}() for _ in T])
    end
end

Base.length(tt::TTCache{V}) where {V} = length(tt.T)


function Base.empty!(tt::TTCache{V}) where {V}
    empty!(tt.cacheleft)
    empty!(tt.cacheright)
    nothing
end


function TTCache(TT::AbstractTensorTrain{ValueType}) where {ValueType}
    TTCache(TT.T)
end

function ttcache(tt::TTCache{V}, leftright::Symbol, b::Int) where {V}
    if leftright == :left
        return tt.cacheleft[b]
    elseif leftright == :right
        return tt.cacheright[b]
    else
        throw(ArgumentError("Choose from left or right cache."))
    end
end

function evaluateleft(
    tt::TTCache{V},
    indexset::AbstractVector{Int}
)::Vector{V} where {V}
    if length(indexset) > length(tt)
        throw(ArgumentError("For evaluateleft, number of indices must be smaller or equal to the number of TTCache legs."))
    end

    ell = length(indexset)
    if ell == 0
        return V[1]
    elseif ell == 1
        return vec(tt.T[1][:, indexset[1], :])
    end

    cache = ttcache(tt, :left, ell)
    key = collect(indexset)
    if !(key in keys(cache))
        cache[key] = vec(reshape(evaluateleft(tt, indexset[1:ell-1]), 1, :) * tt.T[ell][:, indexset[ell], :])
    end
    return cache[key]
end

function evaluateright(
    tt::TTCache{V},
    indexset::AbstractVector{Int}
)::Vector{V} where {V}
    if length(indexset) > length(tt)
        throw(ArgumentError("For evaluateright, number of indices must be smaller or equal to the number of TTCache legs."))
    end

    if length(indexset) == 0
        return V[1]
    elseif length(indexset) == 1
        return vec(tt.T[end][:, indexset[1], :])
    end
    ell = length(tt) - length(indexset) + 1

    cache = ttcache(tt, :right, ell)
    key = collect(indexset)
    if !(key in keys(cache))
        cache[key] = tt.T[ell][:, indexset[1], :] * evaluateright(tt, indexset[2:end])
    end
    return cache[key]
end

function (tt::TTCache{V})(indexset::AbstractVector{Int}) where {V}
    return evaluate(tt, indexset; usecache=true)
end

function evaluate(
    tt::TTCache{V},
    indexset::AbstractVector{Int};
    usecache::Bool=true
)::V where {V}
    if length(tt) != length(indexset)
        throw(ArgumentError("To evaluate a tensor train of length $(length(tt)), need $(length(tt)) index values, but only got $(length(indexset))."))
    end
    if usecache
        midpoint = div(length(tt), 2)
        return sum(
            evaluateleft(tt, indexset[1:midpoint]) .*
            evaluateright(tt, indexset[midpoint+1:end]))
    else
        return sum(prod(T[:, i, :] for (T, i) in zip(tt, indexset)))
    end
end


function batchevaluate(
    tt::TTCache{V},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{V,M + 2} where {V,M}
    N = length(tt.T)
    nleft = length(leftindexset[1])
    nright = length(rightindexset[1])
    nleftindexset = length(leftindexset)
    nrightindexset = length(rightindexset)
    ncent = N - nleft - nright

    if ncent != M
        error("Invalid parameter M: $(M)")
    end

    DL = nleft == 0 ? 1 : size(tt.T[nleft], 3)
    lenv = ones(V, nleftindexset, DL)
    if nleft > 0
        for (il, lindex) in enumerate(leftindexset)
            lenv[il, :] .= evaluateleft(tt, lindex)
        end
    end

    DR = nright == 0 ? 1 : size(tt.T[nleft+ncent+1], 1)
    renv = ones(V, DR, nrightindexset)
    if nright > 0
        for (ir, rindex) in enumerate(rightindexset)
            renv[:, ir] .= evaluateright(tt, rindex)
        end
    end

    localdim = zeros(Int, ncent)
    for n in nleft+1:(N-nright)
        # (nleftindexset, d, ..., d, D) x (D, d, D)
        T_ = tt.T[n]
        localdim[n-nleft] = size(T_, 2)
        bonddim_ = size(T_, 1)
        lenv = reshape(lenv, :, bonddim_) * reshape(T_, bonddim_, :)
    end

    # (nleftindexset, d...d * D) x (D, nrightindexset)
    bonddim_ = size(renv, 1)
    lenv = reshape(lenv, :, bonddim_) * reshape(renv, bonddim_, :)

    return reshape(lenv, nleftindexset, localdim..., nrightindexset)
end


isbatchevaluable(f) = false
isbatchevaluable(f::BatchEvaluator) = true