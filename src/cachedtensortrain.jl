abstract type CachedTensorTrain{V} <: AbstractTensorTrain{V} end

function evaluatepartial(
    tt::CachedTensorTrain{V},
    indexset::Union{AbstractVector{LocalIndex}, NTuple{N, LocalIndex}},
    ell::Int
)::Matrix{V} where {N, V}
    if ell < 1 || ell > length(tt)
        throw(ArgumentError("Got site index $ell for a tensor train of length $(length(tt))."))
    end

    if ell == 1
        return tt[1][:, indexset[1], :]
    end

    cache = ttcache(tt, ell)
    key = collect(indexset[1:ell])
    if !(key in keys(cache))
        cache[key] = evaluatepartial(tt, indexset, ell-1) * tt[ell][:, indexset[ell], :]
    end
    return cache[key]
end

function evaluate(
    tt::CachedTensorTrain{V},
    indexset::Union{AbstractVector{LocalIndex}, NTuple{N, LocalIndex}}
)::V where {N, V}
    if length(tt) != length(indexset)
        throw(ArgumentError("To evaluate a tensor train of length $(length(tt)), need $(length(tt)) index values, but only got $(length(indexset))."))
    end
    return only(evaluatepartial(tt, indexset, length(tt)))
end
