abstract type BatchEvaluator{V} <: AbstractTensorTrain{V} end


"""
    struct TTCache{ValueType, N}

Cached evalulation of TT
"""
struct TTCache{ValueType} <: BatchEvaluator{ValueType}
    sitetensors::Vector{Array{ValueType}}
    cacheleft::Vector{Dict{MultiIndex,Vector{ValueType}}}
    cacheright::Vector{Dict{MultiIndex,Vector{ValueType}}}

    function TTCache(sitetensors::AbstractVector{<:AbstractArray{ValueType}}) where {ValueType}
        new{ValueType}(
            sitetensors,
            [Dict{MultiIndex,Vector{ValueType}}() for _ in sitetensors],
            [Dict{MultiIndex,Vector{ValueType}}() for _ in sitetensors])
    end

end

function Base.empty!(tt::TTCache{V}) where {V}
    empty!(tt.cacheleft)
    empty!(tt.cacheright)
    nothing
end


function TTCache(TT::AbstractTensorTrain{ValueType}) where {ValueType}
    TTCache(sitetensors(TT))
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
        return vec(tt.sitetensors[1][:, indexset[1], :])
    end

    cache = ttcache(tt, :left, ell)
    key = collect(indexset)
    if !(key in keys(cache))
        cache[key] = vec(reshape(evaluateleft(tt, indexset[1:ell-1]), 1, :) * tt.sitetensors[ell][:, indexset[ell], :])
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
        return vec(tt.sitetensors[end][:, indexset[1], :])
    end
    ell = length(tt) - length(indexset) + 1

    cache = ttcache(tt, :right, ell)
    key = collect(indexset)
    if !(key in keys(cache))
        cache[key] = tt.sitetensors[ell][:, indexset[1], :] * evaluateright(tt, indexset[2:end])
    end
    return cache[key]
end

function (tt::TTCache{V})(indexset::AbstractVector{Int}) where {V}
    return evaluate(tt, indexset; usecache=true)
end

_dot(x::Vector{Float64}, y::Vector{Float64}) = LA.BLAS.dot(x, y)
_dot(x::Vector{ComplexF64}, y::Vector{ComplexF64}) = LA.BLAS.dotu(x, y)

function evaluate(
    tt::TTCache{V},
    indexset::AbstractVector{Int};
    usecache::Bool=true,
    midpoint::Int=div(length(tt), 2)
)::V where {V}
    if length(tt) != length(indexset)
        throw(ArgumentError("To evaluate a tensor train of length $(length(tt)), need $(length(tt)) index values, but got $(length(indexset))."))
    end
    if usecache
        return _dot(
            evaluateleft(tt, view(indexset, 1:midpoint)),
            evaluateright(tt, view(indexset, midpoint+1:length(indexset))))
    else
        return sum(prod(T[:, i, :] for (T, i) in zip(tt, indexset)))
    end
end

"""
projector: 0 means no projection, otherwise the index of the projector
"""
function batchevaluate(tt::TTCache{V},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
    projector::Union{Nothing,AbstractVector{Int},NTuple{M,Int}}=nothing)::Array{V,M + 2} where {V,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{V,M + 2}(undef, ntuple(d -> 0, M + 2)...)
    end
    if projector !== nothing && length(projector) != M
        error("Invalid length of projector: $(projector), correct length should be M=$M")
    end
    N = length(tt)
    nleft = length(leftindexset[1])
    nright = length(rightindexset[1])
    nleftindexset = length(leftindexset)
    nrightindexset = length(rightindexset)
    ncent = N - nleft - nright

    if ncent != M
        error("Invalid parameter M: $(M)")
    end

    DL = nleft == 0 ? 1 : size(tt[nleft], 3)
    lenv = ones(V, nleftindexset, DL)
    if nleft > 0
        for (il, lindex) in enumerate(leftindexset)
            lenv[il, :] .= evaluateleft(tt, lindex)
        end
    end

    DR = nright == 0 ? 1 : linkdim(tt, nleft + ncent)
    renv = ones(V, DR, nrightindexset)
    if nright > 0
        for (ir, rindex) in enumerate(rightindexset)
            renv[:, ir] .= evaluateright(tt, rindex)
        end
    end

    localdim = zeros(Int, ncent)
    for n in nleft+1:(N-nright)
        # (nleftindexset, d, ..., d, D) x (D, d, D)
        T_ = if (projector === nothing || projector[n-nleft]==0)
            sitetensor(tt, n)
        else
            s = sitetensor(tt, n)[:, projector[n-nleft], :]
            reshape(s, size(s, 1), 1, size(s, 2))
        end
        localdim[n-nleft] = size(T_, 2)
        bonddim_ = size(T_, 1)
        lenv = reshape(lenv, :, bonddim_) * reshape(T_, bonddim_, :)
    end

    # (nleftindexset, d...d * D) x (D, nrightindexset)
    bonddim_ = size(renv, 1)
    lenv = reshape(lenv, :, bonddim_) * reshape(renv, bonddim_, :)

    return reshape(lenv, nleftindexset, localdim..., nrightindexset)
end



function (tt::TTCache{V})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{V,M + 2} where {V,M}
    return batchevaluate(tt, leftindexset, rightindexset, Val(M))
end


isbatchevaluable(f) = false
isbatchevaluable(f::BatchEvaluator) = true
