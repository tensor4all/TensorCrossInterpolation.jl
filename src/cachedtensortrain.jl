abstract type BatchEvaluator{V} <: AbstractTensorTrain{V} end


"""
    struct TTCache{ValueType}

Cached evalulation of a tensor train. This is useful when the same TT is evaluated multiple times with the same indices. The number of site indices per tensor core can be arbitray irrespective of the number of site indices of the original tensor train.
"""
struct TTCache{ValueType} <: BatchEvaluator{ValueType}
    sitetensors::Vector{Array{ValueType,3}}
    cacheleft::Vector{Dict{MultiIndex,Vector{ValueType}}}
    cacheright::Vector{Dict{MultiIndex,Vector{ValueType}}}
    sitedims::Vector{Vector{Int}}

    function TTCache{ValueType}(sitetensors::AbstractVector{<:AbstractArray{ValueType}}, sitedims) where {ValueType}
        length(sitetensors) == length(sitedims) || throw(ArgumentError("The number of site tensors and site dimensions must be the same."))
        for n in 1:length(sitetensors)
            prod(sitedims[n]) == prod(size(sitetensors[n])[2:end-1]) || error("Site dimensions do not match the site tensor dimensions at $n.")
        end
        new{ValueType}(
            [reshape(x, size(x, 1), :, size(x)[end]) for x in sitetensors],
            [Dict{MultiIndex,Vector{ValueType}}() for _ in sitetensors],
            [Dict{MultiIndex,Vector{ValueType}}() for _ in sitetensors],
            sitedims
        )
    end
    function TTCache{ValueType}(sitetensors::AbstractVector{<:AbstractArray{ValueType}}) where {ValueType}
        return TTCache{ValueType}(sitetensors, [collect(size(x)[2:end-1]) for x in sitetensors])
    end
end

TTCache(tt::AbstractTensorTrain{ValueType}) where {ValueType} = TTCache{ValueType}(sitetensors(tt))

TTCache(sitetensors::AbstractVector{<:AbstractArray{ValueType}}) where {ValueType} = TTCache{ValueType}(sitetensors)

TTCache(sitetensors::AbstractVector{<:AbstractArray{ValueType}}, sitedims) where {ValueType} = TTCache{ValueType}(sitetensors, sitedims)

TTCache(tt::AbstractTensorTrain{ValueType}, sitedims) where {ValueType} = TTCache{ValueType}(sitetensors(tt), sitedims)

Base.length(obj::TTCache) = length(obj.sitetensors)

sitedims(obj::TTCache)::Vector{Vector{Int}} = obj.sitedims

function sitetensors(tt::TTCache{V}) where {V}
    return [sitetensor(tt, n) for n in 1:length(tt)]
end

function sitetensor(tt::TTCache{V}, i) where {V}
    sitetensor = tt.sitetensors[i]
    return reshape(
        sitetensor,
        size(sitetensor)[1], tt.sitedims[i]..., size(sitetensor)[end]
    )
end

function Base.empty!(tt::TTCache{V}) where {V}
    empty!(tt.cacheleft)
    empty!(tt.cacheright)
    nothing
end


#function TTCache(TT::AbstractTensorTrain{ValueType}) where {ValueType}
    #TTCache(sitetensors(TT))
#end

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
    projector::Union{Nothing,AbstractVector{<:AbstractVector{<:Integer}}}=nothing)::Array{V,M + 2} where {V,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{V,M + 2}(undef, ntuple(d -> 0, M + 2)...)
    end
    N = length(tt)
    nleft = length(leftindexset[1])
    nright = length(rightindexset[1])
    nleftindexset = length(leftindexset)
    nrightindexset = length(rightindexset)
    ncent = N - nleft - nright
    s_, e_ = nleft + 1, N - nright

    if ncent != M
        error("Invalid parameter M: $(M)")
    end
    if projector === nothing
        projector = [fill(0, length(s)) for s in sitedims(tt)[nleft+1:(N-nright)]]
    end
    if length(projector) != M
        error("Invalid length of projector: $(projector), correct length should be M=$M")
    end
    for n in s_:e_
        length(projector[n-s_+1]) == length(tt.sitedims[n]) || error("Invalid projector at $n: $(projector[n - s_ + 1]), the length must be $(length(tt.sitedims[n]))")
        all(0 .<= projector[n-s_+1] .<= tt.sitedims[n]) || error("Invalid projector: $(projector[n - s_ + 1])")
    end

    DL = nleft == 0 ? 1 : prod(tt.sitedims[nleft])
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
        T_ = begin
            slice, slice_size = projector_to_slice(projector[n-nleft])
            s = sitetensor(tt, n)[:, slice..., :]
            reshape(s, size(s)[1], :, size(s)[end])
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
