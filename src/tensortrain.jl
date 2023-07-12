"""
    struct TensorTrain{ValueType}

Represents a tensor train, also known as MPS.

The tensor train can be evaluated using standard function call notation:
```julia
    tt = TensorTrain(...)
    value = tt([1, 2, 3, 4])
```
The corresponding function is:

    function (tt::TensorTrain{V})(indexset) where {V}

Evaluates the tensor train `tt` at indices given by `indexset`.
"""
mutable struct TensorTrain{ValueType,N} <: CachedTensorTrain{ValueType}
    T::Vector{Array{ValueType,N}}
    cache::Vector{Dict{MultiIndex,Matrix{ValueType}}}

    function TensorTrain{ValueType,N}(T::Vector{Array{ValueType,N}}) where {ValueType,N}
        for i in 1:length(T)-1
            if (last(size(T[i])) != size(T[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        new{ValueType,N}(T, [Dict{MultiIndex,Matrix{ValueType}}() for _ in 1:length(T)])
    end
end

"""
    function TensorTrain(T::Vector{Array{V, 3}}) where {V}

Create a tensor train out of a vector of tensors. Each tensor should have links to the
previous and next tensor as dimension 1 and 3, respectively; the local index ("physical
index" for MPS in physics) is dimension 2.
"""
function TensorTrain(T::Vector{Array{V,N}}) where {V,N}
    return TensorTrain{V,N}(T)
end

"""
    function TensorTrain(tci::TensorCI{V}) where {V}

Convert the TCI object into a tensor train, also known as an MPS.

See also: [`crossinterpolate`](@ref), [`TensorCI`](@ref)
"""
function TensorTrain(tci::TensorCI{V})::TensorTrain{V,3} where {V}
    return TensorTrain{V,3}(TtimesPinv.(tci, 1:length(tci)))
end

"""
    function TensorTrain(tci::TensorCI2{V}) where {V}

Convert the TCI2 object into a tensor train, also known as an MPS.

See also: [`crossinterpolate`](@ref), [`TensorCI`](@ref)
"""
function TensorTrain(tci::TensorCI2{V})::TensorTrain{V,3} where {V}
    return TensorTrain(tci.T)
end

function tensortrain(tci)
    return TensorTrain(tci)
end

function ttcache(tt::TensorTrain{V,N}, b::Int) where {V,N}
    return tt.cache[b]
end

function emptycache!(tt::TensorTrain{V,N}) where {V,N}
    for d in tt.cache
        empty!(d)
    end
    nothing
end



"""
    struct TTCache{ValueType, N}

Represents a Matrix Product Operator / Tensor Train with N legs on each tensor.
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