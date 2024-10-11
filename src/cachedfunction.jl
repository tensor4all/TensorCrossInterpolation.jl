"""
Represents a function that maps ArgType -> ValueType and automatically caches
function values. A `CachedFunction` object can be called in the same way as the original
function using the usual function call syntax.
The type `K` denotes the type of the keys used to cache function values, which could be an integer type. This defaults to `UInt128`. A safer but slower alternative is `BigInt`, which is better suited for functions with a large number of arguments.
`CachedFunction` does not support batch evaluation of function values.
"""
struct CachedFunction{ValueType,K<:Union{UInt32,UInt64,UInt128,BigInt}} <: BatchEvaluator{ValueType}
    f::Function
    localdims::Vector{Int}
    cache::Dict{K,ValueType}
    coeffs::Vector{K}
    function CachedFunction{ValueType,K}(f, localdims, cache) where {ValueType,K}
        coeffs = ones(K, length(localdims))
        for n in 2:length(localdims)
            coeffs[n] = localdims[n-1] * coeffs[n-1]
        end
        if K != BigInt
            sum(coeffs .* (localdims .- 1)) < typemax(K) || error("Too many dimensions. Use BigInt instead of UInt128.")
        end
        new(f, localdims, cache, coeffs)
    end
end

function Base.show(io::IO, f::CachedFunction{ValueType}) where {ValueType}
    print(io, "$(typeof(f)) with $(length(f.cache)) entries")
end

Base.broadcastable(x::CachedFunction) = Ref(x)

"""
    function CachedFunction{ValueType}(
        f::Function,
        localdims::Vector{Int}
    ) where {ValueType}

Constructor for a cached function that avoids repeated evaluation of the same values.

Arguments:
- `ValueType` is the return type of `f`.
- `K` is the type for the keys used to cache function values. This defaults to `UInt128`.
- `f` is the function to be wrapped. It should take a vector of integers as its sole
argument, and return a value of type `ValueType`.
- `localdims` is a Vector that describes the local dimensions of each argument of `f`.
"""
function CachedFunction{ValueType,K}(
    f::Function,
    localdims::Vector{Int}
) where {ValueType,K}
    return CachedFunction{ValueType,K}(f, localdims, Dict{K,ValueType}())
end


function CachedFunction{ValueType}(
    f::Function,
    localdims::Vector{Int}
) where {ValueType}
    return CachedFunction{ValueType,UInt128}(f, localdims)
end


function (cf::CachedFunction{ValueType,K})(
    x::Vector{T}
) where {ValueType, T<:Number, K}
    length(x) == length(cf.coeffs) || error("Invalid length of x")
    return get!(cf.cache, _key(cf, x)) do
        cf.f(x)
    end
end


function (f::CachedFunction{V,K})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{V,M + 2} where {V,K,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{V,M+2}(undef, ntuple(d->0, M+2)...)
    end
    if isbatchevaluable(f.f)
       return _batcheval_imp_for_batchevaluator(f, leftindexset, rightindexset, Val(M))
    else
       return _batcheval_imp_default(f, leftindexset, rightindexset, Val(M))
    end
end


function _batcheval_imp_default(f::CachedFunction{V,K},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{V,M + 2} where {V,K,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{V,M + 2}(undef, ntuple(d -> 0, M + 2)...)
    end
    nl = length(first(leftindexset))
    nr = length(first(rightindexset))
    L = length(f.localdims)
    sz = vcat(length(leftindexset), f.localdims[nl+1:L-nr], length(rightindexset))
    result = Array{V,M+2}(undef, sz...)
    for (j, rightindex) in enumerate(rightindexset)
        for k in Iterators.product(ntuple(x->1:f.localdims[nl+x], M)...)
            for (i, leftindex) in enumerate(leftindexset)
                indexset = vcat(leftindex, collect(Iterators.flatten(k)), rightindex)
                result[i, k..., j] = get!(f.cache, _key(f, indexset)) do
                    f.f(indexset)
                end
            end
        end
    end
    return result
end


function _batcheval_imp_for_batchevaluator(f::CachedFunction{V,K},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{V,M + 2} where {V,K,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{V,M + 2}(undef, ntuple(d -> 0, M + 2)...)
    end
    nl = length(first(leftindexset))
    nr = length(first(rightindexset))
    L = length(f.localdims)

    result = fill(V(NaN), length(leftindexset), f.localdims[nl+1:L-nr]..., length(rightindexset))

    # Compute the values that are already in the cache, leaving the rest as NaN
    for (j, rightindex) in enumerate(rightindexset)
        for k in Iterators.product(ntuple(x->1:f.localdims[nl+x], M)...)
            for (i, leftindex) in enumerate(leftindexset)
                indexset = vcat(leftindex, collect(Iterators.flatten(k)), rightindex)
                _k = _key(f, indexset)
                if _k ∈ keys(f.cache)
                    result[i, k..., j] = f.cache[_k]
                end
            end
        end
    end

    # Compute the values that are not in the cache and store them in the cache
    leftindexset_ = [leftindex for (i, leftindex) in enumerate(leftindexset) if any(result[i, ..] .=== V(NaN))]
    rightindexset_ = [rightindex for (j, rightindex) in enumerate(rightindexset) if any(result[..,j] .=== V(NaN))]
    result_ = f.f(leftindexset_, rightindexset_, Val(M))
    for (j, rightindex) in enumerate(rightindexset_)
        for k in Iterators.product(ntuple(x->1:f.localdims[nl+x],M)...)
            for (i, leftindex) in enumerate(leftindexset_)
                indexset = vcat(leftindex, collect(Iterators.flatten(k)), rightindex)
                f.cache[_key(f, indexset)] = result_[i, k..., j]
            end
        end
    end

    # Fill in the rest of the result array
    for (j, rightindex) in enumerate(rightindexset)
        for k in Iterators.product(ntuple(x->1:f.localdims[nl+x], M)...)
            for (i, leftindex) in enumerate(leftindexset)
                if result[i, k..., j] !== V(NaN)
                    continue
                end
                indexset = vcat(leftindex, collect(Iterators.flatten(k)), rightindex)
                result[i, k..., j] = f.cache[_key(f, indexset)]
            end
        end
    end

    return result
end

function Base.setindex!(cf::CachedFunction{ValueType,K}, val::ValueType, indexset::Vector{T})::T where {ValueType, T<:Number, K}
    cf.cache[_key(cf, indexset)] = val
end

function _key(cf::CachedFunction{ValueType,K}, indexset::Vector{T})::K where {ValueType, T<:Number, K}
    result = zero(K)
    length(indexset) == length(cf.coeffs) || error("Invalid length of indexset")
    @inbounds @fastmath for i in 1:length(indexset)
        result += cf.coeffs[i] * (indexset[i] - 1)
    end
    return result
end

encodecachekey(cf::CachedFunction{ValueType,K}, index::Vector{Int}) where {ValueType, K} = _key(cf, index)

function decodecachekey(cf::CachedFunction{ValueType,K}, key::K) where {ValueType, K}
    N = length(cf.localdims)
    index = Vector{Int}(undef, N)
    for i in 1:N
        key, r = divrem(key, cf.localdims[i])
        index[i] = r + 1
    end
    return index
end

"""
Get all cached data of a `CachedFunction` object.
"""
function cachedata(cf::CachedFunction{ValueType,K}) where {ValueType, K}
    return Dict(decodecachekey(cf, k) => v for (k, v) in cf.cache)
end

Base.in(x::Vector{T}, cf::CachedFunction{ValueType,K}) where {ValueType, T<:Number, K} = _key(cf, x) ∈ keys(cf.cache)