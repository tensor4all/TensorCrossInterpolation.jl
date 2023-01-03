"""
Represents a function that maps ArgType -> ValueType and automatically caches
function values.
"""
struct CachedFunction{ValueType}
    f::Function
    dims::Vector{Int}
    d::Dict{BigInt,ValueType}
    coeffs::Vector{BigInt}
    function CachedFunction{ValueType}(f, dims, d) where ValueType
        coeffs = ones(BigInt, length(dims))
        for n in 2:length(dims)
            coeffs[n] = dims[n-1] * coeffs[n-1]
        end
        new(f, dims, d, coeffs)
    end
end

function Base.show(io::IO, f::CachedFunction{ValueType}) where {ValueType}
    print(io, "$(typeof(f)) with $(length(f)) entries")
end

function CachedFunction{ValueType}(
    f::Function,
    dims::Vector{Int}
) where {ValueType}
    return CachedFunction{ValueType}(f, dims, Dict{BigInt,ValueType}())
end

function (cf::CachedFunction{ValueType})(
    x::Vector{T}
) where {ValueType, T<:Number}
    length(x) == length(cf.coeffs) || error("Invalid length of x")
    return get!(cf.d, _key(cf, x)) do
        cf.f(x)
    end
end

_key(cf::CachedFunction{ValueType}, x::Vector{T}) where  {ValueType, T<:Number} = sum(cf.coeffs .* x)