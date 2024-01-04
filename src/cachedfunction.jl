"""
Represents a function that maps ArgType -> ValueType and automatically caches
function values. A `CachedFunction` object can be called in the same way as the original
function using the usual function call syntax.
The type `K` denotes the type of the keys used to cache function values, which could be an integer type. This defaults to `UInt128`. A safer but slower alternative is `BigInt`, which is better suited for functions with a large number of arguments.
"""
struct CachedFunction{ValueType,K}
    f::Function
    dims::Vector{Int}
    d::Dict{K,ValueType}
    coeffs::Vector{K}
    function CachedFunction{ValueType,K}(f, dims, d) where {ValueType,K}
        coeffs = ones(K, length(dims))
        for n in 2:length(dims)
            coeffs[n] = dims[n-1] * coeffs[n-1]
        end
        if K != BigInt
            sum(coeffs .* (dims .- 1)) < typemax(K) || error("Too many dimensions. Use BigInt instead of UInt128.")
        end
        new(f, dims, d, coeffs)
    end
end

function Base.show(io::IO, f::CachedFunction{ValueType}) where {ValueType}
    print(io, "$(typeof(f)) with $(length(f.d)) entries")
end

Base.broadcastable(x::CachedFunction) = Ref(x)

"""
    function CachedFunction{ValueType}(
        f::Function,
        dims::Vector{Int}
    ) where {ValueType}

Constructor for a cached function that avoids repeated evaluation of the same values.

Arguments:
- `ValueType` is the return type of `f`.
- `K` is the type for the keys used to cache function values. This defaults to `UInt128`.
- `f` is the function to be wrapped. It should take a vector of integers as its sole
argument, and return a value of type `ValueType`.
- `dims` is a Vector that describes the local dimensions of each argument of `f`.
"""
function CachedFunction{ValueType,K}(
    f::Function,
    dims::Vector{Int}
) where {ValueType,K}
    return CachedFunction{ValueType,K}(f, dims, Dict{K,ValueType}())
end

function CachedFunction{ValueType}(
    f::Function,
    dims::Vector{Int}
) where {ValueType}
    return CachedFunction{ValueType,UInt128}(f, dims)
end


function (cf::CachedFunction{ValueType,K})(
    x::Vector{T}
) where {ValueType, T<:Number, K}
    length(x) == length(cf.coeffs) || error("Invalid length of x")
    return get!(cf.d, _key(cf, x)) do
        cf.f(x)
    end
end

function _key(cf::CachedFunction{ValueType,K}, x::Vector{T}) where {ValueType, T<:Number, K}
    all(1 .<= x .<= cf.dims) || error("Invalid x $x $(cf.dims)")
    return sum(cf.coeffs .* (x .- 1))
end
