"""
Represents a function that maps ArgType -> ValueType and automatically caches
function values. A `CachedFunction` object can be called in the same way as the original
function using the usual function call syntax.
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

Base.broadcastable(x::CachedFunction) = Ref(x)

"""
    function CachedFunction{ValueType}(
        f::Function,
        dims::Vector{Int}
    ) where {ValueType}

Constructor for a cached function that avoids repeated evaluation of the same values.

Arguments:
- `ValueType` is the return type of `f`.
- `f` is the function to be wrapped. It should take a vector of integers as its sole
argument, and return a value of type `ValueType`.
- `dims` is a Vector that describes the local dimensions of each argument of `f`.
"""
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

function _key(cf::CachedFunction{ValueType}, x::Vector{T}) where  {ValueType, T<:Number}
    all(1 .<= x .<= cf.dims) || error("Invalid x $x $(cf.dims)")
    return sum(cf.coeffs .* x)
end
