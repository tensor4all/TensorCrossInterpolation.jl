"""
Represents a function that maps ArgType -> ValueType and automatically caches
function values.
"""
struct CachedFunction{ValueType}
    f::Function
    d::Dict{UInt,ValueType}
end

function Base.show(io::IO, f::CachedFunction{ValueType}) where {ValueType}
    print(io, "$(typeof(f)) with $(length(f)) entries")
end

function CachedFunction{ValueType}(
    f::Function
) where {ValueType}
    return CachedFunction(f, Dict{UInt,ValueType}())
end

function (cf::CachedFunction{ValueType})(
    x::ArgType
) where {ArgType,ValueType}
    hx = hash(x)
    return get!(cf.d, hx) do
        cf.f(x)
    end
end
