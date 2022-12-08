"""
Represents a function that maps ArgType -> ValueType and automatically caches
function values.
"""
struct CachedFunction{ArgType,ValueType}
    f::Function
    d::Dict{ArgType,ValueType}
end

function CachedFunction{ArgType,ValueType}(
    f::Function
) where {ArgType,ValueType}
    return CachedFunction(f, Dict{ArgType,ValueType}())
end

function (cf::CachedFunction{ArgType,ValueType})(
    x::ArgType
) where {ArgType,ValueType}
    return get!(cf.d, x) do
        cf.f(x)
    end
end
