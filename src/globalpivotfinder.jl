
"""
Default implementation of global pivot finder that uses random search.
"""
struct DefaultGlobalPivotFinder <: AbstractGlobalPivotFinder
    nsearch::Int
    maxnglobalpivot::Int
    tolmarginglobalsearch::Float64
end

# Constructor for backward compatibility
function DefaultGlobalPivotFinder(;
    nsearch::Int=5,
    maxnglobalpivot::Int=5,
    tolmarginglobalsearch::Float64=10.0
)
    return DefaultGlobalPivotFinder(nsearch, maxnglobalpivot, tolmarginglobalsearch)
end

"""
Find global pivots where the interpolation error exceeds the tolerance.
"""
function (finder::AbstractGlobalPivotFinder)(
    tci::TensorCI2{ValueType},
    f,
    abstol::Float64;
    verbosity::Int=0
)::Vector{MultiIndex} where {ValueType}
    error("find_global_pivots not implemented for $(typeof(finder))")
end

# Default implementation using the existing searchglobalpivots
function (finder::DefaultGlobalPivotFinder)(
    tci::TensorCI2{ValueType},
    f,
    abstol::Float64;
    verbosity::Int=0
)::Vector{MultiIndex} where {ValueType}
    return searchglobalpivots(
        tci, f, finder.tolmarginglobalsearch * abstol,
        verbosity=verbosity,
        nsearch=finder.nsearch,
        maxnglobalpivot=finder.maxnglobalpivot
    )
end

# Helper function for backward compatibility
function _create_default_finder(;
    nsearch::Int=5,
    maxnglobalpivot::Int=5,
    tolmarginglobalsearch::Float64=10.0
)
    return DefaultGlobalPivotFinder(
        nsearch=nsearch,
        maxnglobalpivot=maxnglobalpivot,
        tolmarginglobalsearch=tolmarginglobalsearch
    )
end 