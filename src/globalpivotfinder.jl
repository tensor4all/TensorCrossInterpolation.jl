import Random: AbstractRNG, default_rng

"""
    GlobalPivotSearchInput{ValueType}

Input data structure for global pivot search algorithms.

# Fields
- `localdims::Vector{Int}`: Dimensions of each tensor index
- `current_tt::TensorTrain{ValueType,3}`: Current tensor train approximation
- `maxsamplevalue::ValueType`: Maximum absolute value of the function
- `Iset::Vector{Vector{MultiIndex}}`: Set of left indices
- `Jset::Vector{Vector{MultiIndex}}`: Set of right indices
"""
struct GlobalPivotSearchInput{ValueType}
    localdims::Vector{Int}
    current_tt::TensorTrain{ValueType,3}
    maxsamplevalue::Float64
    Iset::Vector{Vector{MultiIndex}}
    Jset::Vector{Vector{MultiIndex}}

    """
        GlobalPivotSearchInput(
            localdims::Vector{Int},
            current_tt::TensorTrain{ValueType,3},
            maxsamplevalue::ValueType,
            Iset::Vector{Vector{MultiIndex}},
            Jset::Vector{Vector{MultiIndex}}
        ) where {ValueType}

    Construct a GlobalPivotSearchInput with the given fields.
    """
    function GlobalPivotSearchInput{ValueType}(
        localdims::Vector{Int},
        current_tt::TensorTrain{ValueType,3},
        maxsamplevalue::Float64,
        Iset::Vector{Vector{MultiIndex}},
        Jset::Vector{Vector{MultiIndex}}
    ) where {ValueType}
        new{ValueType}(
            localdims,
            current_tt,
            maxsamplevalue,
            Iset,
            Jset
        )
    end
end


"""
    AbstractGlobalPivotFinder

Abstract type for global pivot finders that search for indices with high interpolation error.
"""
abstract type AbstractGlobalPivotFinder end

"""
    (finder::AbstractGlobalPivotFinder)(
        input::GlobalPivotSearchInput{ValueType},
        f,
        abstol::Float64;
        verbosity::Int=0,
        rng::AbstractRNG=Random.default_rng()
    )::Vector{MultiIndex} where {ValueType}

Find global pivots using the given finder algorithm.

# Arguments
- `input`: Input data for the search algorithm
- `f`: Function to be interpolated
- `abstol`: Absolute tolerance for the interpolation error
- `verbosity`: Verbosity level (default: 0)
- `rng`: Random number generator (default: Random.default_rng())

# Returns
- `Vector{MultiIndex}`: Set of indices with high interpolation error
"""
function (finder::AbstractGlobalPivotFinder)(
    input::GlobalPivotSearchInput{ValueType},
    f,
    abstol::Float64;
    verbosity::Int=0,
    rng::AbstractRNG=Random.default_rng()
)::Vector{MultiIndex} where {ValueType}
    error("find_global_pivots not implemented for $(typeof(finder))")
end

"""
    DefaultGlobalPivotFinder

Default implementation of global pivot finder that uses random search.

# Fields
- `nsearch::Int`: Number of initial points to search from
- `maxnglobalpivot::Int`: Maximum number of pivots to add in each iteration
- `tolmarginglobalsearch::Float64`: Search for pivots where the interpolation error is larger than the tolerance multiplied by this factor
"""
struct DefaultGlobalPivotFinder <: AbstractGlobalPivotFinder
    nsearch::Int
    maxnglobalpivot::Int
    tolmarginglobalsearch::Float64
end

"""
    DefaultGlobalPivotFinder(;
        nsearch::Int=5,
        maxnglobalpivot::Int=5,
        tolmarginglobalsearch::Float64=10.0
    )

Construct a DefaultGlobalPivotFinder with the given parameters.
"""
function DefaultGlobalPivotFinder(;
    nsearch::Int=5,
    maxnglobalpivot::Int=5,
    tolmarginglobalsearch::Float64=10.0
)
    return DefaultGlobalPivotFinder(nsearch, maxnglobalpivot, tolmarginglobalsearch)
end

"""
    (finder::DefaultGlobalPivotFinder)(
        input::GlobalPivotSearchInput{ValueType},
        f,
        abstol::Float64;
        verbosity::Int=0,
        rng::AbstractRNG=Random.default_rng()
    )::Vector{MultiIndex} where {ValueType}

Find global pivots using random search.

# Arguments
- `input`: Input data for the search algorithm
- `f`: Function to be interpolated
- `abstol`: Absolute tolerance for the interpolation error
- `verbosity`: Verbosity level (default: 0)
- `rng`: Random number generator (default: Random.default_rng())

# Returns
- `Vector{MultiIndex}`: Set of indices with high interpolation error
"""
function (finder::DefaultGlobalPivotFinder)(
    input::GlobalPivotSearchInput{ValueType},
    f,
    abstol::Float64;
    verbosity::Int=0,
    rng::AbstractRNG=Random.default_rng()
)::Vector{MultiIndex} where {ValueType}
    L = length(input.localdims)
    nsearch = finder.nsearch
    maxnglobalpivot = finder.maxnglobalpivot
    tolmarginglobalsearch = finder.tolmarginglobalsearch

    # Generate random initial points
    initial_points = [[rand(rng, 1:input.localdims[p]) for p in 1:L] for _ in 1:nsearch]

    # Find pivots with high interpolation error
    found_pivots = MultiIndex[]
    for point in initial_points
        # Perform local search from each initial point
        current_point = copy(point)
        best_error = 0.0
        best_point = copy(point)

        # Local search
        for p in 1:L
            for v in 1:input.localdims[p]
                current_point[p] = v
                error = abs(f(current_point) - input.current_tt(current_point))
                if error > best_error
                    best_error = error
                    best_point = copy(current_point)
                end
            end
            current_point[p] = point[p]  # Reset to original point
        end

        # Add point if error is above threshold
        if best_error > abstol * tolmarginglobalsearch
            push!(found_pivots, best_point)
        end
    end

    # Limit number of pivots
    if length(found_pivots) > maxnglobalpivot
        found_pivots = found_pivots[1:maxnglobalpivot]
    end

    if verbosity > 0
        println("Found $(length(found_pivots)) global pivots")
    end

    return found_pivots
end 