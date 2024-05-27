function maxabs(
    maxval::T,
    updates::AbstractArray{U}
) where {T,U}
    return reduce(
        (x, y) -> max(abs(x), abs(y)),
        updates[:],
        init=maxval
    )
end

function padzero(a::AbstractVector{T}) where {T}
    return Iterators.flatten((a, Iterators.repeated(0)))
end

function pushunique!(collection, item)
    if !(item in collection)
        push!(collection, item)
    end
end

function pushunique!(collection, items...)
    for item in items
        pushunique!(collection, item)
    end
end

function isconstant(collection)
    if isempty(collection)
        return true
    end
    c = first(collection)
    return all(collection .== c)
end

function randomsubset(
    set::AbstractArray{ValueType}, n::Int
)::Array{ValueType} where {ValueType}
    n = min(n, length(set))
    if n <= 0
        return ValueType[]
    end

    c = copy(set[:])
    subset = Array{ValueType}(undef, n)
    for i in 1:n
        index = rand(1:length(c))
        subset[i] = c[index]
        deleteat!(c, index)
    end
    return subset
end

function pushrandomsubset!(subset, set, n::Int)
    topush = randomsubset(setdiff(set, subset), n)
    push!(subset, topush...)
    nothing
end

"""
    function optfirstpivot(
        f,
        localdims::Union{Vector{Int},NTuple{N,Int}},
        firstpivot::MultiIndex=ones(Int, length(localdims));
        maxsweep=1000
    ) where {N}

Optimize the first pivot for a tensor cross interpolation.

Arguments:
- `f` is function to be interpolated.
- `localdims::Union{Vector{Int},NTuple{N,Int}}` determines the local dimensions of the function parameters (see [`crossinterpolate1`](@ref)).
- `fistpivot::MultiIndex=ones(Int, length(localdims))` is the starting point for the optimization. It is advantageous to choose it close to a global maximum of the function.
- `maxsweep` is the maximum number of optimization sweeps. Default: `1000`.

See also: [`crossinterpolate1`](@ref)
"""
function optfirstpivot(
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    firstpivot::Vector{Int}=ones(Int, length(localdims));
    maxsweep=1000
) where {N}
    n = length(localdims)
    valf = abs(f(firstpivot))
    pivot = copy(firstpivot)

    # TODO: use batch evaluation
    for _ in 1:maxsweep
        valf_prev = valf
        for i in 1:n
            for d in 1:localdims[i]
                bak = pivot[i]
                pivot[i] = d
                newval = abs(f(pivot))
                if newval > valf
                    valf = newval
                else
                    pivot[i] = bak
                end
            end
        end
        if valf_prev == valf
            break
        end
    end

    return pivot
end

function replacenothing(value::Union{T, Nothing}, default::T)::T where {T}
    if isnothing(value)
        return default
    else
        return value
    end
end


"""
Construct slice for the site indces of one tensor core
Returns a slice and the corresponding shape for `resize`
"""
function projector_to_slice(p::AbstractVector{<:Integer})
    return [x == 0 ? Colon() : x for x in p], [x == 0 ? Colon() : 1 for x in p]
end