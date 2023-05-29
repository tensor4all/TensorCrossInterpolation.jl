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
- `localdims::Union{Vector{Int},NTuple{N,Int}}` determines the local dimensions of the function parameters (see [`crossinterpolate`](@ref)).
- `fistpivot::MultiIndex=ones(Int, length(localdims))` is the starting point for the optimization. It is advantageous to choose it close to a global maximum of the function.
- `maxsweep` is the maximum number of optimization sweeps. Default: `1000`.

See also: [`crossinterpolate`](@ref)
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

    for _ in 1:maxsweep
        valf_prev = valf
        for i in 1:n
            for d in 1:localdims[i]
                bak = pivot[i]
                pivot[i] = d
                if abs(f(pivot)) > valf
                    valf = abs(f(pivot))
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
