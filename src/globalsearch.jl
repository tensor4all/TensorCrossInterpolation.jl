"""
    function estimatetrueerror(
        tt::TensorTrain{ValueType,3}, f;
        nsearch::Int = 100,
        initialpoints::Union{Nothing,AbstractVector{MultiIndex}} = nothing,
    )::Vector{Tuple{MultiIndex,Float64}} where {ValueType}

Estimate the global error by comparing the exact function value and
the TT approximation using a greedy algorithm and returns a unique list of pairs of the pivot and the error (error is the absolute difference between the exact function value and the TT approximation).
On return, the list is sorted in descending order of the error.

Arguments:
- `tt::TensorTrain{ValueType,3}`: The tensor train to be compared with `f`
- `f`: The function to be compared with `tt`.
- `nsearch::Int`: The number of initial points to be used in the search (defaults to 100).
- `initialpoints::Union{Nothing,AbstractVector{MultiIndex}}`: The initial points to be used in the search (defaults to `nothing`). If `initialpoints` is not `nothing`, `nsearch` is ignored.
"""
function estimatetrueerror(
    tt::TensorTrain{ValueType,3}, f;
    nsearch::Int = 100,
    initialpoints::Union{Nothing,AbstractVector{MultiIndex}} = nothing,
)::Vector{Tuple{MultiIndex,Float64}} where {ValueType}
    if nsearch <= 0 && initialpoints === nothing
        error("No search is performed")
    end
    nsearch >= 0 || error("nsearch must be non-negative")

    if nsearch > 0 && initialpoints === nothing
        # Use random initial points
        initialpoints = [[rand(1:first(d)) for d in sitedims(tt)] for _ in 1:nsearch]
    end

    ttcache = TTCache(tt)

    pivoterror = [_floatingzone(ttcache, f; initp=initp) for initp in initialpoints]

    p = sortperm([e for (_, e) in pivoterror], rev=true)

    return unique(pivoterror[p])
end


function _floatingzone(
    ttcache::TTCache{ValueType}, f;
    earlystoptol::Float64 = typemax(Float64),
    nsweeps=typemax(Int), initp::Union{Nothing,MultiIndex} = nothing
)::Tuple{MultiIndex,Float64} where {ValueType}
    nsweeps > 0 || error("nsweeps should be positive!")

    localdims = first.(sitedims(ttcache))

    n = length(ttcache)

    if initp === nothing
        pivot = [rand(1:d) for d in localdims]
    else
        pivot = initp
    end

    maxerror = abs(f(pivot) - ttcache(pivot))

    for isweep in 1:nsweeps
        prev_maxerror = maxerror
        for ipos in 1:n
            exactdata = filltensor(
                ValueType,
                f,
                localdims,
                [pivot[1:ipos-1]],
                [pivot[ipos+1:end]],
                Val(1)
            )
            prediction = filltensor(
                ValueType,
                ttcache,
                localdims,
                [pivot[1:ipos-1]],
                [pivot[ipos+1:end]],
                Val(1)
            )
            err = vec(abs.(exactdata .- prediction))
            pivot[ipos] = argmax(err)
            # In RHS, we compare the maximum of the error vector with the current maxerror
            # to make sure that the error does not decrease even if maxerror is close to machine precision.
            maxerror = max(maximum(err), maxerror)
        end

        if maxerror == prev_maxerror || maxerror > earlystoptol # early stop
            break
        end
    end

    return pivot, maxerror
end


function fillsitetensors!(
    tci::TensorCI2{ValueType}, f) where {ValueType}
    for b in 1:length(tci)
       setsitetensor!(tci, f, b)
    end
    nothing
end


function _sanitycheck(tci::TensorCI2{ValueType})::Bool where {ValueType}
    for b in 1:length(tci)-1
        length(tci.Iset[b+1]) == length(tci.Jset[b]) || error("Pivot matrix at bond $(b) is not square!")
    end

    return true
end