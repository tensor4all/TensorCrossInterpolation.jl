mutable struct TensorCIBase{ValueType}
    Iset::Vector{IndexSet{MultiIndex}}
    Jset::Vector{IndexSet{MultiIndex}}
    localset::Vector{Vector{Int}} # Do we need this?

    T::Vector{Array{ValueType,3}} 
    P::Vector{Matrix{ValueType}}
end


function Base.show(io::IO, tci::TensorCIBase{ValueType}) where {ValueType}
    #print(io, "$(typeof(tci)) with ranks $(rank(tci))")
    print(io, "$(typeof(tci))")
end


function Base.length(t::TensorCIBase)
    return length(t.localset)
end

function TensorCIBase{ValueType}(
    func::F,
    localdims::Vector{Int},
    firstpivot::Vector{Int}
) where {F,ValueType}
    n  = length(localdims)

    Iset = [IndexSet([firstpivot[1:p-1]]) for p in 1:n]
    Jset = [IndexSet([firstpivot[p+1:end]]) for p in 1:n]
    localset = [collect(1:d) for d in localdims]

    T = [zeros(ValueType, length(Iset[p]), localdims[p], length(Jset[p])) for p in 1:n]
    P = [zeros(ValueType, length(Iset[p+1]), length(Jset[p])) for p in 1:n-1]
    push!(P, ones(ValueType, 1, 1))

    tci = TensorCIBase{ValueType}(Iset, Jset, localset, T, P)

    fill_TP!(tci, func)

    return tci
end


"""
Fill in T and P tensors from Iset and Jset
"""
function fill_TP!(tci::TensorCIBase{V}, func) where {V}
    n = length(tci)
    length(tci.T) == n || error("Invalid length of T")
    length(tci.P) == n || error("Invalid length of P")

    # Fill in T tensors
    for p in 1:n
        for (posj, j) in enumerate(tci.Jset[p].fromint)
            for (posd, d) in enumerate(tci.localset[p])
                for (posi, i) in enumerate(tci.Iset[p].fromint)
                    tci.T[p][posi, posd, posj] = func([i..., d, j...])
                end
            end
        end
    end

    # Fill in P tensors
    for p in 1:n-1
        for (posj, j) in enumerate(tci.Jset[p].fromint)
            for (posi, i) in enumerate(tci.Iset[p+1].fromint)
                tci.P[p][posi, posj] = func([i..., j...])
            end
        end
    end
    tci.P[end] = ones(V, 1, 1)
    
    return nothing
end



"""
    function evaluate(tci::TensorCIBase{V}, indexset::AbstractVector{LocalIndex}) where {V}

Evaluate the TCI at a specific set of indices. This method is inefficient; to evaluate
many points, first convert the TCI object into a tensor train using `tensortrain(tci)`.
"""
function evaluate(tci::TensorCIBase{V}, indexset::AbstractVector{LocalIndex}) where {V}
    return prod(
        AtimesBinv(tci.T[p][:, indexset[p], :], tci.P[p])
        for p in 1:length(tci))[1, 1]
end
