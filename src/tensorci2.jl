mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{IndexSet{MultiIndex}}
    Jset::Vector{IndexSet{MultiIndex}}
    localset::Vector{Vector{LocalIndex}}

    T::Vector{Array{ValueType,3}}

    pivoterrors::Vector{Float64}
    maxsample::Float64

    function TensorCI2{ValueType}(
        localdims::Union{Vector{Int},NTuple{N,Int}}
    ) where {ValueType,N}
        n = length(localdims)
        new{ValueType}(
            [IndexSet{MultiIndex}() for _ in 1:n],  # Iset
            [IndexSet{MultiIndex}() for _ in 1:n],  # Jset
            [collect(1:d) for d in localdims],      # localset
            [zeros(0, d, 0) for d in localdims],    # T
            fill(Inf, n - 1),                       # pivoterrors
            0.0                                     # maxsample
        )
    end
end

function TensorCI2{ValueType}(
    func::F,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    firstpivot::Vector{LocalIndex}
) where{F, ValueType,N}
    tci = TensorCI2{ValueType}(localdims)
    f(x) = convert(ValueType, func(x))
    addglobalpivot!(tci, f, firstpivot)
end

function addglobalpivot!(
    tci::TensorCI2{ValueType},
    f::F,
    pivot::Vector{LocalIndex}
) where {F, ValueType}
    if length(tci) != length(pivot)
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the MPS."))
    end

    for b in 1:length(tci)
        push!(Iset[b], pivot[1:b])
        push!(Jset[b], pivot[b+1:end])
    end

    makecanonical!(tci, f)
end

function getPi(
    f::F,
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex}
) where {ValueType,F}
    return ValueType[f(vcat(i, j)) for i in Iset, j in Jset]
end

function kronecker(Iset::Vector{MultiIndex}, localset::Vector{LocalIndex})
    return MultiIndex[[is..., j] for is in Iset, j in localset][:]
end

function kronecker(localset::Vector{LocalIndex}, Jset::Vector{MultiIndex})
    return MultiIndex[[i, js...] for i in localset, js in Jset][:]
end

function makecanonical!(
    tci::TensorCI2{ValueType},
    f::F,
    reltol::Float64,
    maxbonddim::Int
) where {F, ValueType}
    for b in 1:length(tci)
        Icombined = kronecker(tci.Iset[b], tci.localset[b])
        ludecomp = rrlu(
            getPi(f, Icombined, tci.Jset[b]),
            reltol=reltol,
            maxrank=maxbonddim
        )
        tci.Iset[b+1] = Icombined[rowpivots(lu)]
    end

    #TODO
end
