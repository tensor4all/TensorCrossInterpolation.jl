mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{Vector{MultiIndex}}
    Jset::Vector{Vector{MultiIndex}}
    localset::Vector{Vector{LocalIndex}}

    T::Vector{Array{ValueType,3}}

    pivoterrors::Vector{Float64}
    maxsamplevalue::Float64

    function TensorCI2{ValueType}(
        localdims::Union{Vector{Int},NTuple{N,Int}}
    ) where {ValueType,N}
        n = length(localdims)
        new{ValueType}(
            [Vector{MultiIndex}() for _ in 1:n],  # Iset
            [Vector{MultiIndex}() for _ in 1:n],  # Jset
            [collect(1:d) for d in localdims],      # localset
            [zeros(0, d, 0) for d in localdims],    # T
            [],                                     # pivoterrors
            0.0                                     # maxsample
        )
    end
end

function TensorCI2{ValueType}(
    func::F,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex} = [ones(Int, length(localdims))]
) where{F, ValueType,N}
    tci = TensorCI2{ValueType}(localdims)
    f(x) = convert(ValueType, func(x))
    addglobalpivots!(tci, f, initialpivots)
    return tci
end

function updatepivoterror!(tci::TensorCI2{T}, errors::AbstractVector{Float64}) where {T}
    nerrors = length(tci.pivoterrors)
    tci.pivoterrors = max.(tci.pivoterrors, abs.(errors[1:nerrors]))
    append!(tci.pivoterrors, abs.(errors[nerrors+1:end]))
end

function pivoterror(tci::TensorCI2{T}) where {T}
    return last(tci.pivoterrors)
end

function addglobalpivots!(
    tci::TensorCI2{ValueType},
    f::F,
    pivots::Vector{MultiIndex};
    reltol=1e-14,
    maxbonddim=typemax(Int)
) where {F, ValueType}
    if any(length(tci) .!= length.(pivots))
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the MPS."))
    end

    for pivot in pivots
        for b in 1:length(tci)
            push!(tci.Iset[b], pivot[1:b])
            push!(tci.Jset[b], pivot[b+1:end])
        end
    end

    makecanonical!(tci, f; reltol=reltol, maxbonddim=maxbonddim)
end

function getPi(
    ::Type{ValueType},
    f,
    Iset::Union{Vector{MultiIndex}, IndexSet{MultiIndex}},
    Jset::Union{Vector{MultiIndex}, IndexSet{MultiIndex}}
) where {ValueType}
    return ValueType[f(vcat(i, j)) for i in Iset, j in Jset]
end

function kronecker(
    Iset::Union{Vector{MultiIndex}, IndexSet{MultiIndex}},
    localset::Vector{LocalIndex}
)
    return MultiIndex[[is..., j] for is in Iset, j in localset][:]
end

function kronecker(
    localset::Vector{LocalIndex},
    Jset::Union{Vector{MultiIndex}, IndexSet{MultiIndex}}
)
    return MultiIndex[[i, js...] for i in localset, js in Jset][:]
end

function setT!(
    tci::TensorCI2{ValueType}, b::Int, T::AbstractMatrix{ValueType}
) where {ValueType}
    tci.T[b] = reshape(
        T,
        length(tci.Iset[b]),
        length(tci.localset[b]),
        length(tci.Jset[b])
    )
end

function updatemaxsample!(tci::TensorCI2{V}, samples::Array{V}) where {V}
    tci.maxsamplevalue = maxabs(tci.maxsamplevalue, samples)
end

function makecanonical!(
    tci::TensorCI2{ValueType},
    f::F;
    reltol::Float64=1e-14,
    maxbonddim::Int=typemax(Int)
) where {F, ValueType}
    L = length(tci)

    for b in 1:L-1
        Icombined = kronecker(tci.Iset[b], tci.localset[b])
        Pi = getPi(ValueType, f, Icombined, tci.Jset[b])
        updatemaxsample!(tci, Pi)
        ludecomp = rrlu(
            Pi,
            reltol=reltol,
            maxrank=maxbonddim,
            leftorthogonal=true
        )
        tci.Iset[b+1] = Icombined[rowindices(ludecomp)]
        updatepivoterror!(tci, pivoterrors(ludecomp))
    end

    for b in L:-1:2
        Jcombined = kronecker(tci.localset[b], tci.Jset[b])
        Pi = getPi(ValueType, f, tci.Iset[b], Jcombined)
        updatemaxsample!(tci, Pi)
        ludecomp = rrlu(
            Pi,
            reltol=reltol,
            maxrank=maxbonddim,
            leftorthogonal=false
        )
        tci.Jset[b-1] = Jcombined[colindices(ludecomp)]
        updatepivoterror!(tci, pivoterrors(ludecomp))
    end

    for b in 1:L-1
        Icombined = kronecker(tci.Iset[b], tci.localset[b])
        Pi = getPi(ValueType, f, Icombined, tci.Jset[b])
        updatemaxsample!(tci, Pi)
        luci = MatrixLUCI(
            Pi,
            reltol=reltol,
            maxrank=maxbonddim,
            leftorthogonal=true
        )
        tci.Iset[b+1] = Icombined[rowindices(luci)]
        tci.Jset[b] = tci.Jset[b][colindices(luci)]
        tci.T[b] = setT!(tci, b, left(luci))
        updatepivoterror!(tci, pivoterrors(luci))
    end
    setT!(tci, L, getPi(ValueType, f, tci.Iset[end], [[i] for i in tci.localset[end]]))
end

function updatepivots!(
    tci::TensorCI2{ValueType},
    b::Int,
    f::F,
    leftorthogonal::Bool;
    reltol::Float64=1e-14,
    maxbonddim::Int=typemax(Int)
) where {F, ValueType}
    Icombined = kronecker(tci.Iset[b], tci.localset[b])
    Jcombined = kronecker(tci.localset[b+1], tci.Jset[b+1])
    Pi = getPi(ValueType, f, Icombined, Jcombined)
    updatemaxsample!(tci, Pi)
    luci = MatrixLUCI(
        Pi,
        reltol=reltol,
        maxrank=maxbonddim,
        leftorthogonal=leftorthogonal
    )
    tci.Iset[b+1] = Icombined[rowindices(luci)]
    tci.Jset[b] = Jcombined[colindices(luci)]
    setT!(tci, b, left(luci))
    setT!(tci, b+1, right(luci))
    updatepivoterror!(tci, pivoterror(luci))
end

function crossinterpolate2(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
    tolerance::Float64=1e-8,
    maxiter::Int=200,
    sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.backandforth,
    pivottolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    verbosity::Int=0,
    normalizeerror::Bool=true
) where {ValueType, N}
    tci = TensorCI2{ValueType}(f, localdims, initialpivots)
    n = length(tci)
    errors = Float64[]
    ranks = Int[]

    for iter in rank(tci)+1:maxiter
        if forwardsweep(sweepstrategy, iter)
            updatepivots!.(tci, 1:n-1, f, true; reltol=tolerance, maxbonddim=maxbonddim)
        else
            updatepivots!.(tci, (n-1):-1:1, f, true; reltol=tolerance, maxbonddim=maxbonddim)
        end

        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        push!(errors, pivoterror(tci))
        push!(ranks, rank(tci))
        if verbosity > 0 && mod(iter, 10) == 0
            println(
                "iteration = $iter, rank = $(last(ranks))), error= $(last(errors))")
        end
        if last(errors) < tolerance * errornormalization
            break
        end
    end

    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    return tci, ranks, errors ./ errornormalization
end
