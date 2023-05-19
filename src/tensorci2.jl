mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{Vector{MultiIndex}}
    Jset::Vector{Vector{MultiIndex}}
    localset::Vector{Vector{LocalIndex}}

    T::Vector{Array{ValueType,3}}

    # first half is for forward sweeps, second half is for backward sweeps
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
            fill(0.0, 2*(length(localdims)-1)),    # max truncated pivoterrors
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

function updatepivoterror!(tci::TensorCI2{T}, b::Int, sweepdirection::Symbol, error::Float64) where {T}
    if sweepdirection == :forward
        tci.pivoterrors[b] = error
    elseif sweepdirection == :backward
        tci.pivoterrors[b + length(tci) - 1] = error
    end
    nothing
end

function pivoterror(tci::TensorCI2{T}) where {T}
    return maximum(tci.pivoterrors)
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
            pushunique!(tci.Iset[b], pivot[1:b-1])
            pushunique!(tci.Jset[b], pivot[b+1:end])
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
        updatepivoterror!(tci, b, :forward, estimatedtruncationerror(ludecomp))
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
        updatepivoterror!(tci, b-1, :backward, estimatedtruncationerror(ludecomp))
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
        updatepivoterror!(tci, b, :forward, estimatedtruncationerror(luci))
    end
    setT!(tci, L, getPi(ValueType, f, tci.Iset[end], [[i] for i in tci.localset[end]]))
end

function updatepivots!(
    tci::TensorCI2{ValueType},
    b::Int,
    f::F,
    leftorthogonal::Bool;
    reltol::Float64=1e-14,
    maxbonddim::Int=typemax(Int),
    sweepdirection::Symbol=:forward
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
    updatepivoterror!(tci, b, sweepdirection, estimatedtruncationerror(luci))
end

@doc raw"""
    function crossinterpolate2(
        ::Type{ValueType},
        f,
        localdims::Union{Vector{Int},NTuple{N,Int}},
        initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
        tolerance::Float64=1e-8,
        maxbonddim::Int=typemax(Int),
        maxiter::Int=200,
        sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.backandforth,
        verbosity::Int=0,
        normalizeerror::Bool=true
    ) where {ValueType, N}

Cross interpolate a function ``f(\mathbf{u})`` using the TCI2 algorithm. Here, the domain of ``f`` is ``\mathbf{u} \in [1, \ldots, d_1] \times [1, \ldots, d_2] \times \ldots \times [1, \ldots, d_{\mathscr{L}}]`` and ``d_1 \ldots d_{\mathscr{L}}`` are the local dimensions.

Arguments:
- `ValueType` is the return type of `f`. Automatic inference is too error-prone.
- `f` is the function to be interpolated. `f` should have a single parameter, which is a vector of the same length as `localdims`. The return type should be `ValueType`.
- `localdims::Union{Vector{Int},NTuple{N,Int}}` is a `Vector` (or `Tuple`) that contains the local dimension of each index of `f`.
- `initialpivots::Vector{MultiIndex}` is a vector of pivots to be used for initialization. Default: `[1, 1, ...]`.
- `tolerance::Float64` is a float specifying the target tolerance for the interpolation. Default: `1e-8`.
- `maxbonddim::Int` specifies the maximum bond dimension for the TCI. Default: `typemax(Int)`, i.e. effectively unlimited.
- `maxiter::Int` is the maximum number of iterations (i.e. optimization sweeps) before aborting the TCI construction. Default: `200`.
- `sweepstrategy::SweepStrategies.SweepStrategy` specifies whether to sweep forward, backward, or back and forth during optimization. Default: `SweepStrategies.back_and_forth`.
- `verbosity::Int` can be set to `>= 1` to get convergence information on standard output during optimization. Default: `0`.
- `normalizeerror::Bool` determines whether to scale the error by the maximum absolute value of `f` found during sampling. If set to `false`, the algorithm continues until the *absolute* error is below `tolerance`. If set to `true`, the algorithm uses the absolute error divided by the maximum sample instead. This is helpful if the magnitude of the function is not known in advance. Default: `true`.
- `ncheckhistory::Int` is the number of history points to use for convergence checks. Default: `3`.

Notes:
- Set `tolerance` to be > 0 or `maxbonddim` to some reasonable value. Otherwise, convergence is not reachable.
- By default, no caching takes place. Use the [`CachedFunction`](@ref) wrapper if your function is expensive to evaluate.


See also: [`SweepStrategies`](@ref), [`optfirstpivot`](@ref), [`CachedFunction`](@ref), [`crossinterpolate`](@ref)
"""
function crossinterpolate2(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
    tolerance::Float64=1e-8,
    maxbonddim::Int=typemax(Int),
    maxiter::Int=200,
    sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.backandforth,
    verbosity::Int=0,
    normalizeerror::Bool=true,
    ncheckhistory = 3
) where {ValueType, N}
    tci = TensorCI2{ValueType}(f, localdims, initialpivots)
    n = length(tci)
    errors = Float64[]
    ranks = Int[]

    if maxbonddim >= typemax(Int) && tolerance <= 0
        throw(ArgumentError(
            "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence criterion is not reachable!"))
    end

    for iter in rank(tci)+1:maxiter
        if forwardsweep(sweepstrategy, iter) # forward sweep
            updatepivots!.(tuple(tci), 1:n-1, f, true; reltol=tolerance, maxbonddim=maxbonddim, sweepdirection=:forward)
        else # backward sweep
            updatepivots!.(tuple(tci), (n-1):-1:1, f, false; reltol=tolerance, maxbonddim=maxbonddim, sweepdirection=:backward)
        end

        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        push!(errors, pivoterror(tci))
        push!(ranks, rank(tci))
        if verbosity > 0 && mod(iter, 10) == 0
            println("iteration = $iter, rank = $(last(ranks)), error= $(last(errors))")
        end
        if length(errors) > ncheckhistory
            if minimum(errors[end-ncheckhistory+1:end-1]) >= last(errors) && # last error is smallest
                maximum(errors[end-ncheckhistory+1:end]) < tolerance * errornormalization && # error criterion reached
                length(unique(ranks[end-ncheckhistory+1:end])) == 1 # rank is constant
                break
            end
        end
    end

    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    return tci, ranks, errors ./ errornormalization
end
