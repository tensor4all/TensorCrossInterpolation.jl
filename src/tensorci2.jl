mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{Vector{MultiIndex}}
    Jset::Vector{Vector{MultiIndex}}
    localset::Vector{Vector{LocalIndex}}

    T::Vector{Array{ValueType,3}}

    "Error estimate for backtruncation of bonds."
    pivoterrors::Vector{Float64}
    "Error estimate per bond from last forward sweep."
    bonderrorsforward::Vector{Float64}
    "Error estimate per bond from last backward sweep."
    bonderrorsbackward::Vector{Float64}
    "Maximum sample for error normalization."
    maxsamplevalue::Float64

    function TensorCI2{ValueType}(
        localdims::Union{Vector{Int},NTuple{N,Int}}
    ) where {ValueType,N}
        n = length(localdims)
        new{ValueType}(
            [Vector{MultiIndex}() for _ in 1:n],    # Iset
            [Vector{MultiIndex}() for _ in 1:n],    # Jset
            [collect(1:d) for d in localdims],      # localset
            [zeros(0, d, 0) for d in localdims],    # T
            [],                                     # pivoterrors
            zeros(length(localdims) - 1),           # bonderrors, forward sweep
            zeros(length(localdims) - 1),           # bonderrors, backward sweep
            0.0                                     # maxsample
        )
    end
end

function TensorCI2{ValueType}(
    func::F,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))]
) where {F,ValueType,N}
    tci = TensorCI2{ValueType}(localdims)
    f(x) = convert(ValueType, func(x))
    addglobalpivots!(tci, f, initialpivots)
    return tci
end

function linkdims(tci::TensorCI2{V})::Vector{Int} where {V}
    return [linkdim(tci, i) for i in 1:length(tci)-1]
end

function linkdim(tci::TensorCI2{V}, i::Int)::Int where {V}
    return length(tci.Iset[i+1])
end

function flushT!(tci::TensorCI2{V}) where {V}
    for i in 1:length(tci)
       tci.T[i] = zeros(0, length(tci.localset[i]), 0)
    end
    nothing
end


@doc raw"""
    function printnestinginfo(tci::TensorCI2{T}) where {T}

Print information about fulfillment of the nesting criterion (``I_\ell < I_{\ell+1}`` and ``J_\ell < J_{\ell+1}``) on each pair of bonds ``\ell, \ell+1`` of `tci` to stdout.
"""
function printnestinginfo(tci::TensorCI2{T}) where {T}
    printnestinginfo(stdout, tci)
end

@doc raw"""
    function printnestinginfo(io::IO, tci::TensorCI2{T}) where {T}

Print information about fulfillment of the nesting criterion (``I_\ell < I_{\ell+1}`` and ``J_\ell < J_{\ell+1}``) on each pair of bonds ``\ell, \ell+1`` of `tci` to `io`.
"""
function printnestinginfo(io::IO, tci::TensorCI2{T}) where {T}
    println(io, "Nesting info: Iset")
    for i in 1:length(tci.Iset)-1
        if isnested(tci.Iset[i], tci.Iset[i+1], :row)
            println(io, "  Nested: $(i) < $(i+1)")
        else
            println(io, "  Not nested: $(i) !< $(i+1)")
        end
    end

    println(io)
    println(io, "Nesting info: Jset")
    for i in 1:length(tci.Jset)-1
        if isnested(tci.Jset[i+1], tci.Jset[i], :col)
            println(io, "  Nested: $(i+1) < $i")
        else
            println(io, "  Not nested: ! $(i+1) < $i")
        end
    end
end

function isfullynested(tci::TensorCI2{T})::Bool where {T}
    for i in 1:length(tci.Iset)-1
        if !isnested(tci.Iset[i], tci.Iset[i+1], :row)
            return false
        end
    end

    for i in 1:length(tci.Jset)-1
        if !isnested(tci.Jset[i+1], tci.Jset[i], :col)
            return false
        end
    end

    return true
end

function updatebonderror!(
    tci::TensorCI2{T}, b::Int, sweepdirection::Symbol, error::Float64
) where {T}
    if sweepdirection == :forward
        tci.bonderrorsforward[b] = error
    elseif sweepdirection == :backward
        tci.bonderrorsbackward[b] = error
    end
    nothing
end

function maxbonderror(tci::TensorCI2{T}) where {T}
    return max(maximum(tci.bonderrorsforward), maximum(tci.bonderrorsbackward))
end

function updatepivoterror!(tci::TensorCI2{T}, errors::AbstractVector{Float64}) where {T}
    erroriter = Iterators.map(max, padzero(tci.pivoterrors), padzero(errors))
    tci.pivoterrors = Iterators.take(
        erroriter,
        max(length(tci.pivoterrors), length(errors))
    ) |> collect
    nothing
end

function flushpivoterror!(tci::TensorCI2{T}) where {T}
    tci.pivoterrors = Float64[]
    nothing
end

function pivoterror(tci::TensorCI2{T}) where {T}
    return maxbonderror(tci)
end

function updateerrors!(
    tci::TensorCI2{T}, b::Int, sweepdirection::Symbol, errors::AbstractVector{Float64}, lastpivoterror::Float64
) where {T}
    updatebonderror!(tci, b, sweepdirection, lastpivoterror)
    updatepivoterror!(tci, errors)
    nothing
end

function addglobalpivots!(
    tci::TensorCI2{ValueType},
    f::F,
    pivots::Vector{MultiIndex};
    reltol=1e-14,
    maxbonddim=typemax(Int)
) where {F,ValueType}
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


# Default slow implementation
function _batchevaluate_dispatch(
    ::Type{ValueType},
    f,
    localset::Vector{Vector{LocalIndex}},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
) where {ValueType}
    N = length(localset)
    nl = length(first(Iset))
    nr = length(first(Jset))
    ncent = N - nl - nr
    return ValueType[f(collect(Iterators.flatten(i))) for i in Iterators.product(Iset, localset[nl+1:nl+ncent]..., Jset)]
end


function _batchevaluate_dispatch(
    ::Type{ValueType},
    f::BatchEvaluator{ValueType},
    localset::Vector{Vector{LocalIndex}},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
) where {ValueType}
    N = length(localset)
    nl = length(first(Iset))
    nr = length(first(Jset))
    ncent = N - nl - nr
    return batchevaluate(f, Iset, Jset, Val(ncent))
end


function _batchevaluate(
    ::Type{ValueType},
    f,
    localset::Vector{Vector{LocalIndex}},
    Iset::Vector{Vector{MultiIndex}},
    Jset::Vector{Vector{MultiIndex}},
    ipos, jpos
) where {ValueType}
    Iset_ = Iset[ipos]
    Jset_ = Jset[jpos]
    N = length(localset)
    nl = ipos - 1
    nr = N - jpos 
    ncent = N - nl - nr
    result = _batchevaluate_dispatch(ValueType, f, localset, Iset[ipos], Jset[jpos])
    expected_size = (length(Iset_), length.(localset[nl+1:nl+ncent])..., length(Jset_))
    size(result) == expected_size || throw(DimensionMismatch("Result has wrong size $(size(result)) != expected $(expected_size)"))
    return result
end


function kronecker(
    Iset::Union{Vector{MultiIndex},IndexSet{MultiIndex}},
    localset::Vector{LocalIndex}
)
    return MultiIndex[[is..., j] for is in Iset, j in localset][:]
end

function kronecker(
    localset::Vector{LocalIndex},
    Jset::Union{Vector{MultiIndex},IndexSet{MultiIndex}}
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


function makefullnesting!(tci::TensorCI2{ValueType}; updateIset=true, updateJset=true) where {ValueType}
    if updateIset
        for n in reverse(2:length(tci))
            for I in tci.Iset[n]
                pushunique!(tci.Iset[n-1], I[1:end-1])
            end
        end
    end
    if updateJset
        for n in 1:length(tci)-1
            for J in tci.Jset[n]
                pushunique!(tci.Jset[n+1], J[2:end])
            end
        end
    end
    nothing
end

function makecanonical!(
    tci::TensorCI2{ValueType},
    f::F;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int)
) where {F,ValueType}
    L = length(tci)
    #flushT!(tci)

    flushpivoterror!(tci)
    for b in 1:L-1
        Icombined = kronecker(tci.Iset[b], tci.localset[b])
        Pi = reshape(_batchevaluate(ValueType, f, tci.localset, tci.Iset, tci.Jset, b, b), length(Icombined), length(tci.Jset[b]))
        updatemaxsample!(tci, Pi)
        ludecomp = rrlu(
            Pi,
            reltol=reltol,
            abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=true
        )
        tci.Iset[b+1] = Icombined[rowindices(ludecomp)]
        updateerrors!(tci, b, :forward, pivoterrors(ludecomp), lastpivoterror(ludecomp))
    end

    flushpivoterror!(tci)
    for b in L:-1:2
        Jcombined = kronecker(tci.localset[b], tci.Jset[b])
        Pi = reshape(_batchevaluate(ValueType, f, tci.localset, tci.Iset, tci.Jset, b , b), length(tci.Iset[b]), length(Jcombined))
        updatemaxsample!(tci, Pi)
        ludecomp = rrlu(
            Pi,
            reltol=reltol,
            abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=false
        )
        tci.Jset[b-1] = Jcombined[colindices(ludecomp)]
        updateerrors!(tci, b - 1, :backward, pivoterrors(ludecomp), lastpivoterror(ludecomp))
    end

    flushpivoterror!(tci)
    for b in 1:L-1
        Icombined = kronecker(tci.Iset[b], tci.localset[b])
        Pi = reshape(_batchevaluate(ValueType, f, tci.localset, tci.Iset, tci.Jset, b, b), length(Icombined), length(tci.Jset[b]))
        updatemaxsample!(tci, Pi)
        luci = MatrixLUCI(
            Pi,
            reltol=reltol,
            abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=true
        )
        tci.Iset[b+1] = Icombined[rowindices(luci)]
        tci.Jset[b] = tci.Jset[b][colindices(luci)]
        updateerrors!(tci, b, :forward, pivoterrors(luci), lastpivoterror(luci))
    end
    localtensor = reshape(
        _batchevaluate(ValueType, f, tci.localset, tci.Iset, tci.Jset, length(tci), length(tci)),
        length(tci.Iset[end]), length(tci.localset[end])
    )
    setT!(tci, L, localtensor)

    #tensortrain!(tci.T, tci, f)
end

function _randompivot(localset)
    return [rand(1:localset[n]) for n in 1:length(localset)]
end

function updatepivots!(
    tci::TensorCI2{ValueType},
    b::Int,
    f::F,
    leftorthogonal::Bool;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int),
    sweepdirection::Symbol=:forward,
    randompivot=false
) where {F,ValueType}
    localdims = [length(s) for s in tci.localset]

    if randompivot
        if leftorthogonal # forward sweep
            nrnd = max(floor(Int, 0.1 * length(tci.Jset[b+1])), 10)
            for _ in 1:nrnd
                pushunique!(tci.Jset[b+1], _randompivot(localdims[b+2:end]))
            end
        else # backward sweep
            nrnd = max(floor(Int, 0.1 * length(tci.Iset[b])), 10)
            for _ in 1:nrnd
                pushunique!(tci.Iset[b], _randompivot(localdims[1:b-1]))
            end
        end
    end

    Icombined = kronecker(tci.Iset[b], tci.localset[b])
    Jcombined = kronecker(tci.localset[b+1], tci.Jset[b+1])

    Pi = reshape(
        _batchevaluate(ValueType, f, tci.localset, tci.Iset, tci.Jset, b, b + 1),
        length(Icombined), length(Jcombined)
    )
    updatemaxsample!(tci, Pi)
    luci = MatrixLUCI(
        Pi,
        reltol=reltol,
        abstol=abstol,
        maxrank=maxbonddim,
        leftorthogonal=leftorthogonal
    )
    tci.Iset[b+1] = Icombined[rowindices(luci)]
    tci.Jset[b] = Jcombined[colindices(luci)]
    setT!(tci, b, left(luci))
    setT!(tci, b + 1, right(luci))
    updateerrors!(tci, b, sweepdirection, pivoterrors(luci), lastpivoterror(luci))

    if randompivot
        if leftorthogonal # forward sweep
            for n in b+1:length(tci)
                for j in tci.Jset[n-1]
                    pushunique!(tci.Jset[n], j[2:end])
                end
            end
        else
            for n in reverse(1:b)
                for i in tci.Iset[n+1]
                pushunique!(tci.Iset[n], i[1:end-1])
                end
            end
        end
    end
end

function convergencecriterion(
    ranks::AbstractVector{Int},
    errors::AbstractVector{Float64},
    tolerance::Float64,
    maxbonddim::Int,
    ncheckhistory::Int,
)::Bool
    if length(errors) < ncheckhistory
        return false
    end
    lastranks = last(ranks, ncheckhistory)
    return (
        all(last(errors, ncheckhistory) .< tolerance) &&
        minimum(lastranks) == lastranks[end]
    ) || all(lastranks .>= maxbonddim)
end

"""
    function optimize!(
        tci::TensorCI2{ValueType},
        f;
        tolerance::Float64=1e-8,
        pivottolerance::Float64=tolerance,
        maxbonddim::Int=typemax(Int),
        maxiter::Int=200,
        sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.backandforth,
        verbosity::Int=0,
        loginterval::Int=10,
        normalizeerror::Bool=true,
        ncheckhistory=3
    ) where {ValueType}

Perform optimization sweeps using the TCI2 algorithm. This will sucessively improve the TCI approximation of a function until it fits `f` with an error smaller than `tolerance`, or until the maximum bond dimension (`maxbonddim`) is reached.

Arguments:
- `tci::TensorCI2{ValueType}`: The TCI to optimize.
- `f`: The function to fit.
- `f` is the function to be interpolated. `f` should have a single parameter, which is a vector of the same length as `localdims`. The return type should be `ValueType`.
- `localdims::Union{Vector{Int},NTuple{N,Int}}` is a `Vector` (or `Tuple`) that contains the local dimension of each index of `f`.
- `initialpivots::Vector{MultiIndex}` is a vector of pivots to be used for initialization. Default: `[1, 1, ...]`.
- `tolerance::Float64` is a float specifying the target tolerance for the interpolation. Default: `1e-8`.
- `maxbonddim::Int` specifies the maximum bond dimension for the TCI. Default: `typemax(Int)`, i.e. effectively unlimited.
- `maxiter::Int` is the maximum number of iterations (i.e. optimization sweeps) before aborting the TCI construction. Default: `200`.
- `sweepstrategy::SweepStrategies.SweepStrategy` specifies whether to sweep forward, backward, or back and forth during optimization. Default: `SweepStrategies.back_and_forth`.
- `verbosity::Int` can be set to `>= 1` to get convergence information on standard output during optimization. Default: `0`.
- `loginterval::Int` can be set to `>= 1` to specify how frequently to print convergence information. Default: `10`.
- `normalizeerror::Bool` determines whether to scale the error by the maximum absolute value of `f` found during sampling. If set to `false`, the algorithm continues until the *absolute* error is below `tolerance`. If set to `true`, the algorithm uses the absolute error divided by the maximum sample instead. This is helpful if the magnitude of the function is not known in advance. Default: `true`.
- `ncheckhistory::Int` is the number of history points to use for convergence checks. Default: `3`.

Notes:
- Set `tolerance` to be > 0 or `maxbonddim` to some reasonable value. Otherwise, convergence is not reachable.
- By default, no caching takes place. Use the [`CachedFunction`](@ref) wrapper if your function is expensive to evaluate.


See also: [`crossinterpolate2`](@ref), [`SweepStrategies`](@ref), [`optfirstpivot`](@ref), [`CachedFunction`](@ref), [`crossinterpolate`](@ref)
"""
function optimize!(
    tci::TensorCI2{ValueType},
    f;
    tolerance::Float64=1e-8,
    pivottolerance::Float64=tolerance,
    maxbonddim::Int=typemax(Int),
    maxiter::Int=200,
    sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.backandforth,
    verbosity::Int=0,
    loginterval::Int=10,
    normalizeerror::Bool=true,
    ncheckhistory=3
) where {ValueType}
    n = length(tci)
    errors = Float64[]
    ranks = Int[]

    if maxbonddim >= typemax(Int) && tolerance <= 0
        throw(ArgumentError(
            "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence criterion is not reachable!"
        ))
    end

    for iter in rank(tci)+1:maxiter
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        flushpivoterror!(tci)
        if forwardsweep(sweepstrategy, iter) # forward sweep
            for bondindex in 1:n-1
                updatepivots!(
                    tci, bondindex, f, true;
                    abstol=pivottolerance * errornormalization, maxbonddim=maxbonddim, sweepdirection=:forward
                )
                #makefullnesting!(tci)
            end
        else # backward sweep
            for bondindex in (n-1):-1:1
                updatepivots!(
                    tci, bondindex, f, false;
                    abstol=pivottolerance * errornormalization, maxbonddim=maxbonddim, sweepdirection=:backward
                )
            end
        end

        push!(errors, pivoterror(tci))
        push!(ranks, rank(tci))
        if verbosity > 0 && mod(iter, loginterval) == 0
            println("iteration = $iter, rank = $(last(ranks)), error= $(last(errors)), maxsamplevalue= $(tci.maxsamplevalue)")
        end
        if convergencecriterion(
            ranks, errors, tolerance * errornormalization, maxbonddim, ncheckhistory
        )
            break
        end
    end

    # Recompute tennsor train tensors using QR
    #tensortrain!(tci.T, tci, f)

    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    return ranks, errors ./ errornormalization
end

@doc raw"""
    function crossinterpolate2(
        ::Type{ValueType},
        f,
        localdims::Union{Vector{Int},NTuple{N,Int}},
        initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
        tolerance::Float64=1e-8,
        pivottolerance::Float64=tolerance,
        maxbonddim::Int=typemax(Int),
        maxiter::Int=200,
        sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.backandforth,
        verbosity::Int=0,
        loginterval::Int=10,
        normalizeerror::Bool=true,
        ncheckhistory=3
    ) where {ValueType,N}

Cross interpolate a function ``f(\mathbf{u})`` using the TCI2 algorithm. Here, the domain of ``f`` is ``\mathbf{u} \in [1, \ldots, d_1] \times [1, \ldots, d_2] \times \ldots \times [1, \ldots, d_{\mathscr{L}}]`` and ``d_1 \ldots d_{\mathscr{L}}`` are the local dimensions.

Arguments:
- `ValueType` is the return type of `f`. Automatic inference is too error-prone.
- `f` is the function to be interpolated. `f` should have a single parameter, which is a vector of the same length as `localdims`. The return type should be `ValueType`.
- `localdims::Union{Vector{Int},NTuple{N,Int}}` is a `Vector` (or `Tuple`) that contains the local dimension of each index of `f`.
- `initialpivots::Vector{MultiIndex}` is a vector of pivots to be used for initialization. Default: `[1, 1, ...]`.
- `tolerance::Float64` is a float specifying the target tolerance for the interpolation. Default: `1e-8`.
- `pivottolerance::Float64` is a float that specifies the tolerance for adding new pivots, i.e. the truncation of tensor train bonds. It should be <= tolerance, otherwise convergence may be impossible. Default: `tolerance`.
- `maxbonddim::Int` specifies the maximum bond dimension for the TCI. Default: `typemax(Int)`, i.e. effectively unlimited.
- `maxiter::Int` is the maximum number of iterations (i.e. optimization sweeps) before aborting the TCI construction. Default: `200`.
- `sweepstrategy::SweepStrategies.SweepStrategy` specifies whether to sweep forward, backward, or back and forth during optimization. Default: `SweepStrategies.back_and_forth`.
- `verbosity::Int` can be set to `>= 1` to get convergence information on standard output during optimization. Default: `0`.
- `loginterval::Int` can be set to `>= 1` to specify how frequently to print convergence information. Default: `10`.
- `normalizeerror::Bool` determines whether to scale the error by the maximum absolute value of `f` found during sampling. If set to `false`, the algorithm continues until the *absolute* error is below `tolerance`. If set to `true`, the algorithm uses the absolute error divided by the maximum sample instead. This is helpful if the magnitude of the function is not known in advance. Default: `true`.
- `ncheckhistory::Int` is the number of history points to use for convergence checks. Default: `3`.

Notes:
- Set `tolerance` to be > 0 or `maxbonddim` to some reasonable value. Otherwise, convergence is not reachable.
- By default, no caching takes place. Use the [`CachedFunction`](@ref) wrapper if your function is expensive to evaluate.


See also: [`optimize!`](@ref), [`SweepStrategies`](@ref), [`optfirstpivot`](@ref), [`CachedFunction`](@ref), [`crossinterpolate`](@ref)
"""
function crossinterpolate2(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
    tolerance::Float64=1e-8,
    pivottolerance::Float64=tolerance,
    maxbonddim::Int=typemax(Int),
    maxiter::Int=200,
    sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.backandforth,
    verbosity::Int=0,
    loginterval::Int=10,
    normalizeerror::Bool=true,
    ncheckhistory=3
) where {ValueType,N}
    tci = TensorCI2{ValueType}(f, localdims, initialpivots)
    ranks, errors = optimize!(
        tci, f;
        tolerance=tolerance, pivottolerance=pivottolerance, maxbonddim=maxbonddim,
        maxiter=maxiter, sweepstrategy=sweepstrategy, verbosity=verbosity,
        loginterval=loginterval, normalizeerror=normalizeerror, ncheckhistory=ncheckhistory
    )
    return tci, ranks, errors
end

@doc raw"""
    function insertglobalpivots!(
        tci::TensorCI2{ValueType}, f;
        nsearch = 100,
        tolerance::Float64=1e-8,
        verbosity::Int=0,
        normalizeerror::Bool=true)::Tuple{TensorCI2{ValueType},Int} where {ValueType}

Insert global pivots where the error exeeds `tolerance` into a TCI2 object. This function is useful if `crossinterpolate2` fails to find pivots in some part of the domain.

Arguments:
- `nsearch` Number of global searches to perform. Default: `100`.

The other paramters are the same as for [`crossinterpolate2`](@ref).
"""
function insertglobalpivots!(
    tci::TensorCI2{ValueType}, f;
    nsearch=100,
    tolerance::Float64=1e-8,
    verbosity::Int=0,
    normalizeerror::Bool=true)::Int where {ValueType}

    localdims = [length(s) for s in tci.localset]

    nnewpivot = 0
    for _ in 1:nsearch
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0

        err(x) = abs(evaluate(tci, x) - f(x))
        newpivot_ = optfirstpivot(
            err, localdims, [rand(1:d) for d in localdims])

        e = err(newpivot_)
        if e > tolerance * errornormalization
            nnewpivot += 1
            addglobalpivots!(tci, f, [newpivot_])
            if verbosity > 0
                println("Inserting a global pivot $(newpivot_): error $e > $(tolerance * errornormalization)")
                println("New linkdims is $(maximum(linkdims(tci))).")
            end
        end
    end

    return nnewpivot
end


function computeT(tci::TensorCI2{V}, f::F, p::Int) where {V, F}
    Iset = tci.Iset[p]
    Jset = tci.Jset[p]
    return reshape(
        _batchevaluate(V, f, tci.localset, tci.Iset, tci.Jset, p, p),
        length(Iset), length(tci.localset[p]), length(Jset)
    )
end


function computeP(tci::TensorCI2{V}, f::F, p::Int) where {V, F}
    Iset = tci.Iset[p+1]
    Jset = tci.Jset[p]
    return reshape(
        _batchevaluate(V, f, tci.localset, tci.Iset, tci.Jset, p+1, p),
        length(Iset), length(Jset)
    )
end


"""
Contruct a tensor train from a function `f` and a TCI object `tci` using QR
"""
function tensortrain!(tensors::Vector{Array{V,3}}, tci::TensorCI2{V}, f::F) where {V, F}
    length(tensors) == length(tci) || error("Expected length(tensors) == length(tci)")
    L = length(tci)
    for n in 1:L-1
        tmat = reshape(
            computeT(tci, f, n),
            length(tci.Iset[n]) * length(tci.localset[n]),
            length(tci.Jset[n])
        )
        tensors[n] = reshape(
            AtimesBinv(tmat, computeP(tci, f, n)),
            length(tci.Iset[n]),
            length(tci.localset[n]),
            length(tci.Iset[n+1])
        )
    end
    tensors[L] = computeT(tci, f, L)
    return nothing
end


