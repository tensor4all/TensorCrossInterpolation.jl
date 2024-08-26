
"""
    mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}

Type that represents tensor cross interpolations created using the TCI2 algorithm. Users may want to create these using [`crossinterpolate2`](@ref) rather than calling a constructor directly.
"""
mutable struct TensorCI2{ValueType}
    Iset::Vector{Vector{MultiIndex}}
    Jset::Vector{Vector{MultiIndex}}
    localdims::Vector{Int}

    "Error estimate for backtruncation of bonds."
    pivoterrors::Vector{Float64}
    #"Error estimate per bond by 2site sweep."
    bonderrors::Vector{Float64}
    #"Maximum sample for error normalization."
    maxsamplevalue::Float64

    function TensorCI2{ValueType}(
        localdims::Union{Vector{Int},NTuple{N,Int}}
    ) where {ValueType,N}
        length(localdims) > 1 || error("localdims should have at least 2 elements!")
        n = length(localdims)
        new{ValueType}(
            [Vector{MultiIndex}() for _ in 1:n],    # Iset
            [Vector{MultiIndex}() for _ in 1:n],    # Jset
            collect(localdims),                     # localdims
            [],                                     # pivoterrors
            zeros(length(localdims) - 1),           # bonderrors
            0.0,                                    # maxsamplevalue
        )
    end
end

Base.length(tci::TensorCI2) = length(tci.localdims)

function TensorCI2{ValueType}(
    func::F,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))]
) where {F,ValueType,N}
    tci = TensorCI2{ValueType}(localdims)
    addglobalpivots!(tci, initialpivots)
    tci.maxsamplevalue = maximum(abs, (func(x) for x in initialpivots))
    abs(tci.maxsamplevalue) > 0.0 || error("maxsamplevalue is zero!")
    return tci
end

function linkdims(tci::TensorCI2{T})::Vector{Int} where {T}
    return [length(tci.Iset[b+1]) for b in 1:length(tci)-1]
end

function rank(tci::TensorCI2{T})::Int where {T}
    return maximum(linkdims(tci))
end

function sitedims(tci::TensorCI2{V})::Vector{Vector{Int}} where {V}
    return [[x] for x in tci.localdims]
end

function updatebonderror!(
    tci::TensorCI2{T}, b::Int, error::Float64
) where {T}
    tci.bonderrors[b] = error
    nothing
end

function maxbonderror(tci::TensorCI2{T}) where {T}
    return maximum(tci.bonderrors)
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
    tci::TensorCI2{T},
    b::Int,
    errors::AbstractVector{Float64}
) where {T}
    updatebonderror!(tci, b, last(errors))
    updatepivoterror!(tci, errors)
    nothing
end


"""
Add global pivots to index sets
"""
function addglobalpivots!(
    tci::TensorCI2{ValueType},
    pivots::Vector{MultiIndex}
) where {ValueType}
    if any(length(tci) .!= length.(pivots))
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the MPS."))
    end

    for pivot in pivots
        for b in 1:length(tci)
            pushunique!(tci.Iset[b], pivot[1:b-1])
            pushunique!(tci.Jset[b], pivot[b+1:end])
        end
    end

    nothing
end

"""
Add global pivots to index sets and perform a 2site sweep.
Retry until all pivots are added or until `ntry` iterations are reached.
"""
function addglobalpivots2sitesweep!(
    tci::TensorCI2{ValueType},
    f::F,
    pivots::Vector{MultiIndex};
    tolerance::Float64=1e-8,
    normalizeerror::Bool=true,
    maxbonddim=typemax(Int),
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
) where {F,ValueType}
    if any(length(tci) .!= length.(pivots))
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the MPS."))
    end

    pivots_ = pivots

    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    abstol = tolerance * errornormalization

    addglobalpivots!(tci, pivots_)

    sweep2site!(
        tci, f, 2;
        abstol=abstol,
        maxbonddim=maxbonddim,
        pivotsearch=pivotsearch,
        verbosity=verbosity)

    return nothing
end


"""
Add global pivots to index sets and perform a 1site sweep
"""
#==
function addglobalpivots1sitesweep!(
    tci::TensorCI2{ValueType},
    f::F,
    pivots::Vector{MultiIndex};
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim=typemax(Int)
) where {F,ValueType}
    addglobalpivots!(tci, pivots)
    makecanonical!(tci, f; reltol=reltol, abstol=abstol, maxbonddim=maxbonddim)
end
==#


function existaspivot(
    tci::TensorCI2{ValueType},
    indexset::MultiIndex) where {ValueType}
    return [indexset[1:b-1] ∈ tci.Iset[b] && indexset[b+1:end] ∈ tci.Jset[b] for b in 1:length(tci)]
end


"""
Add global pivots to index sets and perform a 2site sweep.
Retry until all pivots are added or until `ntry` iterations are reached.
"""
#==
function addglobalpivots2sitesweep!(
    tci::TensorCI2{ValueType},
    f::F,
    pivots::Vector{MultiIndex};
    tolerance::Float64=1e-8,
    normalizeerror::Bool=true,
    maxbonddim=typemax(Int),
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    ntry::Int=10,
    strictlynested::Bool=false
)::Int where {F,ValueType}
    if any(length(tci) .!= length.(pivots))
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the MPS."))
    end

    pivots_ = pivots

    for _ in 1:ntry
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        abstol = tolerance * errornormalization

        addglobalpivots!(tci, pivots_)

        sweep2site!(
            tci, f, 2;
            abstol=abstol,
            maxbonddim=maxbonddim,
            pivotsearch=pivotsearch,
            strictlynested=strictlynested,
            verbosity=verbosity)

        newpivots = [p for p in pivots if abs(evaluate(tci, p) - f(p)) > abstol]

        if verbosity > 0
            println("Trying to add $(length(pivots_)) global pivots, $(length(newpivots)) still remain.")
        end

        if length(newpivots) == 0 || Set(newpivots) == Set(pivots_)
            return length(newpivots)
        end

        pivots_ = newpivots
    end
    return length(pivots_)
end
==#

function filltensor(
    ::Type{ValueType},
    f,
    localdims::Vector{Int},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
    ::Val{M}
)::Array{ValueType,M + 2} where {ValueType,M}
    if length(Iset) * length(Jset) == 0
        return Array{ValueType,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    N = length(localdims)
    nl = length(first(Iset))
    nr = length(first(Jset))
    ncent = N - nl - nr
    expected_size = (length(Iset), localdims[nl+1:nl+ncent]..., length(Jset))
    M == ncent || error("Invalid number of central indices")
    return reshape(
        _batchevaluate_dispatch(ValueType, f, localdims, Iset, Jset, Val(ncent)),
        expected_size...
    )
end


function kronecker(
    Iset::Union{Vector{MultiIndex},IndexSet{MultiIndex}},
    localdim::Int
)
    return MultiIndex[[is..., j] for is in Iset, j in 1:localdim][:]
end

function kronecker(
    localdim::Int,
    Jset::Union{Vector{MultiIndex},IndexSet{MultiIndex}}
)
    return MultiIndex[[i, js...] for i in 1:localdim, js in Jset][:]
end


function sweep0site!(
    tci::TensorCI2{ValueType}, f, b::Int;
    reltol=1e-14, abstol=0.0
) where {ValueType}
    P = reshape(
        filltensor(ValueType, f, tci.localdims, tci.Iset[b+1], tci.Jset[b], Val(0)),
        length(tci.Iset[b+1]), length(tci.Jset[b]))
    updatemaxsample!(tci, P)
    F = MatrixLUCI(
        P,
        reltol=reltol,
        abstol=abstol,
        leftorthogonal=true
    )

    ndiag = sum(abs(F.lu.U[i, i]) > abstol && abs(F.lu.U[i, i] / F.lu.U[1, 1]) > reltol for i in eachindex(diag(F.lu.U)))

    tci.Iset[b+1] = tci.Iset[b+1][rowindices(F)[1:ndiag]]
    tci.Jset[b] = tci.Jset[b][colindices(F)[1:ndiag]]
    return nothing
end

# Backward compatibility
const rmbadpivots! = sweep0site!

function updatemaxsample!(tci::TensorCI2{V}, samples::Array{V}) where {V}
    tci.maxsamplevalue = maxabs(tci.maxsamplevalue, samples)
end

function sweep1site!(
    tci::TensorCI2{ValueType},
    f,
    sweepdirection::Symbol=:forward;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int)
) where {ValueType}
    flushpivoterror!(tci)

    if !(sweepdirection === :forward || sweepdirection === :backward)
        throw(ArgumentError("Unknown sweep direction $sweepdirection: choose between :forward, :backward."))
    end

    forwardsweep = sweepdirection === :forward
    for b in (forwardsweep ? (1:length(tci)-1) : (length(tci):-1:2))
        Is = forwardsweep ? kronecker(tci.Iset[b], tci.localdims[b]) : tci.Iset[b]
        Js = forwardsweep ? tci.Jset[b] : kronecker(tci.localdims[b], tci.Jset[b])
        Pi = reshape(
            filltensor(ValueType, f, tci.localdims, tci.Iset[b], tci.Jset[b], Val(1)),
            length(Is), length(Js))
        updatemaxsample!(tci, Pi)
        luci = MatrixLUCI(
            Pi,
            reltol=reltol,
            abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=forwardsweep
        )
        tci.Iset[b+forwardsweep] = Is[rowindices(luci)]
        tci.Jset[b-!forwardsweep] = Js[colindices(luci)]
        updateerrors!(
            tci, b - !forwardsweep,
            pivoterrors(luci),
        )
    end

    nothing
end


mutable struct SubMatrix{T}
    f::Function
    rows::Vector{MultiIndex}
    cols::Vector{MultiIndex}
    maxsamplevalue::Float64

    function SubMatrix{T}(f, rows, cols) where {T}
        new(f, rows, cols, 0.0)
    end
end

function _submatrix_batcheval(obj::SubMatrix{T}, f, irows::Vector{Int}, icols::Vector{Int})::Matrix{T} where {T}
    return [f(vcat(obj.rows[i], obj.cols[j])) for i in irows, j in icols]
end


function _submatrix_batcheval(obj::SubMatrix{T}, f::BatchEvaluator{T}, irows::Vector{Int}, icols::Vector{Int})::Matrix{T} where {T}
    Iset = [obj.rows[i] for i in irows]
    Jset = [obj.cols[j] for j in icols]
    return f(Iset, Jset, Val(0))
end


function (obj::SubMatrix{T})(irows::Vector{Int}, icols::Vector{Int})::Matrix{T} where {T}
    res = _submatrix_batcheval(obj, obj.f, irows, icols)
    obj.maxsamplevalue = max(obj.maxsamplevalue, maximum(abs, res))
    return res
end


"""
Update pivots at bond `b` of `tci` using the TCI2 algorithm.
"""
function updatepivots!(
    tci::TensorCI2{ValueType},
    b::Int,
    f::F,
    leftorthogonal::Bool;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int),
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    extraIset::Vector{MultiIndex}=MultiIndex[],
    extraJset::Vector{MultiIndex}=MultiIndex[],
) where {F,ValueType}
    Icombined = union(kronecker(tci.Iset[b], tci.localdims[b]), extraIset)
    Jcombined = union(kronecker(tci.localdims[b+1], tci.Jset[b+1]), extraJset)

    luci = if pivotsearch === :full
        t1 = time_ns()
        Pi = reshape(
            filltensor(ValueType, f, tci.localdims,
                Icombined, Jcombined, Val(0)),
            length(Icombined), length(Jcombined)
        )
        t2 = time_ns()

        updatemaxsample!(tci, Pi)
        luci = MatrixLUCI(
            Pi,
            reltol=reltol,
            abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=leftorthogonal
        )
        t3 = time_ns()
        if verbosity > 2
            x, y = length(Icombined), length(Jcombined)
            println("    Computing Pi ($x x $y) at bond $b: $(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec")
        end
        luci
    elseif pivotsearch === :rook
        t1 = time_ns()
        I0 = Int.(Iterators.filter(!isnothing, findfirst(isequal(i), Icombined) for i in tci.Iset[b+1]))::Vector{Int}
        J0 = Int.(Iterators.filter(!isnothing, findfirst(isequal(j), Jcombined) for j in tci.Jset[b]))::Vector{Int}
        Pif = SubMatrix{ValueType}(f, Icombined, Jcombined)
        t2 = time_ns()
        res = MatrixLUCI(
            ValueType,
            Pif,
            (length(Icombined), length(Jcombined)),
            I0, J0;
            reltol=reltol, abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=leftorthogonal,
            pivotsearch=:rook,
            usebatcheval=true
        )
        updatemaxsample!(tci, [ValueType(Pif.maxsamplevalue)])

        t3 = time_ns()

        # Fall back to full search if rook search fails
        if npivots(res) == 0
            Pi = reshape(
                filltensor(ValueType, f, tci.localdims,
                    Icombined, Jcombined, Val(0)),
                length(Icombined), length(Jcombined)
            )
            updatemaxsample!(tci, Pi)
            res = MatrixLUCI(
                Pi,
                reltol=reltol,
                abstol=abstol,
                maxrank=maxbonddim,
                leftorthogonal=leftorthogonal
            )
        end

        t4 = time_ns()
        if verbosity > 2
            x, y = length(Icombined), length(Jcombined)
            println("    Computing Pi ($x x $y) at bond $b: $(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec, fall back to full: $(1e-9*(t4-t3)) sec")
        end
        res
    else
        throw(ArgumentError("Unknown pivot search strategy $pivotsearch. Choose from :rook, :full."))
    end
    tci.Iset[b+1] = Icombined[rowindices(luci)]
    tci.Jset[b] = Jcombined[colindices(luci)]
    updateerrors!(tci, b, pivoterrors(luci))
    nothing
end

function convergencecriterion(
    ranks::AbstractVector{Int},
    errors::AbstractVector{Float64},
    nglobalpivots::AbstractVector{Int},
    tolerance::Float64,
    maxbonddim::Int,
    ncheckhistory::Int,
)::Bool
    if length(errors) < ncheckhistory
        return false
    end
    lastranks = last(ranks, ncheckhistory)
    lastngpivots = last(nglobalpivots, ncheckhistory)
    return (
        all(last(errors, ncheckhistory) .< tolerance) &&
        all(lastngpivots .== 0) &&
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
        sweepstrategy::Symbol=:backandforth,
        pivotsearch::Symbol=:full,
        verbosity::Int=0,
        loginterval::Int=10,
        normalizeerror::Bool=true,
        ncheckhistory=3,
        maxnglobalpivot::Int=5,
        nsearchglobalpivot::Int=5,
        tolmarginglobalsearch::Float64=10.0,
    ) where {ValueType}

Perform optimization sweeps using the TCI2 algorithm. This will sucessively improve the TCI approximation of a function until it fits `f` with an error smaller than `tolerance`, or until the maximum bond dimension (`maxbonddim`) is reached.

Arguments:
- `tci::TensorCI2{ValueType}`: The TCI to optimize.
- `f`: The function to fit.
- `f` is the function to be interpolated. `f` should have a single parameter, which is a vector of the same length as `localdims`. The return type should be `ValueType`.
- `localdims::Union{Vector{Int},NTuple{N,Int}}` is a `Vector` (or `Tuple`) that contains the local dimension of each index of `f`.
- `tolerance::Float64` is a float specifying the target tolerance for the interpolation. Default: `1e-8`.
- `maxbonddim::Int` specifies the maximum bond dimension for the TCI. Default: `typemax(Int)`, i.e. effectively unlimited.
- `maxiter::Int` is the maximum number of iterations (i.e. optimization sweeps) before aborting the TCI construction. Default: `200`.
- `sweepstrategy::Symbol` specifies whether to sweep forward (:forward), backward (:backward), or back and forth (:backandforth) during optimization. Default: `:backandforth`.
- `pivotsearch::Symbol` determins how pivots are searched (`:full` or `:rook`). Default: `:full`.
- `verbosity::Int` can be set to `>= 1` to get convergence information on standard output during optimization. Default: `0`.
- `loginterval::Int` can be set to `>= 1` to specify how frequently to print convergence information. Default: `10`.
- `normalizeerror::Bool` determines whether to scale the error by the maximum absolute value of `f` found during sampling. If set to `false`, the algorithm continues until the *absolute* error is below `tolerance`. If set to `true`, the algorithm uses the absolute error divided by the maximum sample instead. This is helpful if the magnitude of the function is not known in advance. Default: `true`.
- `ncheckhistory::Int` is the number of history points to use for convergence checks. Default: `3`.
- `maxnglobalpivot::Int` can be set to `>= 0`. Default: `5`.
- `nsearchglobalpivot::Int` can be set to `>= 0`. Default: `5`.
- `tolmarginglobalsearch` can be set to `>= 1.0`. Seach global pivots where the interpolation error is larger than the tolerance by `tolmarginglobalsearch`.  Default: `10.0`.
- `checkbatchevaluatable::Bool` Check if the function `f` is batch evaluatable. Default: `false`.

Notes:
- Set `tolerance` to be > 0 or `maxbonddim` to some reasonable value. Otherwise, convergence is not reachable.
- By default, no caching takes place. Use the [`CachedFunction`](@ref) wrapper if your function is expensive to evaluate.


See also: [`crossinterpolate2`](@ref), [`optfirstpivot`](@ref), [`CachedFunction`](@ref), [`crossinterpolate1`](@ref)
"""
function optimize!(
    tci::TensorCI2{ValueType},
    f;
    tolerance::Float64=1e-8,
    pivottolerance::Float64=tolerance,
    maxbonddim::Int=typemax(Int),
    maxiter::Int=20,
    sweepstrategy::Symbol=:backandforth,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    loginterval::Int=10,
    normalizeerror::Bool=true,
    ncheckhistory::Int=3,
    maxnglobalpivot::Int=5,
    nsearchglobalpivot::Int=5,
    tolmarginglobalsearch::Float64=10.0,
    checkbatchevaluatable::Bool=false
) where {ValueType}
    errors = Float64[]
    ranks = Int[]
    nglobalpivots = Int[]

    if checkbatchevaluatable && !(f isa BatchEvaluator)
        error("Function `f` is not batch evaluatable")
    end

    #if maxnglobalpivot > 0 && nsearchglobalpivot > 0
    #end
    if nsearchglobalpivot > 0 && nsearchglobalpivot < maxnglobalpivot
        error("nsearchglobalpivot < maxnglobalpivot!")
    end

    tstart = time_ns()

    if maxbonddim >= typemax(Int) && tolerance <= 0
        throw(ArgumentError(
            "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence criterion is not reachable!"
        ))
    end

    globalpivots = MultiIndex[]
    for iter in 1:maxiter
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        abstol = pivottolerance * errornormalization

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: starting 2site sweep")
            flush(stdout)
        end

        sweep2site!(
            tci, f, 2;
            iter1=1,
            abstol=abstol,
            maxbonddim=maxbonddim,
            pivotsearch=pivotsearch,
            verbosity=verbosity,
            sweepstrategy=sweepstrategy,
            fillsitetensors=true
        )
        if verbosity > 0 && length(globalpivots) > 0 && mod(iter, loginterval) == 0
            abserr = [abs(evaluate(tci, p) - f(p)) for p in globalpivots]
            nrejections = length(abserr .> abstol)
            if nrejections > 0
                println("  Rejected $(nrejections) global pivots added in the previous iteration, errors are $(abserr)")
                flush(stdout)
            end
        end
        push!(errors, pivoterror(tci))

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: start searching global pivots")
            flush(stdout)
        end

        # Find global pivots where the error is too large
        # Such gloval pivots are added to the TCI, invalidating site tensors.
        globalpivots = searchglobalpivots(
            tci, f, tolmarginglobalsearch * abstol,
            verbosity=verbosity,
            maxnglobalpivot=maxnglobalpivot,
            nsearch=nsearchglobalpivot
        )
        addglobalpivots!(tci, globalpivots)
        push!(nglobalpivots, length(globalpivots))

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: done searching global pivots")
            flush(stdout)
        end

        push!(ranks, rank(tci))
        if verbosity > 0 && mod(iter, loginterval) == 0
            println("iteration = $iter, rank = $(last(ranks)), error= $(last(errors)), maxsamplevalue= $(tci.maxsamplevalue), nglobalpivot=$(length(globalpivots))")
            flush(stdout)
        end
        if convergencecriterion(
            ranks, errors, nglobalpivots, abstol, maxbonddim, ncheckhistory
        )
            break
        end
    end

    # Extra one sweep by the 1-site update to
    #  (1) Remove unnecessary pivots added by global pivots
    #      Note: a pivot matrix can be non-square after adding global pivots,
    #            or the bond dimension exceeds maxbonddim
    #  (2) Compute site tensors
    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    abstol = pivottolerance * errornormalization
    sweep1site!(
        tci,
        f,
        abstol=abstol,
        maxbonddim=maxbonddim,
    )

    _sanitycheck(tci)

    return ranks, errors ./ errornormalization
end

"""
Perform 2site sweeps on a TCI2.
"""
function sweep2site!(
    tci::TensorCI2{ValueType}, f, niter::Int;
    iter1::Int=1,
    abstol::Float64=1e-8,
    maxbonddim::Int=typemax(Int),
    sweepstrategy::Symbol=:backandforth,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    fillsitetensors::Bool=true
) where {ValueType}
    n = length(tci)

    for iter in iter1:iter1+niter-1

        extraIset = [MultiIndex[] for _ in 1:n]
        extraJset = [MultiIndex[] for _ in 1:n]
        extraIset = deepcopy(tci.Iset)
        extraJset = deepcopy(tci.Jset)

        flushpivoterror!(tci)
        if forwardsweep(sweepstrategy, iter) # forward sweep
            for bondindex in 1:n-1
                updatepivots!(
                    tci, bondindex, f, true;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    extraIset=extraIset[bondindex+1],
                    extraJset=extraJset[bondindex],
                )
            end
        else # backward sweep
            for bondindex in (n-1):-1:1
                updatepivots!(
                    tci, bondindex, f, false;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    extraIset=extraIset[bondindex+1],
                    extraJset=extraJset[bondindex],
                )
            end
        end
    end

    nothing
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
        sweepstrategy::Symbol=:backandforth,
        pivotsearch::Symbol=:full,
        verbosity::Int=0,
        loginterval::Int=10,
        normalizeerror::Bool=true,
        ncheckhistory=3,
        maxnglobalpivot::Int=5,
        nsearchglobalpivot::Int=5,
        tolmarginglobalsearch::Float64=10.0,
        strictlynested::Bool=false
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
- `sweepstrategy::Symbol` specifies whether to sweep forward (:forward), backward (:backward), or back and forth (:backandforth) during optimization. Default: `:backandforth`.
- `pivotsearch::Symbol` determins how pivots are searched (`:full` or `:rook`). Default: `:full`.
- `verbosity::Int` can be set to `>= 1` to get convergence information on standard output during optimization. Default: `0`.
- `loginterval::Int` can be set to `>= 1` to specify how frequently to print convergence information. Default: `10`.
- `normalizeerror::Bool` determines whether to scale the error by the maximum absolute value of `f` found during sampling. If set to `false`, the algorithm continues until the *absolute* error is below `tolerance`. If set to `true`, the algorithm uses the absolute error divided by the maximum sample instead. This is helpful if the magnitude of the function is not known in advance. Default: `true`.
- `ncheckhistory::Int` is the number of history points to use for convergence checks. Default: `3`.
- `maxnglobalpivot::Int` can be set to `>= 0`. Default: `5`.
- `nsearchglobalpivot::Int` can be set to `>= 0`. Default: `5`.
- `tolmarginglobalsearch` can be set to `>= 1.0`. Seach global pivots where the interpolation error is larger than the tolerance by `tolmarginglobalsearch`.  Default: `10.0`.
- `strictlynested::Bool=false` determines whether to preserve partial nesting in the TCI algorithm. Default: `true`.
- `checkbatchevaluatable::Bool` Check if the function `f` is batch evaluatable. Default: `false`.

Notes:
- Set `tolerance` to be > 0 or `maxbonddim` to some reasonable value. Otherwise, convergence is not reachable.
- By default, no caching takes place. Use the [`CachedFunction`](@ref) wrapper if your function is expensive to evaluate.


See also: [`optimize!`](@ref), [`optfirstpivot`](@ref), [`CachedFunction`](@ref), [`crossinterpolate1`](@ref)
"""
function crossinterpolate2(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
    kwargs...
) where {ValueType,N}
    tci = TensorCI2{ValueType}(f, localdims, initialpivots)
    ranks, errors = optimize!(tci, f; kwargs...)
    return tci, ranks, errors
end


"""
Search global pivots where the interpolation error exceeds `abstol`.
"""
function searchglobalpivots(
    tci::TensorCI2{ValueType}, f, abstol;
    verbosity::Int=0,
    nsearch::Int=100,
    maxnglobalpivot::Int=5
)::Vector{MultiIndex} where {ValueType}
    if nsearch == 0 || maxnglobalpivot == 0
        return MultiIndex[]
    end

    pivots = Dict{Float64,MultiIndex}()
    tt = TensorTrain(sitetensors2(tci, f, abstol)[1])
    ttcache = TTCache(tt)
    for _ in 1:nsearch
        pivot, error = _floatingzone(ttcache, f; earlystoptol=10 * abstol, nsweeps=100)
        if error > abstol
            pivots[error] = pivot
        end
        if length(pivots) == maxnglobalpivot
            break
        end
    end

    if length(pivots) == 0
        if verbosity > 1
            println("  No global pivot found")
        end
        return MultiIndex[]
    end

    if verbosity > 1
        maxerr = maximum(keys(pivots))
        println("  Found $(length(pivots)) global pivots: max error $(maxerr)")
    end

    return [p for (_, p) in pivots]
end


function _globalpivots(Isets, Jsets; onlydiagonal=true)::Vector{MultiIndex}
    L = length(Isets)
    p = Set{MultiIndex}()
    # Pivot matrices
    for bondindex in 1:L-1
        if onlydiagonal
            for (x, y) in zip(Isets[bondindex+1], Jsets[bondindex])
                push!(p, vcat(x, y))
            end
        else
            for x in Isets[bondindex+1], y in Jsets[bondindex]
                push!(p, vcat(x, y))
            end
        end
    end
    return collect(p)
end

function _Iset(globalpivots, i::Int)::Vector{MultiIndex}
    return collect(Set(p[1:i-1] for p in globalpivots))
end

function _Jset(globalpivots, i::Int)::Vector{MultiIndex}
    return collect(Set(p[i+1:end] for p in globalpivots))
end

"""
Experimental implmentation of constructing a bettter interpolation from local pivots.

First, we transform all local pivots to global pivots.
Then, we construct new local pivots from the global pivots, and remove redundant local pivots.
"""
function sitetensors(
    tci::TensorCI2{ValueType}, f;
    reltol=1e-14,
    abstol=0.0,
    verbosity=0,
    onlydiagonal=true
) where {ValueType}
    # Transform all local pivots to global pivots
    globalpivots = _globalpivots(tci.Iset, tci.Jset; onlydiagonal)
    if verbosity > 1
        println("  Generated $(length(globalpivots)) global pivots")
    end

    # Remove redundant local pivots by a site0 sweep
    Iset_ = [_Iset(globalpivots, i) for i in 1:length(tci)]
    Jset_ = [_Jset(globalpivots, i) for i in 1:length(tci)]
    for b in 1:length(tci)-1
        P = reshape(
            filltensor(ValueType, f, tci.localdims, Iset_[b+1], Jset_[b], Val(0)),
            length(Iset_[b+1]), length(Jset_[b]))

        F = MatrixLUCI(
            P,
            reltol=reltol,
            abstol=abstol,
            leftorthogonal=true
        )

        ndiag = sum(abs(F.lu.U[i, i]) > abstol && abs(F.lu.U[i, i] / F.lu.U[1, 1]) > reltol for i in eachindex(diag(F.lu.U)))
        if verbosity > 0
            println("  Site0 sweep at bond $b, size=($(length(Iset_[b+1])), $(length(Jset_[b]))), leaving $(ndiag) pivots")
        end

        Iset_[b+1] = Iset_[b+1][rowindices(F)[1:ndiag]]
        Jset_[b] = Jset_[b][colindices(F)[1:ndiag]]
    end

    # Compute a TCI: A_1 A_2 , ..., A_{L-1} T_L
    tensors = Array{ValueType,3}[]
    for l in 1:length(tci)-1
        Iset_l = Iset_[l]
        Jset_l = Jset_[l]
        Iset_lp1 = Iset_[l+1]
        T = reshape(
            filltensor(ValueType, f, tci.localdims, Iset_l, Jset_l, Val(1)),
            length(Iset_l) * tci.localdims[l], length(Jset_l))

        P = reshape(
            filltensor(ValueType, f, tci.localdims, Iset_lp1, Jset_l, Val(0)),
            length(Iset_lp1), length(Jset_l))

        # T P^{-1}
        Tmat = transpose(transpose(P) \ transpose(T))
        push!(tensors, reshape(Tmat, length(Iset_l), tci.localdims[l], length(Iset_lp1)))
    end

    push!(
        tensors,
        filltensor(ValueType, f, tci.localdims, Iset_[end], Jset_[end], Val(1))
    )

    return tensors, Iset_, Jset_
end


function sitetensors2(
    tci::TensorCI2{ValueType}, f,
    tolerance # desired tolerance for interpolation
    ;
    reltol=1e-14, # relative tolerance for site0 sweep
    abstol=0.0, # absolute tolerance for site0 sweep
    verbosity=0,
) where {ValueType}
    Iset_ext = deepcopy(tci.Iset)
    Jset_ext = deepcopy(tci.Jset)

    Pref = Matrix{ValueType}[]
    for b in 1:length(tci)-1
        ref = reshape(
            filltensor(ValueType, f, tci.localdims, tci.Iset[b+1], tci.Jset[b], Val(0)),
            length(tci.Iset[b+1]), length(tci.Jset[b]))
        push!(Pref, ref)
    end

    tensors, _, _ = __sitetensors_site0sweep(ValueType, tci.localdims, Iset_ext, Jset_ext, f; reltol, abstol, verbosity)
    tt = TensorTrain(tensors)
    ttc = TTCache(tt)

    while true
        # Check interpolation erorr on pivot matrices
        globalpivots = Set{MultiIndex}()
        for b in 1:length(tci)-1
            reconst = reshape(
                batchevaluate(ttc, tci.Iset[b+1], tci.Jset[b], Val(0)), length(tci.Iset[b+1]), length(tci.Jset[b])
                )
            for (i_, i) in enumerate(tci.Iset[b+1]), (j_, j) in enumerate(tci.Jset[b])
                diff = abs(reconst[i_, j_] - Pref[b][i_, j_])
                if diff > tolerance
                    push!(globalpivots, vcat(i, j))
                end
            end
        end

        if verbosity > 0
            println("# of global pivots: $(length(globalpivots))")
        end

        if length(globalpivots) == 0
            break
        end

        for p in globalpivots
            for b in 1:length(tci)
                pushunique!(Iset_ext[b], p[1:b-1])
                pushunique!(Jset_ext[b], p[b+1:end])
            end
        end

        tensors, _, _ = __sitetensors_site0sweep(ValueType, tci.localdims, Iset_ext, Jset_ext, f; reltol, abstol, verbosity)
        tt = TensorTrain(tensors)
        ttc = TTCache(tt)
    end

    return tensors, Iset_ext, Jset_ext
end


function __sitetensors_site0sweep(
    ::Type{ValueType},
    localdims,
    Iset::Vector{Vector{MultiIndex}},
    Jset::Vector{Vector{MultiIndex}},
    f;
    reltol=1e-14,
    abstol=0.0,
    verbosity=0,
) where {ValueType}
    Iset_ = deepcopy(Iset)
    Jset_ = deepcopy(Jset)

    # First, we need to make pivot matrices square.
    # This is due to the limitation of the "solve" function with rrLU: non-square matrices are not yet supported.
    for b in 1:length(Iset)-1
        P = reshape(
            filltensor(ValueType, f, localdims, Iset_[b+1], Jset_[b], Val(0)),
            length(Iset_[b+1]), length(Jset_[b]))

        F = MatrixLUCI(
            P,
            reltol=reltol,
            abstol=abstol,
            leftorthogonal=true
        )

        ndiag = sum(abs(F.lu.U[i, i]) > abstol && abs(F.lu.U[i, i] / F.lu.U[1, 1]) > reltol for i in eachindex(diag(F.lu.U)))
        if verbosity > 0
            println("  Site0 sweep at bond $b, size=($(length(Iset_[b+1])), $(length(Jset_[b]))), leaving $(ndiag) pivots")
        end

        Iset_[b+1] = Iset_[b+1][rowindices(F)[1:ndiag]]
        Jset_[b] = Jset_[b][colindices(F)[1:ndiag]]
    end

    # Compute a TCI: A_1 A_2 , ..., A_{L-1} T_L
    tensors = Array{ValueType,3}[]
    for l in 1:length(Iset)-1
        Iset_l = Iset_[l]
        Jset_l = Jset_[l]
        Iset_lp1 = Iset_[l+1]
        T = reshape(
            filltensor(ValueType, f, localdims, Iset_l, Jset_l, Val(1)),
            length(Iset_l) * localdims[l], length(Jset_l))

        P = reshape(
            filltensor(ValueType, f, localdims, Iset_lp1, Jset_l, Val(0)),
            length(Iset_lp1), length(Jset_l))

        # T P^{-1}
        Tmat = transpose(transpose(P) \ transpose(T))
        #Tmat = transpose(transpose(rrlu(P)) \ transpose(T)) # Non-square matrices are not supported yet.
        push!(tensors, reshape(Tmat, length(Iset_l), localdims[l], length(Iset_lp1)))
    end

    push!(
        tensors,
        filltensor(ValueType, f, localdims, Iset_[end], Jset_[end], Val(1))
    )

    return tensors, Iset_, Jset_
end