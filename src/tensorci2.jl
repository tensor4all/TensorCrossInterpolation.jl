
"""
    mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}

Type that represents tensor cross interpolations created using the TCI2 algorithm. Users may want to create these using [`crossinterpolate2`](@ref) rather than calling a constructor directly.
"""
mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{Vector{MultiIndex}}
    Jset::Vector{Vector{MultiIndex}}
    localdims::Vector{Int}

    sitetensors::Vector{Array{ValueType,3}}
    T::Vector{Array{ValueType,3}}
    P::Vector{Array{ValueType,2}}
    lpos::Int
    rpos::Int

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
            [zeros(0, 0, 0) for d in localdims],    # sitetensors
            [zeros(0, 0, 0) for d in localdims],    # T
            [zeros(0, 0) for d in localdims],       # P
            -1,                                     # lpos
            -1,                                     # rpos
            [],                                     # pivoterrors
            zeros(length(localdims) - 1),           # bonderrors
            0.0,                                    # maxsamplevalue
        )
    end
end

function TensorCI2{ValueType}(
    func::F,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))]
) where {F,ValueType,N}
    tci = TensorCI2{ValueType}(localdims)
    addglobalpivots!(tci, initialpivots)
    tci.maxsamplevalue = maximum(abs, (func(x) for x in initialpivots))
    abs(tci.maxsamplevalue) > 0.0 || error("maxsamplevalue is zero!")
    _invalidatesitetensors!(tci)
    return tci
end

@doc raw"""
    function printnestinginfo(tci::TensorCI2{T}) where {T}

Print information about fulfillment of the nesting criterion (``I_\ell < I_{\ell+1}`` and ``J_\ell < J_{\ell+1}``) on each pair of bonds ``\ell, \ell+1`` of `tci` to stdout.
"""
function printnestinginfo(tci::TensorCI2{T}) where {T}
    printnestinginfo(stdout, tci)
end

function linkdims(tci::TensorCI2{T})::Vector{Int} where {T}
    return [length(tci.Iset[b+1]) for b in 1:length(tci)-1]
end



"""
Return if site tensors are available
"""
function issitetensorsavailable(tci::TensorCI2{T}) where {T}
    return all(length(tci.sitetensors[b]) != 0 for b in 1:length(tci))
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
Evaluate TP^{-1} at site `l`
"""
function TtimesPinv(tci::TensorCI2{ValueType}, l::Int)::Array{ValueType,3} where {ValueType}
    Tmat = reshape(tci.T[l], length(tci.Iset[l]) * tci.localdims[l], length(tci.Jset[l]))
    Pmat = reshape(tci.P[l], length(tci.Iset[l+1]), length(tci.Jset[l]))
    return reshape(
        transpose(transpose(Pmat) \ transpose(Tmat)),
        length(tci.Iset[l]), tci.localdims[l], length(tci.Iset[l+1]))
end

"""
Evaluate P^{-1}T at site `l`
"""
function PinvtimesT(tci::TensorCI2{ValueType}, l::Int)::Array{ValueType,3} where {ValueType}
    Tmat = reshape(tci.T[l], length(tci.Iset[l]), tci.localdims[l] * length(tci.Jset[l]))
    Pmat = reshape(tci.P[l-1], length(tci.Iset[l]), length(tci.Jset[l-1]))
    return reshape(Pmat \ Tmat, length(tci.Jset[l-1]), tci.localdims[l], length(tci.Jset[l]))
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

    if length(pivots) > 0
        _invalidatesitetensors!(tci)
    end

    nothing
end


"""
Add global pivots to index sets and perform a 1site sweep
"""
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


function existaspivot(
    tci::TensorCI2{ValueType},
    indexset::MultiIndex) where {ValueType}
    return [indexset[1:b-1] ∈ tci.Iset[b] && indexset[b+1:end] ∈ tci.Jset[b] for b in 1:length(tci)]
end


"""
Extract a list of existing pivots from a TCI2 object.
"""
function globalpivots(tci::TensorCI2{ValueType})::Vector{MultiIndex} where {ValueType}
    result = MultiIndex[]
    for b in 1:length(tci)-1, (x, y) in zip(tci.Iset[b+1], tci.Jset[b])
        push!(result, vcat(x, y))
    end
    return result
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

        sweep2site!(
            tci, f, 2;
            abstol=abstol,
            maxbonddim=maxbonddim,
            pivotsearch=pivotsearch,
            strictlynested=strictlynested,
            verbosity=verbosity,
            externalglobalpivots=pivots_
        )

        err_on_pivots = [abs(evaluate(tci, p) - f(p)) for p in pivots]
        newpivots = [p for (p, e) in zip(pivots, err_on_pivots) if e > abstol]

        if length(newpivots) > 0
            remaining_err = [e for e in err_on_pivots if e > abstol]
            @warn "Trying to add $(length(pivots_)) global pivots, $(length(newpivots)) still remain. Tolerance is $(abstol). Remaing errors are $(remaining_err). Retrying..."
        end

        if length(newpivots) == 0 || Set(newpivots) == Set(pivots_)
            return length(newpivots)
        end

        pivots_ = newpivots
    end
    return length(pivots_)
end

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
    _invalidatesitetensors!(tci)
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
    maxbonddim::Int=typemax(Int),
    updatetensors::Bool=true
) where {ValueType}
    flushpivoterror!(tci)

    _invalidatesitetensors!(tci)

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
        if any(isnan.(tci.sitetensors[b]))
            error("Error: NaN in tensor T[$b]")
        end
        updateerrors!(
            tci, b - !forwardsweep,
            pivoterrors(luci),
        )
    end

    if updatetensors
        _fillsitetensors!(tci, f)
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
    sweep1site!(tci, f, :forward; reltol, abstol, maxbonddim, updatetensors=false)
    sweep1site!(tci, f, :backward; reltol, abstol, maxbonddim, updatetensors=false)
    sweep1site!(tci, f, :forward; reltol, abstol, maxbonddim, updatetensors=true)
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
Site tensors will be invalidated.
"""
function updatepivots!(
    tci::TensorCI2{ValueType},
    b::Int,
    f::F,
    leftorthogonal::Bool;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int),
    sweepdirection::Symbol=:forward,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    extraouterIset::Vector{MultiIndex}=MultiIndex[],
    extraouterJset::Vector{MultiIndex}=MultiIndex[],
) where {F,ValueType}
    _invalidatesitetensors!(tci)

    if length(extraouterIset) > 0
        only(unique(length.(extraouterIset))) == length(tci.Iset[b][1]) || error("Invalid extraouterIset")
    end
    if length(extraouterJset) > 0
        only(unique(length.(extraouterJset))) == length(tci.Jset[b+1][1]) || error("Invalid extraouterJset")
    end

    Icombined = kronecker(unique(vcat(tci.Iset[b], extraouterIset)), tci.localdims[b])
    Jcombined = kronecker(tci.localdims[b+1], unique(vcat(tci.Jset[b+1], extraouterJset)))

    @assert length(unique(length.(Icombined))) == 1
    @assert length(unique(length.(Jcombined))) == 1

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
        _find_pivot(pivots, IJ)::Vector{Int} = unique(Int.(Iterators.filter(!isnothing, findfirst(isequal(i), IJ) for i in pivots)))
        I0 = _find_pivot(tci.Iset[b+1], Icombined)
        J0 = _find_pivot(tci.Jset[b], Jcombined)

        for (pos, i) in enumerate(Icombined)
            if i[1:end-1] ∈ extraouterIset
                push!(I0, pos)
            end
        end

        for (pos, j) in enumerate(Jcombined)
            if j[2:end] ∈ extraouterJset
                push!(J0, pos)
            end
        end

        I0 = unique(I0)
        J0 = unique(J0)

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
    #if length(extraouterIset) == 0 && length(extraouterJset) == 0
    #setsitetensor!(tci, b, left(luci))
    #setsitetensor!(tci, b + 1, right(luci))
    #end
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
        strictlynested::Bool=false
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
- `strictlynested::Bool` determines whether to preserve partial nesting in the TCI algorithm. Default: `false`.
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
    strictlynested::Bool=false,
    checkbatchevaluatable::Bool=false
) where {ValueType}
    errors = Float64[]
    ranks = Int[]
    nglobalpivots = Int[]

    if checkbatchevaluatable && !(f isa BatchEvaluator)
        error("Function `f` is not batch evaluatable")
    end

    #if maxnglobalpivot > 0 && nsearchglobalpivot > 0
    #!strictlynested || error("nglobalpivots > 0 requires strictlynested=false!")
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
            strictlynested=strictlynested,
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


#function _project_globalpivots(globalpivots::Vector{MultiIndex}, bondindex::Int)
#return collect(Set(p[1:bondindex] for p in globalpivots)), collect(Set(p[bondindex+1:end] for p in globalpivots))
#end
function _project_globalpivots(globalpivots::Vector{MultiIndex}, bondindex::Int)
    i =
        if bondindex > 1
            collect(Set(p[1:bondindex-1] for p in globalpivots))
        else
            MultiIndex[]
        end

    j =
        if bondindex == length(first(globalpivots)) - 1
            MultiIndex[]
        else
            collect(Set(p[bondindex+2:end] for p in globalpivots))
        end

    return i, j
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
    strictlynested::Bool=false,
    fillsitetensors::Bool=true,
    externalglobalpivots::Vector{MultiIndex}=MultiIndex[]
) where {ValueType}
    _invalidatesitetensors!(tci)

    if length(externalglobalpivots) > 0 && strictlynested
        @warn "strictlynested=true is not compatible with externalglobalpivots"
    end

    n = length(tci)

    allpivots::Vector{MultiIndex} = vcat(globalpivots(tci), externalglobalpivots)

    extraouterIset = MultiIndex[]
    extraouterJset = MultiIndex[]

    for iter in iter1:iter1+niter-1
        flushpivoterror!(tci)
        if forwardsweep(sweepstrategy, iter) # forward sweep
            for bondindex in 1:n-1
                if !strictlynested
                    extraouterIset, extraouterJset = _project_globalpivots(allpivots, bondindex)
                end
                updatepivots!(
                    tci, bondindex, f, true;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    sweepdirection=:forward,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    extraouterIset=extraouterIset,
                    extraouterJset=extraouterJset,
                )
            end
        else # backward sweep
            for bondindex in (n-1):-1:1
                if !strictlynested
                    extraouterIset, extraouterJset = _project_globalpivots(allpivots, bondindex)
                end
                updatepivots!(
                    tci, bondindex, f, false;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    sweepdirection=:backward,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    extraouterIset=extraouterIset,
                    extraouterJset=extraouterJset,
                )
            end
        end
    end

    if fillsitetensors
        _fillsitetensors!(tci, f)
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

    if !issitetensorsavailable(tci)
        _fillsitetensors!(tci, f)
    end

    pivots = Dict{Float64,MultiIndex}()
    ttcache = TTCache(tci)
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


"""
Invalidate sitetensors
"""
function _invalidatesitetensors!(tci::TensorCI2{T}) where {T}
    for l in 1:length(tci)
        tci.sitetensors[l] = zeros(T, 0, 0, 0)
        tci.T[l] = zeros(T, 0, 0, 0)
    end
    for p in 1:length(tci)-1
        tci.P[p] = zeros(T, 0, 0)
    end
    tci.lpos = -1
    tci.rpos = -1
    nothing
end


function _fillsitetensors!(tci::TensorCI2{ValueType}, f; lpos=0) where {ValueType}
    # fill P matrices
    for p in 1:length(tci)-1
        tci.P[p] = reshape(
            filltensor(ValueType, f, tci.localdims, tci.Iset[p+1], tci.Jset[p], Val(0)),
            length(tci.Iset[p+1]), length(tci.Jset[p]))
    end

    # fill T tensors
    for l in 1:length(tci)
        tci.T[l] = filltensor(ValueType, f, tci.localdims, tci.Iset[l], tci.Jset[l], Val(1))
    end

    # fill sitetensors
    rpos = lpos + 2
    for l in 1:lpos
        tci.sitetensors[l] = TtimesPinv(tci, l)
    end
    for l in lpos+1:rpos-1
        tci.sitetensors[l] = copy(tci.T[l])
    end
    for l in rpos:length(tci)
        tci.sitetensors[l] = PinvtimesT(tci, l)
    end
    tci.lpos = lpos
    tci.rpos = lpos + 2

    nothing
end