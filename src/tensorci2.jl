
"""
    mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}

Type that represents tensor cross interpolations created using the TCI2 algorithm. Users may want to create these using [`crossinterpolate2`](@ref) rather than calling a constructor directly.
"""
mutable struct TensorCI2{ValueType}
    globalpivots::Vector{MultiIndex}
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
        new{ValueType}(
            MultiIndex[],                           # globalpivots
            collect(localdims),                     # localdims
            [],                                     # pivoterrors
            zeros(length(localdims) - 1),           # bonderrors
            0.0,                                    # maxsamplevalue
        )
    end
end

Base.length(tci::TensorCI2) = length(tci.localdims)

rank(tci::TensorCI2) = length(tci.globalpivots)

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

function Iset(tci::TensorCI2{T}, i::Int)::Vector{MultiIndex} where {T}
    return collect(Set(p[1:i-1] for p in tci.globalpivots))
end

function Jset(tci::TensorCI2{T}, i::Int)::Vector{MultiIndex} where {T}
    return collect(Set(p[i+1:end] for p in tci.globalpivots))
end

function linkdims(tci::TensorCI2{T})::Vector{Int} where {T}
    return [min(length(Iset(tci, b+1)), length(Jset(tci, b))) for b in 1:length(tci)-1]
end

"""
Invalidate the site tensor at bond `b`.
"""
#function invalidatesitetensors!(tci::TensorCI2{T}) where {T}
    #for b in 1:length(tci)
        #tci.sitetensors[b] = zeros(T, 0, 0, 0)
    #end
    #nothing
#end

"""
Return if site tensors are available
"""
#function issitetensorsavailable(tci::TensorCI2{T}) where {T}
    #return all(length(tci.sitetensors[b]) != 0 for b in 1:length(tci))
#end


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

    tci.globalpivots = unique(vcat(tci.globalpivots, pivots))

    nothing
end

"""
Extract a list of existing pivots from a TCI2 object.
"""
function globalpivots(tci::TensorCI2{ValueType})::Vector{MultiIndex} where {ValueType}
    return tci.globalpivots
end


function globalpivots!(p::Vector{MultiIndex}, Iset, Jset)::Vector{MultiIndex}
    for x in Iset, y in Jset
        pushunique!(p, vcat(x, y))
    end
    return p
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
    verbosity::Int=0
)::Nothing where {F,ValueType}
    if any(length(tci) .!= length.(pivots))
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the MPS."))
    end

    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    abstol = tolerance * errornormalization

    addglobalpivots!(tci, pivots)
    sweep2site!(
        tci, f, 2;
        abstol=abstol,
        maxbonddim=maxbonddim,
        pivotsearch=pivotsearch,
        verbosity=verbosity
        )
    nothing
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

#function setsitetensor!(
    #tci::TensorCI2{ValueType}, b::Int, T::AbstractArray{ValueType,N}
#) where {ValueType,N}
    #tci.sitetensors[b] = reshape(
        #T,
        #length(Iset(tci, b)),
        #tci.localdims[b],
        #length(Jset(tci, b))
    #)
#end

function sitetensors(
    tci::TensorCI2{ValueType}, f; orthocenter = length(tci)
)::Vector{Array{ValueType,3}} where {ValueType}
    tensors = Array{ValueType,3}[]
    for b in 1:orthocenter-1
        push!(tensors, Atensor(tci,f, b))
    end
    push!(tensors, Ttensor(tci, f, orthocenter))
    for b in orthocenter+1:length(tci)
        push!(tensors, Btensor(tci,f, b))
    end
    return tensors
end


function sitetensors_site0update(
    tci::TensorCI2{ValueType}, f;
    reltol=1e-14,
    abstol=0.0
)::Vector{Array{ValueType,3}} where {ValueType}

    # site0 update
    Iset_ = [Iset(tci, b) for b in 1:length(tci)]
    Jset_ = [Jset(tci, b) for b in 1:length(tci)]

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

         ndiag = sum(abs(F.lu.U[i,i]) > abstol && abs(F.lu.U[i,i]/F.lu.U[1,1]) > reltol for i in eachindex(diag(F.lu.U)))

         Iset_[b+1] = Iset_[b+1][rowindices(F)[1:ndiag]]
         Jset_[b] = Jset_[b][colindices(F)[1:ndiag]]
    end

    tensors = Array{ValueType,3}[]

    # Compute A tensors
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

    return tensors
end

function Ttensor(
    tci::TensorCI2{ValueType}, f, b::Int
)::Array{ValueType,3} where {ValueType}
    Iset_b = Iset(tci, b)
    Jset_b = Jset(tci, b)
    return filltensor(ValueType, f, tci.localdims, Iset_b, Jset_b, Val(1))
end


function Atensor(
    tci::TensorCI2{ValueType}, f, b::Int
)::Array{ValueType,3} where {ValueType}
    Iset_b = Iset(tci, b)
    Jset_b = Jset(tci, b)

    Pi1 = reshape(
        filltensor(ValueType, f, tci.localdims, Iset_b, Jset_b, Val(1)),
        length(Iset_b) * tci.localdims[b], length(Jset_b))

    Iset_bp1 = Iset(tci, b+1)
    P = reshape(
        filltensor(ValueType, f, tci.localdims, Iset_bp1, Jset_b, Val(0)),
        length(Iset_bp1), length(Jset_b))

    # T P^{-1}
    Tmat = transpose(transpose(P) \ transpose(Pi1))
    return reshape(Tmat, length(Iset_b), tci.localdims[b], length(Iset_bp1))
end


function Btensor(
    tci::TensorCI2{ValueType}, f, b::Int
)::Array{ValueType,3} where {ValueType}
    Iset_b = Iset(tci, b)
    Jset_b = Jset(tci, b)
    Jset_bm1 = Jset(tci, b-1)

    Pi1 = reshape(
        filltensor(ValueType, f, tci.localdims, Iset_b, Jset_b, Val(1)),
        length(Iset_b), tci.localdims[b] * length(Jset_b))

    P = reshape(
        filltensor(ValueType, f, tci.localdims, Iset_b, Jset_bm1, Val(0)),
        length(Iset_b), length(Jset_bm1))

    # P^{-1} T
    Tmat = P \ Pi1
    return reshape(Tmat, length(Jset_bm1), tci.localdims[b], length(Jset_b))
end




function setsitetensor!(
    tci::TensorCI2{ValueType}, f, b::Int; leftorthogonal=true
) where {ValueType}
    leftorthogonal || error("leftorthogonal==false is not supported!")

    Iset_b = Iset(tci, b)
    Jset_b = Jset(tci, b)

    Is = leftorthogonal ? kronecker(Iset_b, tci.localdims[b]) : Iset_b
    Js = leftorthogonal ? Jset_b : kronecker(tci.localdims[b], Jset_b)
    Pi1 = reshape(
        filltensor(ValueType, f, tci.localdims, Iset_b, Jset_b, Val(1)),
        length(Is), length(Js))
    updatemaxsample!(tci, Pi1)

    if (leftorthogonal && b == length(tci)) ||
       (!leftorthogonal && b == 1)
        setsitetensor!(tci, b, Pi1)
        return tci.sitetensors[b]
    end

    Iset_bp1 = Iset(tci, b+1)
    P = reshape(
        filltensor(ValueType, f, tci.localdims, Iset_bp1, Jset_b, Val(0)),
        length(Iset_bp1), length(Jset_b))

    # T P^{-1}
    Tmat = transpose(transpose(P) \ transpose(Pi1))
    tci.sitetensors[b] = reshape(Tmat, length(Iset_b), tci.localdims[b], length(Iset_bp1))
    return tci.sitetensors[b]
end


function updatemaxsample!(tci::TensorCI2{V}, samples::Array{V}) where {V}
    tci.maxsamplevalue = maxabs(tci.maxsamplevalue, samples)
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
function updatepivots(
    tci::TensorCI2{ValueType},
    b::Int,
    f::F,
    leftorthogonal::Bool;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int),
    sweepdirection::Symbol=:forward,
    pivotsearch::Symbol=:full,
    verbosity::Int=0
) where {F,ValueType}
    Iset_b = Iset(tci, b)
    Jset_bp1 = Jset(tci, b+1)
    Icombined = kronecker(Iset_b, tci.localdims[b])
    Jcombined = kronecker(tci.localdims[b+1], Jset_bp1)

    @assert length(unique(length.(Icombined))) == 1
    @assert length(unique(length.(Jcombined))) == 1

    maxsamplevalue = tci.maxsamplevalue

    luci = if pivotsearch === :full
        t1 = time_ns()
        Pi = reshape(
            filltensor(ValueType, f, tci.localdims,
            Icombined, Jcombined, Val(0)),
            length(Icombined), length(Jcombined)
        )
        t2 = time_ns()

        maxsamplevalue = max(maxsamplevalue, maximum(abs, Pi))
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
        I0 = _find_pivot(Iset(tci, b+1), Icombined)
        J0 = _find_pivot(Jset(tci, b), Jcombined)

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
        maxsamplevalue = max(maxsamplevalue, Pif.maxsamplevalue)

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
    return Icombined[rowindices(luci)], Jcombined[colindices(luci)], pivoterrors(luci), maxsamplevalue
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
    tolmarginglobalsearch::Float64=10.0,
    checkbatchevaluatable::Bool=false
) where {ValueType}
    errors = Float64[]
    ranks = Int[]
    nglobalpivots = Int[]

    if checkbatchevaluatable && !(f isa BatchEvaluator)
        error("Function `f` is not batch evaluatable")
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
            sweepstrategy=sweepstrategy
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
        push!(nglobalpivots, 0)

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
    #sweep1site!(
    #tci,
    #f,
    #abstol=abstol,
    #maxbonddim=maxbonddim,
    #)

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
) where {ValueType}
    n = length(tci)

    for iter in iter1:iter1+niter-1
        flushpivoterror!(tci)
        globalpivots_new = MultiIndex[]
        if forwardsweep(sweepstrategy, iter) # forward sweep
            for bondindex in 1:n-1
                Iset, Jset, errors, maxsamplevalue = updatepivots(
                    tci, bondindex, f, true;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    sweepdirection=:forward,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    #extraouterIset=extraouterIset,
                    #extraouterJset=extraouterJset,
                )
                updatemaxsample!(tci, [maxsamplevalue])
                updateerrors!(tci, bondindex, errors)
                globalpivots!(globalpivots_new, Iset, Jset)
            end
        else # backward sweep
            for bondindex in (n-1):-1:1
                Iset, Jset, errors, maxsamplevalue = updatepivots(
                    tci, bondindex, f, false;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    sweepdirection=:backward,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    #extraouterIset=extraouterIset,
                    #extraouterJset=extraouterJset,
                )
                updatemaxsample!(tci, [maxsamplevalue])
                updateerrors!(tci, bondindex, errors)
                globalpivots!(globalpivots_new, Iset, Jset)
            end
        end
        replaceglobalpivots!(tci, globalpivots_new)
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
        tolmarginglobalsearch::Float64=10.0,
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
- `tolmarginglobalsearch` can be set to `>= 1.0`. Seach global pivots where the interpolation error is larger than the tolerance by `tolmarginglobalsearch`.  Default: `10.0`.
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


function replaceglobalpivots!(tci::TensorCI2{V}, p::AbstractVector{MultiIndex}) where {V}
    addglobalpivots!(tci, p)
end