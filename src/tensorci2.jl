
"""
    mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}

Type that represents tensor cross interpolations created using the TCI2 algorithm. Users may want to create these using [`crossinterpolate2`](@ref) rather than calling a constructor directly.
"""
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

    cache::Vector{Dict{MultiIndex,Matrix{ValueType}}}

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
    addglobalpivots!(tci, func, initialpivots)
    return tci
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


function updatebonderror!(
    tci::TensorCI2{T}, b::Int, sweepdirection::Symbol, error::Float64
) where {T}
    if sweepdirection === :forward
        tci.bonderrorsforward[b] = error
    elseif sweepdirection === :backward
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
    tci::TensorCI2{T},
    b::Int,
    sweepdirection::Symbol,
    errors::AbstractVector{Float64},
    lastpivoterror::Float64
) where {T}
    updatebonderror!(tci, b, sweepdirection, lastpivoterror)
    updatepivoterror!(tci, errors)
    nothing
end

function addglobalpivots!(
    tci::TensorCI2{ValueType},
    f::F,
    pivots::Vector{MultiIndex};
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
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

    makecanonical!(tci, f; reltol=reltol, abstol=abstol, maxbonddim=maxbonddim)
end


# Default slow implementation
function _batchevaluate_dispatch(
    ::Type{ValueType},
    f,
    localset::Vector{Vector{LocalIndex}},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
)::Array{ValueType} where {ValueType}
    if length(Iset) * length(Jset) == 0
        return ValueType[]
    end
    N = length(localset)
    nl = length(first(Iset))
    nr = length(first(Jset))
    ncent = N - nl - nr
    return ValueType[
        f(collect(Iterators.flatten(i)))
        for i in Iterators.product(Iset, localset[nl+1:nl+ncent]..., Jset)
    ]
end


function _batchevaluate_dispatch(
    ::Type{ValueType},
    f::BatchEvaluator{ValueType},
    localset::Vector{Vector{LocalIndex}},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
)::Array{ValueType} where {ValueType}
    if length(Iset) * length(Jset) == 0
        return ValueType[]
    end
    N = length(localset)
    nl = length(first(Iset))
    nr = length(first(Jset))
    ncent = N - nl - nr
    return f(Iset, Jset, Val(ncent))
end


function _batchevaluate(
    ::Type{ValueType},
    f,
    localset::Vector{Vector{LocalIndex}},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex}
) where {ValueType}
    if length(Iset) * length(Jset) == 0
        return ValueType[]
    end

    N = length(localset)
    nl = length(first(Iset))
    nr = length(first(Jset))
    ncent = N - nl - nr
    expected_size = (length(Iset), length.(localset[nl+1:nl+ncent])..., length(Jset))
    result = reshape(
        _batchevaluate_dispatch(ValueType, f, localset, Iset, Jset),
        expected_size...
    )
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

    if !(sweepdirection === :forward || sweepdirection === :backward)
        throw(ArgumentError("Unknown sweep direction $sweepdirection: choose between :forward, :backward."))
    end

    forwardsweep = sweepdirection === :forward
    for b in (forwardsweep ? (1:length(tci)-1) : (length(tci):-1:2))
        Is = forwardsweep ? kronecker(tci.Iset[b], tci.localset[b]) : tci.Iset[b]
        Js = forwardsweep ? tci.Jset[b] : kronecker(tci.localset[b], tci.Jset[b])
        Pi = reshape(
            _batchevaluate(ValueType, f, tci.localset, tci.Iset[b], tci.Jset[b]),
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
        if updatetensors
            setT!(tci, b, forwardsweep ? left(luci) : right(luci))
        end
        if any(isnan.(tci.T[b]))
            error("Error: NaN in tensor T[$b]")
        end
        updateerrors!(
            tci, b - !forwardsweep, sweepdirection,
            pivoterrors(luci), lastpivoterror(luci)
        )
    end

    # Update last tensor according to last index set
    if updatetensors
        lastupdateindex = forwardsweep ? length(tci) : 1
        shape = if forwardsweep
            (length(tci.Iset[end]), length(tci.localset[end]))
        else
            (length(tci.localset[begin]), length(tci.Jset[begin]))
        end
        localtensor = reshape(_batchevaluate(
            ValueType, f, tci.localset, tci.Iset[lastupdateindex], tci.Jset[lastupdateindex]), shape)
        setT!(tci, lastupdateindex, localtensor)
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
    return batchevaluate(f, Iset, Jset, 0)
end
 

function (obj::SubMatrix{T})(irows::Vector{Int}, icols::Vector{Int})::Matrix{T} where T
    res = _submatrix_batcheval(obj, obj.f, irows, icols)
    obj.maxsamplevalue = max(obj.maxsamplevalue, maximum(abs, res))
    return res
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
    pivotsearch::Symbol=:full
) where {F,ValueType}
    Icombined = kronecker(tci.Iset[b], tci.localset[b])
    Jcombined = kronecker(tci.localset[b+1], tci.Jset[b+1])
    luci = if pivotsearch === :full
        Pi = reshape(
            _batchevaluate(ValueType, f, tci.localset, tci.Iset[b], tci.Jset[b+1]),
            length(Icombined), length(Jcombined)
        )
        updatemaxsample!(tci, Pi)
        MatrixLUCI(
            Pi,
            reltol=reltol,
            abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=leftorthogonal
        )
    elseif pivotsearch === :rook
        I0 = Int.(Iterators.filter(!isnothing, findfirst(isequal(i), Icombined) for i in tci.Iset[b+1]))
        J0 = Int.(Iterators.filter(!isnothing, findfirst(isequal(j), Jcombined) for j in tci.Jset[b]))
        Pif = SubMatrix{ValueType}(f, Icombined, Jcombined)
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
        res
    else
        throw(ArgumentError("Unknown pivot search strategy $pivotsearch. Choose from :rook, :full."))
    end
    tci.Iset[b+1] = Icombined[rowindices(luci)]
    tci.Jset[b] = Jcombined[colindices(luci)]
    setT!(tci, b, left(luci))
    setT!(tci, b + 1, right(luci))
    updateerrors!(tci, b, sweepdirection, pivoterrors(luci), lastpivoterror(luci))
    nothing
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
    maxiter::Int=20,
    sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.backandforth,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    loginterval::Int=10,
    normalizeerror::Bool=true,
    ncheckhistory=3,
    lengthfz=5
) where {ValueType}
    n = length(tci)
    errors = Float64[]
    ranks = Int[]

    tstart = time_ns()

    if maxbonddim >= typemax(Int) && tolerance <= 0
        throw(ArgumentError(
            "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence criterion is not reachable!"
        ))
    end

    for iter in rank(tci)+1:maxiter
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0

        if verbosity > 0
            println("Walltime $(1e-9*(time_ns() - tstart)) sec: starting floatingzone")
        end
        _, fz_err = floatingzone!(
            tci, f;
            verbosity=verbosity,
            lengthzone=min(lengthfz, n),
        )

        flushpivoterror!(tci)
        if verbosity > 0
            println("Walltime $(1e-9*(time_ns() - tstart)) sec: starting two-site sweep")
        end
        if forwardsweep(sweepstrategy, iter) # forward sweep
            for bondindex in 1:n-1
                updatepivots!(
                    tci, bondindex, f, true;
                    abstol=pivottolerance * errornormalization,
                    maxbonddim=maxbonddim,
                    sweepdirection=:forward,
                    pivotsearch=pivotsearch
                )
            end
        else # backward sweep
            for bondindex in (n-1):-1:1
                updatepivots!(
                    tci, bondindex, f, false;
                    abstol=pivottolerance * errornormalization,
                    maxbonddim=maxbonddim,
                    sweepdirection=:backward,
                    pivotsearch=pivotsearch
                )
            end
        end
        if verbosity > 0
            println("Walltime $(1e-9*(time_ns() - tstart)) sec: done two-site sweep")
        end

        push!(errors, max(pivoterror(tci), fz_err))
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
- `pivotsearch::Symbol` determins how pivots are searched (`:full` or `:rook`). Default: `:full`.

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
    kwargs...
) where {ValueType,N}
    tci = TensorCI2{ValueType}(f, localdims, initialpivots)
    ranks, errors = optimize!(tci, f; kwargs...)
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
    normalizeerror::Bool=true,
    maxsweep::Int=1000
)::Int where {ValueType}
    localdims = [length(s) for s in tci.localset]

    nnewpivot = 0
    for isearch in 1:nsearch
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0

        err(x) = abs(evaluate(tci, x) - f(x))
        newpivot_ = optfirstpivot(err, localdims, [rand(1:d) for d in localdims]; maxsweep=maxsweep)

        e = err(newpivot_)
        if isnan(e)
            ttval = evaluate(tci, newpivot_)
            error("NaN error for pivot $(newpivot_), ttval=$ttval")
        end

        if verbosity > 0 && mod(isearch, 100) == 0
            println()
            println("$(isearch)-th attempt: error $e")
        end
        if e > tolerance * errornormalization
            nnewpivot += 1
            if verbosity > 0
                println()
                println("$(isearch)-th attempt: Inserting a global pivot $(newpivot_): error $e > $(tolerance * errornormalization)")
                println("Old error is $(err(newpivot_)).")
            end
            addglobalpivots!(
                tci, f, [newpivot_];
                abstol=tolerance * errornormalization,
                reltol=0.0
                )
            if verbosity > 0
                nan =  [any(isnan.(t)) for t in tci.T]
                any(nan) && error("tt is $nan")
                println("New linkdims is $(maximum(linkdims(tci))). New error is $(err(newpivot_)).")
            end
        end
    end

    return nnewpivot
end


function floatingzone!(
    tci::TensorCI2{ValueType}, f;
    nmaxglobalpivots = 1,
    verbosity::Int=0,
    lengthzone=1,
)::Tuple{Int,Float64} where {ValueType}
    result = Dict{Float64,MultiIndex}()
    for nl in 0:(length(tci)-lengthzone)
        nr = length(tci) - nl - lengthzone
        maxerror, pivot = _floatingzone(tci, f, nl, nr)
        if pivot !== nothing
            result[maxerror] = pivot 
        end
    end

    result_sorted = sort(collect(result); by=x->x[1], rev=true)
    if length(result_sorted) > nmaxglobalpivots
        result_sorted = result_sorted[1:nmaxglobalpivots]
    end
    
    bonddim_prev = maximum(linkdims(tci))
    t1 = time_ns()

    addglobalpivots!(
        tci, f, collect(map(x->x[2], result_sorted));
        abstol=0.0, reltol=1e-15
    )
    bonddim = maximum(linkdims(tci))

    t2 = time_ns()
    if verbosity > 0
        println("Added $(length(result_sorted)) global pivots (max error $(result_sorted[1][1]), min error $(result_sorted[end][1]))) " *
        ": bonddim $(bonddim_prev) -> $(bonddim) with $((t2-t1)*1e-9) sec")
    end

    return length(result_sorted), result_sorted[1][1]
end


function _floatingzone(
    tci::TensorCI2{ValueType}, f,
    nl, nr;
    #tolerance::Float64=1e-8,
    #verbosity::Int=0,
    nsweeps=10,
    #normalizeerror::Bool=true,
) where {ValueType}
    localdims = [length(s) for s in tci.localset]

    n = length(tci)

    ttcache = TTCache(tci.T)

    lengthzone = length(tci) - nl - nr

    idxzone = [rand(1:d) for d in localdims[nl+1:nl+lengthzone]]

    maxerror = 0.0
    pivot::Union{Nothing,MultiIndex} = nothing

    for isweep in 1:nsweeps
        prev_maxerror = maxerror
        for ipos in 1:lengthzone
            zoneset = [vcat(idxzone[1:ipos-1], it, idxzone[ipos+1:end]) for it in tci.localset[ipos+nl]]
            Jset = [vcat(z, j) for z in zoneset, j in tci.Jset[n-nr]]

            exactdata = _batchevaluate(
                ValueType,
                f,
                tci.localset,
                tci.Iset[nl+1],
                vec(Jset)
            )
            prediction = _batchevaluate(
                ValueType,
                ttcache,
                tci.localset,
                tci.Iset[nl+1],
                vec(Jset)
            )
            err = reshape(abs.(exactdata .- prediction), length(tci.Iset[nl+1]), localdims[ipos+nl], length(tci.Jset[n-nr]))

            maxerror = maximum(err)
            argmax_ = argmax(err)
            pivot = vcat(tci.Iset[nl+1][argmax_[1]], zoneset[argmax_[2]], tci.Jset[n-nr][argmax_[3]])

            idxzone[ipos] = tci.localset[ipos+nl][argmax_[2]]
        end

        if maxerror == prev_maxerror
            break
        end
    end

    return maxerror, pivot
end
