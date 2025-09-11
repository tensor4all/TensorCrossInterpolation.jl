"""
    mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}

Type that represents tensor cross interpolations created using the TCI2 algorithm. Users may want to create these using [`crossinterpolate2`](@ref) rather than calling a constructor directly.
"""
mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{Vector{MultiIndex}}
    Jset::Vector{Vector{MultiIndex}}
    localdims::Vector{Int}

    sitetensors::Vector{Array{ValueType,3}}

    "Error estimate for backtruncation of bonds."
    pivoterrors::Vector{Float64}
    #"Error estimate per bond by 2site sweep."
    bonderrors::Vector{Float64}
    #"Maximum sample for error normalization."
    maxsamplevalue::Float64

    Iset_history::Vector{Vector{Vector{MultiIndex}}}
    Jset_history::Vector{Vector{Vector{MultiIndex}}}

    function TensorCI2{ValueType}(
        localdims::Union{Vector{Int},NTuple{N,Int}}
    ) where {ValueType,N}
        length(localdims) > 1 || error("localdims should have at least 2 elements!")
        n = length(localdims)
        new{ValueType}(
            [Vector{MultiIndex}() for _ in 1:n],    # Iset
            [Vector{MultiIndex}() for _ in 1:n],    # Jset
            collect(localdims),                     # localdims
            [zeros(0, d, 0) for d in localdims],    # sitetensors
            [],                                     # pivoterrors
            zeros(length(localdims) - 1),           # bonderrors
            0.0,                                    # maxsamplevalue
            Vector{Vector{MultiIndex}}[],           # Iset_history
            Vector{Vector{MultiIndex}}[],           # Jset_history
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
    invalidatesitetensors!(tci)
    return tci
end

"""
    Initialize a TCI2 object with local pivot lists.
"""
function TensorCI2{ValueType}(
    func::F,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    Iset::Vector{Vector{MultiIndex}},
    Jset::Vector{Vector{MultiIndex}}
) where {F,ValueType,N}
    tci = TensorCI2{ValueType}(localdims)
    tci.Iset = Iset
    tci.Jset = Jset
    pivots = reconstractglobalpivotsfromijset(localdims, tci.Iset, tci.Jset)
    tci.maxsamplevalue = maximum(abs, (func(bit) for bit in pivots))
    abs(tci.maxsamplevalue) > 0.0 || error("maxsamplevalue is zero!")
    invalidatesitetensors!(tci)
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
Invalidate the site tensor at bond `b`.
"""
function invalidatesitetensors!(tci::TensorCI2{T}) where {T}
    for b in 1:length(tci)
        tci.sitetensors[b] = zeros(T, 0, 0, 0)
    end
    nothing
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

function reconstractglobalpivotsfromijset(
    localdims::Union{Vector{Int},NTuple{N,Int}},
    Isets::Vector{Vector{MultiIndex}}, 
    Jsets::Vector{Vector{MultiIndex}}
) where {N}
    pivots = []
    l = length(Isets)
    for i in 1:l
        for Iset in Isets[i]
            for Jset in Jsets[i]
                for j in 1:localdims[i]
                    pushunique!(pivots, vcat(Iset, [j], Jset))
                end
            end
        end
    end
    return pivots
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
        invalidatesitetensors!(tci)
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
    strictlynested::Bool=false,
    multithread_eval::Bool=false
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
            verbosity=verbosity,
            multithread_eval=multithread_eval)

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

function filltensor(
    ::Type{ValueType},
    f,
    localdims::Vector{Int},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
    ::Val{M}
)::Array{ValueType,M+2} where {ValueType,M}
    if length(Iset) * length(Jset) == 0
        return Array{ValueType,M+2}(undef, ntuple(i->0, M+2)...)
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

function setsitetensor!(
    tci::TensorCI2{ValueType}, b::Int, T::AbstractArray{ValueType,N}
) where {ValueType,N}
    tci.sitetensors[b] = reshape(
        T,
        length(tci.Iset[b]),
        tci.localdims[b],
        length(tci.Jset[b])
    )
end


function sweep0site!(
    tci::TensorCI2{ValueType}, f, b::Int;
    reltol=1e-14, abstol=0.0
) where {ValueType}
    invalidatesitetensors!(tci)
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

    ndiag = sum(abs(F.lu.U[i,i]) > abstol && abs(F.lu.U[i,i]/F.lu.U[1,1]) > reltol for i in eachindex(diag(F.lu.U)))

    tci.Iset[b+1] = tci.Iset[b+1][rowindices(F)[1:ndiag]]
    tci.Jset[b] = tci.Jset[b][colindices(F)[1:ndiag]]
    return nothing
end

# Backward compatibility
const rmbadpivots! = sweep0site!

function setsitetensor!(
    tci::TensorCI2{ValueType}, f, b::Int; leftorthogonal=true
) where {ValueType}
    leftorthogonal || error("leftorthogonal==false is not supported!")

    Is = leftorthogonal ? kronecker(tci.Iset[b], tci.localdims[b]) : tci.Iset[b]
    Js = leftorthogonal ? tci.Jset[b] : kronecker(tci.localdims[b], tci.Jset[b])
    Pi1 = reshape(
        filltensor(ValueType, f, tci.localdims, tci.Iset[b], tci.Jset[b], Val(1)),
        length(Is), length(Js))
    updatemaxsample!(tci, Pi1)

    if (leftorthogonal && b == length(tci)) ||
        (!leftorthogonal && b == 1)
        setsitetensor!(tci, b, Pi1)
        return tci.sitetensors[b]
    end

    P = reshape(
        filltensor(ValueType, f, tci.localdims, tci.Iset[b+1], tci.Jset[b], Val(0)),
        length(tci.Iset[b+1]), length(tci.Jset[b]))
    length(tci.Iset[b+1]) == length(tci.Jset[b]) || error("Pivot matrix at bond $(b) is not square!")

    #Tmat = transpose(transpose(rrlu(P)) \ transpose(Pi1))
    Tmat = transpose(transpose(P) \ transpose(Pi1))
    tci.sitetensors[b] = reshape(Tmat, length(tci.Iset[b]), tci.localdims[b], length(tci.Iset[b+1]))
    return tci.sitetensors[b]
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

    invalidatesitetensors!(tci)

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
        if updatetensors
            setsitetensor!(tci, b, forwardsweep ? left(luci) : right(luci))
        end
        if any(isnan.(tci.sitetensors[b]))
            error("Error: NaN in tensor T[$b]")
        end
        updateerrors!(
            tci, b - !forwardsweep,
            pivoterrors(luci),
        )
    end

    # Update last tensor according to last index set
    if updatetensors
        lastupdateindex = forwardsweep ? length(tci) : 1
        shape = if forwardsweep
            (length(tci.Iset[end]), tci.localdims[end])
        else
            (tci.localdims[begin], length(tci.Jset[begin]))
        end
        localtensor = reshape(filltensor(
            ValueType, f, tci.localdims, tci.Iset[lastupdateindex], tci.Jset[lastupdateindex], Val(1)), shape)
        setsitetensor!(tci, lastupdateindex, localtensor)
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
    # The first half-sweep is performed exactly without compression.
    sweep1site!(tci, f, :forward; reltol=0.0, abstol=0.0, maxbonddim=typemax(Int), updatetensors=false)
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
    Noderanges(nprocs, nbonds, localdim, estimatedbond; interbond=false, estimatedbonds=nothing, algorithm=:tci, efficiencycheck=false)

Distributes the computational workload efficiently across multiple nodes for tensor train algorithms.

# Arguments
- `nprocs::Int`: Number of processes (nodes) to distribute the workload across.
- `nbonds::Int`: Number of bonds in the tensor train.
- `localdim::Int`: Local dimension of the tensor train.
- `estimatedbond::Int`: Estimated bond dimension after compression, set by the user to help balance the workload.
- `interbond::Bool=false`: If `true`, distributes workload across bonds; if `false`, only function evaluation is distributed, equivalent to serial TCI.
- `estimatedbonds::Union{Nothing, Vector{Int}}=nothing`: Optional vector of estimated bond dimensions for each bond. If not provided, it is computed from `estimatedbond`.
- `algorithm::Symbol=:tci`: Algorithm being parallelized (e.g., `:tci`, `:mpompo`). Different algorithms may have different workload characteristics.
- `efficiencycheck::Bool=false`: If `true`, the function also returns the efficiency of the distribution.

# Returns
A tuple containing the ranges of work assigned to each node. If `efficiencycheck` is `true`, also returns the efficiency of the distribution.

# Notes
- The function is designed to optimize workload distribution for tensor train computations, taking into account the estimated bond dimensions and the chosen algorithm.
- Proper setting of `estimatedbond` or `estimatedbonds` can improve parallel efficiency.
"""


function _noderanges(nprocs::Int, nbonds::Int, localdim::Int, estimatedbond::Int; interbond::Bool=true, estimatedbonds::Union{Vector{Int},Nothing}=nothing, algorithm::Symbol=:tci, efficiencycheck::Bool=false)::Union{Vector{UnitRange{Int}},Tuple{Vector{UnitRange{Int}},Float64}}
    if !interbond
        return [1:nbonds for _ in 1:nprocs]
    end
    if estimatedbonds === nothing
        estimatedbonds = [estimatedbond for _ in 1:nbonds-1]
        i = 1
        while localdim^i < estimatedbond && i <= nbonds/2
            estimatedbonds[i] = localdim^i
            estimatedbonds[end-i+1] = localdim^i
            i = i + 1
        end 
    end

    if algorithm == :tci
        # ~ chi^3
        costs = [(i == 1 ? 1 : estimatedbonds[i-1]) * (i == nbonds ? 1 : estimatedbonds[i]) * min((i == 1 ? 1 : estimatedbonds[i-1]) * (i == nbonds ? 1 : estimatedbonds[i])) for i in 1:nbonds]
    else
        # ~ chi^4
        costs = [(i == 1 ? 1 : estimatedbonds[i-1]) * (i == nbonds ? 1 : estimatedbonds[i]) * estimatedbond * min((i == 1 ? 1 : estimatedbonds[i-1]) * (i == nbonds ? 1 : estimatedbonds[i])) for i in 1:nbonds]
    end
    total_cost = sum(costs)
    target_cost = total_cost / nprocs

    assignments = Vector{UnitRange{Int}}()

    i = 1  # bond index
    node = 1
    current_cost = 0
    range_start = 1

    while i <= nbonds && node < nprocs
        bond_cost = costs[i]

        if current_cost + bond_cost <= target_cost * 1.3
            # Add the current bond to the node's range
            current_cost += bond_cost
            i += 1
        else
            # Current node is full, move to the next node
            push!(assignments, range_start:i-1)
            node += 1
            current_cost = bond_cost
            range_start = i
            i += 1
        end
    end
    push!(assignments, range_start:nbonds)
    leftover = nprocs-node
    if leftover > 0 
        for k in 1:Int(ceil(leftover/node))
            for j in 1:(k == Int(ceil(leftover/node)) ? rem(leftover, node) : node)
                if j%2 == 1
                    indx = Int(j + (j-1)/2 * (k-1))
                    insert!(assignments, indx, assignments[indx])
                else
                    indx = Int(length(assignments)+2-j - (j/2)*(k-1))
                    insert!(assignments, indx, assignments[indx])
                end
            end
        end
    end
    works = [sum(costs[i]) for i in assignments]
    efficiency = total_cost / maximum(works)
    if efficiencycheck 
        return assignments, efficiency
    else
        return assignments
    end
end

"""
    _leaders(nprocs::Int, noderanges::Vector{UnitRange{Int}}) -> (leaders::Vector{Int}, leaderslist::Vector{Int})

Determine the leader processes among `nprocs` processes based on their assigned node ranges.

A process is considered a leader (`+1`) if its node range is equal to the next process's node range, and not a leader (`-1`) if its node range is equal to the previous process's node range. All other processes are marked as `0`. Only leaders are responsible for communication.

# Arguments
- `nprocs::Int`: The total number of processes.
- `noderanges::Vector{UnitRange{Int}}`: A vector specifying the node range assigned to each process.

# Returns
- `leaders::Vector{Int}`: An array where each entry is `1` for a leader, `-1` for a worker, and `0` otherwise.
- `leaderslist::Vector{Int}`: A list of process indices that are leaders and participate in communication.

# Notes
- Communication is performed only between leaders.
"""
function _leaders(nprocs::Int, noderanges::Vector{UnitRange{Int}})
    leaders = zeros(Int, nprocs)
    for i in 1:nprocs
        if i < nprocs
            if noderanges[i] == noderanges[i + 1]
                leaders[i] = 1
            end
        end
        if i > 1
            if noderanges[i] == noderanges[i - 1]
                leaders[i] = -1
            end
        end
    end
    leaderslist = []
    for i in 1:nprocs
        if leaders[i] != -1
            push!(leaderslist, i)
        end
    end
    return leaders, leaderslist
end

"""
multithreadPi(ValueType, f, localdims, Icombined, Jcombined)
Evaluate the function `f` on a grid defined by `Icombined` and `Jcombined` using multiple threads.
"""
function multithreadPi(::Type{ValueType}, f, localdims::Union{Vector{Int},NTuple{N,Int}}, Icombined::Vector{MultiIndex}, Jcombined::Vector{MultiIndex}) where {N,ValueType}
    Pi = zeros(ValueType, length(Icombined), length(Jcombined))
    nthreads = Threads.nthreads()
    chunk_size, remainder = divrem(length(Jcombined), nthreads)
    col_ranges = [((i - 1) * chunk_size + min(i - 1, remainder) + 1):(i * chunk_size + min(i, remainder)) for i in 1:nthreads]
    @threads for tid in 1:nthreads
        if !isempty(col_ranges[tid])
            Pi[:, col_ranges[tid]] = filltensor(
                ValueType, f, localdims,
                Icombined, Jcombined[col_ranges[tid]], Val(0)
            )
        end
    end
    return Pi
end

function parallelfullsearch(::Type{ValueType}, f, tci, Icombined, Jcombined, teamsize, multithread_eval, teamjuliarank, subcomm) where {ValueType}
    if teamsize == 1
        t1 = time_ns()
        if multithread_eval
            Pi = multithreadPi(ValueType, f, tci.localdims, Icombined, Jcombined)
        else
            Pi = reshape(
                filltensor(ValueType, f, tci.localdims,
                Icombined, Jcombined, Val(0)),
                length(Icombined), length(Jcombined)
            )
        end
        t2 = time_ns()
        # For compilation purpouse with MPI
        Icombined_copy = Icombined
        Jcombined_copy = Jcombined
    else # Teamwork
        # Coordinate
        t1 = time_ns()
        Icombined_copy = MPI.bcast([Icombined], subcomm)[1]
        Jcombined_copy = MPI.bcast([Jcombined], subcomm)[1]
        # Subdivide
        chunksize, remainder = divrem(length(Jcombined_copy), teamsize)
        col_chunks = [chunksize + (i <= remainder ? 1 : 0) for i in 1:teamsize]
        row_chunks = [length(Icombined_copy) for _ in 1:teamsize]
        ranges = [(vcat([0],cumsum(col_chunks))[i] + 1):(cumsum(col_chunks)[i]) for i in 1:teamsize]
        Jcombined_copy_local = Jcombined_copy[ranges[teamjuliarank]]
        if !isempty(Jcombined_copy_local)
            tlocalPi = time_ns()
            if multithread_eval
                localPi = multithreadPi(ValueType, f, tci.localdims, Icombined_copy, Jcombined_copy_local)
            else
                localPi = reshape(
                    filltensor(ValueType, f, tci.localdims,
                    Icombined_copy, Jcombined_copy_local, Val(0)),
                    length(Icombined_copy), length(Jcombined_copy_local)
                )
            end
            tlocalPi = (time_ns() - tlocalPi) * 1e-9
        else
            localPi = zeros(length(Icombined_copy), length(Jcombined_copy_local))
        end
        # Unite
        Pi = zeros(length(Icombined_copy), length(Jcombined_copy))
        sizes = vcat(row_chunks', col_chunks')
        counts = vec(prod(sizes, dims=1))
        Pi_vbuf = VBuffer(Pi, counts)
        MPI.Allgatherv!(localPi, Pi_vbuf, subcomm)
        t2 = time_ns()
    end
    return Pi, t1, t2
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
    abstol::Float64=1e-14, #it was 0...
    maxbonddim::Int=typemax(Int),
    sweepdirection::Symbol=:forward,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    extraIset::Vector{MultiIndex}=MultiIndex[],
    extraJset::Vector{MultiIndex}=MultiIndex[],
    estimatedbond::Int=100,
    interbond::Bool=true,
    multithread_eval::Bool=false,
    subcomm=nothing
) where {F,ValueType}
    invalidatesitetensors!(tci)


    if sweepdirection == :parallel && MPI.Initialized()
        comm = MPI.COMM_WORLD
        mpirank = MPI.Comm_rank(comm)
        juliarank = mpirank + 1
        nprocs = MPI.Comm_size(comm)
        noderanges = _noderanges(nprocs, length(tci) - 1, tci.localdims[1], estimatedbond; interbond)
        leaders, leaderslist = _leaders(nprocs, noderanges)
        teamrank = MPI.Comm_rank(subcomm)
        teamjuliarank = teamrank + 1
        teamsize = MPI.Comm_size(subcomm)
    end

    Icombined = union(kronecker(tci.Iset[b], tci.localdims[b]), extraIset)
    Jcombined = union(kronecker(tci.localdims[b+1], tci.Jset[b+1]), extraJset)

    luci = if pivotsearch === :full
        if sweepdirection == :parallel && MPI.Initialized()
            Pi, t1, t2 = parallelfullsearch(ValueType, f, tci, Icombined, Jcombined, teamsize, multithread_eval, teamjuliarank, subcomm)
        else # Serial
            t1 = time_ns()
            if multithread_eval
                Pi = multithreadPi(ValueType, f, tci.localdims, Icombined, Jcombined)
            else
                Pi = reshape(
                    filltensor(ValueType, f, tci.localdims,
                    Icombined, Jcombined, Val(0)),
                    length(Icombined), length(Jcombined)
                )
            end
            t2 = time_ns()
        end
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
        t2 = time_ns()
        if verbosity > 2
            x, y = length(Icombined), length(Jcombined)
            print("    Computing Pi ($x x $y) at bond $b: ")
        end
        if sweepdirection == :parallel && MPI.Initialized()
            I0 = MPI.bcast([I0], subcomm)[1]
            J0 = MPI.bcast([J0], subcomm)[1]
            Icombined_copy = MPI.bcast([Icombined], subcomm)[1]
            Jcombined_copy = MPI.bcast([Jcombined], subcomm)[1]
            Pif = SubMatrix{ValueType}(f, Icombined_copy, Jcombined_copy)
            res = MatrixLUCI(
                ValueType,
                Pif,
                (length(Icombined_copy), length(Jcombined_copy)),
                I0, J0;
                reltol=reltol, abstol=abstol,
                maxrank=maxbonddim,
                leftorthogonal=leftorthogonal,
                pivotsearch=:rook,
                usebatcheval=true,
                leaders=leaders,leaderslist=leaderslist, subcomm=subcomm
            )
            MPI.Barrier(subcomm)
        else # Serial
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
                usebatcheval=true,
            )
        end
        updatemaxsample!(tci, [ValueType(Pif.maxsamplevalue)])

        t3 = time_ns()

        # Fall back to full search if rook search fails
        if npivots(res) == 0
            if sweepdirection == :parallel && MPI.Initialized()
                Pi, _, _ = parallelfullsearch(ValueType, f, tci, Icombined, Jcombined, teamsize, multithread_eval, teamjuliarank, subcomm)
            else # Serial
                t1 = time_ns()
                if multithread_eval
                    Pi = multithreadPi(ValueType, f, tci.localdims, Icombined, Jcombined)
                else
                    Pi = reshape(
                        filltensor(ValueType, f, tci.localdims,
                        Icombined, Jcombined, Val(0)),
                        length(Icombined), length(Jcombined)
                    )
                end
                t2 = time_ns()
            end
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
        #if verbosity > 2
        #    x, y = length(Icombined), length(Jcombined)
        #    println("    Computing Pi ($x x $y) at bond $b: $(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec, fall back to full: $(1e-9*(t4-t3)) sec")
        #end
        if verbosity > 2
            x, y = length(Icombined), length(Jcombined)
            println("$(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec, fall back to full: $(1e-9*(t4-t3)) sec")
        end
        res
    else
        throw(ArgumentError("Unknown pivot search strategy $pivotsearch. Choose from :rook, :full."))
    end
    # Receive after the send
    if sweepdirection == :parallel && MPI.Initialized()
        sreq = MPI.Request[]
        if leaders[juliarank] != -1
            if length(noderanges[juliarank]) == 1 && b == noderanges[juliarank][1] && juliarank != leaderslist[1] && juliarank != leaderslist[end] # If processor has only 1 bond then communicates both left and right
                currentleader = findfirst(isequal(juliarank), leaderslist)
                if currentleader != nothing
                    push!(sreq, MPI.isend([Jcombined[colindices(luci)]], comm; dest=leaderslist[currentleader - 1] - 1, tag=(2*b+1)))
                    push!(sreq, MPI.isend([Icombined[rowindices(luci)]], comm; dest=leaderslist[currentleader + 1] - 1, tag=2*(b+1)))
                else
                    println("Error! Missmatch between leaders and leaderslist, unexpected behaviour! Please open an issue")
                end
            elseif b == noderanges[juliarank][1] && juliarank != leaderslist[1] # processor communicates left
                currentleader = findfirst(isequal(juliarank), leaderslist)
                if currentleader != nothing
                    push!(sreq, MPI.isend([Jcombined[colindices(luci)]], comm; dest=leaderslist[currentleader - 1] - 1, tag=(2*b+1)))
                else
                    println("Error! Missmatch between leaders and leaderslist, unexpected behaviour! Please open an issue")
                end
                tci.Iset[b+1] = Icombined[rowindices(luci)]
            elseif b == noderanges[juliarank][end] && juliarank != leaderslist[end] # processor communicates right
                currentleader = findfirst(isequal(juliarank), leaderslist)
                if currentleader != nothing
                    push!(sreq, MPI.isend([Icombined[rowindices(luci)]], comm; dest=leaderslist[currentleader + 1] - 1, tag=2*(b+1)))
                else
                    println("Error! Missmatch between leaders and leaderslist, unexpected behaviour! Please open an issue")
                end
                tci.Jset[b] = Jcombined[colindices(luci)]
            else # normal updates without MPI communications
                tci.Iset[b+1] = Icombined[rowindices(luci)]
                tci.Jset[b] = Jcombined[colindices(luci)]
            end
        end
    else
        tci.Iset[b+1] = Icombined[rowindices(luci)]
        tci.Jset[b] = Jcombined[colindices(luci)]
    end
    if length(extraIset) == 0 && length(extraJset) == 0 && sweepdirection != :parallel
        setsitetensor!(tci, b, left(luci))
        setsitetensor!(tci, b + 1, right(luci))
    end
    updateerrors!(tci, b, pivoterrors(luci))
    nothing
end

function convergencecriterion(
    ranks::AbstractVector{Int},
    errors::AbstractVector{Float64},
    nglobalpivots::AbstractVector{Int},
    tolerance::Float64,
    maxbonddim::Int,
    ncheckhistory::Int;
    checkconvglobalpivot::Bool=true
)::Bool
    if length(errors) < ncheckhistory
        return false
    end
    lastranks = last(ranks, ncheckhistory)
    lastngpivots = last(nglobalpivots, ncheckhistory)
    return (
        all(last(errors, ncheckhistory) .< tolerance) &&
        (checkconvglobalpivot ? all(lastngpivots .== 0) : true) &&
        minimum(lastranks) == lastranks[end]
    ) || all(lastranks .>= maxbonddim)
end


"""
    GlobalPivotSearchInput(tci::TensorCI2{ValueType}) where {ValueType}

Construct a GlobalPivotSearchInput from a TensorCI2 object.
"""
function GlobalPivotSearchInput(tci::TensorCI2{ValueType}) where {ValueType}
    return GlobalPivotSearchInput{ValueType}(
        tci.localdims,
        TensorTrain(tci),
        tci.maxsamplevalue,
        tci.Iset,
        tci.Jset
    )
end


"""
    function optimize!(
        tci::TensorCI2{ValueType},
        f;
        tolerance::Float64=1e-8,
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
        strictlynested::Bool=false,
        checkbatchevaluatable::Bool=false,
        multithread_eval::Bool=false,
        estimatedbond::Int=100,
        interbond::Bool=true
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
- `sweepstrategy::Symbol` specifies whether to sweep forward (:forward), backward (:backward), back and forth (:backandforth) or in parallel (:parallel) during optimization. Default: `:backandforth`.
- `pivotsearch::Symbol` determins how pivots are searched (`:full` or `:rook`). Default: `:full`.
- `verbosity::Int` can be set to `>= 1` to get convergence information on standard output during optimization. Default: `0`.
- `loginterval::Int` can be set to `>= 1` to specify how frequently to print convergence information. Default: `10`.
- `normalizeerror::Bool` determines whether to scale the error by the maximum absolute value of `f` found during sampling. If set to `false`, the algorithm continues until the *absolute* error is below `tolerance`. If set to `true`, the algorithm uses the absolute error divided by the maximum sample instead. This is helpful if the magnitude of the function is not known in advance. Default: `true`.
- `ncheckhistory::Int` is the number of history points to use for convergence checks. Default: `3`.
- `globalpivotfinder::Union{AbstractGlobalPivotFinder, Nothing}` is a global pivot finder to use for searching global pivots. Default: `nothing`. If `nothing`, a default global pivot finder is used.
- `maxnglobalpivot::Int` can be set to `>= 0`. Default: `5`. The maximum number of global pivots to add in each iteration.
- `strictlynested::Bool` determines whether to preserve partial nesting in the TCI algorithm. Default: `false`.
- `checkbatchevaluatable::Bool` Check if the function `f` is batch evaluatable. Default: `false`.
- `multithread_eval::Bool` determines whether to use multithreading for function evaluation. Default: `false`.
- `estimatedbond::Int` is the estimated bond dimension after compression, set by the user to help balance the workload. Default: `100`.
- `interbond::Bool` determines whether to distribute the workload across bonds or not. Default: `true`.
- `checkconvglobalpivot::Bool` Check if the global pivot finder is converged. Default: `true`. In the future, this will be set to `false` by default.

Arguments (deprecated):
- `pivottolerance::Float64` is the tolerance for the pivot search. Deprecated.
- `nsearchglobalpivot::Int` is the number of search points for the global pivot finder. Deprecated.
- `tolmarginglobalsearch::Float64` is the tolerance for the global pivot finder. Deprecated.

Notes:
- Set `tolerance` to be > 0 or `maxbonddim` to some reasonable value. Otherwise, convergence is not reachable.
- By default, no caching takes place. Use the [`CachedFunction`](@ref) wrapper if your function is expensive to evaluate.


See also: [`crossinterpolate2`](@ref), [`optfirstpivot`](@ref), [`CachedFunction`](@ref), [`crossinterpolate1`](@ref)
"""
function optimize!(
    tci::TensorCI2{ValueType},
    f;
    tolerance::Union{Float64, Nothing}=nothing,
    pivottolerance::Union{Float64, Nothing}=nothing,
    maxbonddim::Int=typemax(Int),
    maxiter::Int=20,
    sweepstrategy::Symbol=:backandforth,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    loginterval::Int=10,
    normalizeerror::Bool=true,
    ncheckhistory::Int=3,
    globalpivotfinder::Union{AbstractGlobalPivotFinder, Nothing}=nothing,
    maxnglobalpivot::Int=5,
    nsearchglobalpivot::Int=5,
    tolmarginglobalsearch::Float64=10.0,
    strictlynested::Bool=false,
    checkbatchevaluatable::Bool=false,
    multithread_eval::Bool=false,
    estimatedbond::Int=100,
    interbond::Bool=true,
    checkconvglobalpivot::Bool=true
) where {ValueType}
    errors = Float64[]
    ranks = Int[]
    nglobalpivots = Int[]
    local tol::Float64

    if checkbatchevaluatable && !(f isa BatchEvaluator)
        error("Function `f` is not batch evaluatable")
    end

    if nsearchglobalpivot > 0 && nsearchglobalpivot < maxnglobalpivot
        error("nsearchglobalpivot < maxnglobalpivot!")
    end

    # Deprecate the pivottolerance option
    if !isnothing(pivottolerance)
        if !isnothing(tolerance) && (tolerance != pivottolerance)
            throw(ArgumentError("Got different values for pivottolerance and tolerance in optimize!(TCI2). For TCI2, both of these options have the same meaning. Please assign only `tolerance`."))
        else
            @warn "The option `pivottolerance` of `optimize!(tci::TensorCI2, f)` is deprecated. Please update your code to use `tolerance`, as `pivottolerance` will be removed in the future."
            tol = pivottolerance
        end
    elseif !isnothing(tolerance)
        tol = tolerance
    else # pivottolerance == tolerance == nothing, therefore set tol to default value
        tol = 1e-8
    end

    tstart = time_ns()

    if maxbonddim >= typemax(Int) && tol <= 0
        throw(ArgumentError(
            "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence criterion is not reachable!"
        ))
    end

    if (sweepstrategy == :parallel) && !MPI.Initialized()
        println("Warning! Parallel strategy has been chosen, but MPI is not initialized, please use initializempi() before using crossinterpolate2()/quanticscrossinterpolate() and use finalizempi() afterwards")
    end

    if sweepstrategy == :parallel && MPI.Initialized()
        comm = MPI.COMM_WORLD
        mpirank = MPI.Comm_rank(comm)
        juliarank = mpirank + 1
	    nprocs = MPI.Comm_size(comm)
        noderanges = _noderanges(nprocs, length(tci) - 1, tci.localdims[1], estimatedbond; interbond)
        leaders, leaderslist = _leaders(nprocs, noderanges)
    end
    # Create the global pivot finder
    finder = if isnothing(globalpivotfinder)
        DefaultGlobalPivotFinder(
            nsearch=nsearchglobalpivot,
            maxnglobalpivot=maxnglobalpivot,
            tolmarginglobalsearch=tolmarginglobalsearch
        )
    else
        globalpivotfinder
    end

    globalpivots = MultiIndex[]
    for iter in 1:maxiter
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        abstol = tol * errornormalization;

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: starting 2site sweep")
            flush(stdout)
        end

        sweep2site!(
            tci, f, 2;
            iter1 = 1,
            abstol=abstol,
            maxbonddim=maxbonddim,
            pivotsearch=pivotsearch,
            strictlynested=strictlynested,
            verbosity=verbosity,
            sweepstrategy=sweepstrategy,
            fillsitetensors=true,
            estimatedbond=estimatedbond,
            interbond=interbond,
            multithread_eval=multithread_eval
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
        if sweepstrategy != :parallel
            input = GlobalPivotSearchInput(tci)
            globalpivots = finder(
                input, f, abstol;
                verbosity=verbosity,
                rng=Random.default_rng()
            )
            addglobalpivots!(tci, globalpivots)
            push!(nglobalpivots, length(globalpivots))
        end

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: done searching global pivots")
            flush(stdout)
        end

        push!(ranks, rank(tci))
        if verbosity > 0 && mod(iter, loginterval) == 0
            println("iteration = $iter, rank = $(last(ranks)), error= $(last(errors)), maxsamplevalue= $(tci.maxsamplevalue), nglobalpivot=$(length(globalpivots))")
            flush(stdout)
        end
        # Checks if every processor has converged with its own subtrain
        if sweepstrategy == :parallel && MPI.Initialized()
            converged = leaders[juliarank] == -1 || convergencecriterion(ranks, errors, nglobalpivots, abstol, maxbonddim, ncheckhistory; checkconvglobalpivot=checkconvglobalpivot)
            MPI.Barrier(comm)
            global_converged = MPI.Allreduce(converged, MPI.LAND, comm)
            if global_converged # If all the processors have converged
                break
            end
        elseif convergencecriterion(
            ranks, errors, nglobalpivots, abstol, maxbonddim, ncheckhistory; checkconvglobalpivot=checkconvglobalpivot
        )
            break
        end
    end

    # After convergence
    # Allgatherv! not possible given type constraints
    if sweepstrategy == :parallel && MPI.Initialized()
        if juliarank == 1 # If we are the first processor, we take all the information from the other processes
            for leader in leaderslist[2:end]
                for b in noderanges[leader]
                    tci.Iset[b] = MPI.recv(comm; source=leader - 1, tag=2*(b))[1]
                    tci.Jset[b+1] = MPI.recv(comm; source=leader - 1, tag=2*(b+1)+1)[1]
                    if leader == leaderslist[end]
                        tci.Iset[length(tci)] = MPI.recv(comm; source=leader - 1, tag=2*(length(tci)+1)+1)[1]
                    end
                end
            end
        else # Other processors pass the infomation to the first one
            if leaders[juliarank] != -1
                for b in noderanges[juliarank]
                    MPI.send([tci.Iset[b]], comm; dest=0, tag=2*(b))
                    MPI.send([tci.Jset[b+1]], comm; dest=0, tag=2*(b+1)+1)
                    if juliarank == leaderslist[end]
                        MPI.send([tci.Iset[length(tci)]], comm; dest=0, tag=2*(length(tci)+1)+1)
                    end
                end
            end
        end
        # BCast to all, so that all nodes have full knowledge
        tci.Iset = MPI.bcast(tci.Iset, comm)
        tci.Jset = MPI.bcast(tci.Jset, comm)
    end

    # Extra one sweep by the 1-site update to
    #  (1) Remove unnecessary pivots added by global pivots
    #      Note: a pivot matrix can be non-square after adding global pivots,
    #            or the bond dimension exceeds maxbonddim
    #  (2) Compute site tensors
    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    abstol = tol * errornormalization;
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
    strictlynested::Bool=false,
    fillsitetensors::Bool=true,
    estimatedbond::Int=100,
    interbond::Bool=true,
    multithread_eval::Bool=false
) where {ValueType}
    invalidatesitetensors!(tci)

    n = length(tci)

    if sweepstrategy == :parallel && MPI.Initialized()
        comm = MPI.COMM_WORLD
        mpirank = MPI.Comm_rank(comm)
        juliarank = mpirank + 1
        nprocs = MPI.Comm_size(comm)
    end

    for iter in iter1:iter1+niter-1
        extraIset = [MultiIndex[] for _ in 1:n]
        extraJset = [MultiIndex[] for _ in 1:n]
        if !strictlynested && length(tci.Iset_history) > 0
            extraIset = tci.Iset_history[end]
            extraJset = tci.Jset_history[end]
        end

        push!(tci.Iset_history, deepcopy(tci.Iset))
        push!(tci.Jset_history, deepcopy(tci.Jset))

        flushpivoterror!(tci)
        if sweepstrategy == :parallel && MPI.Initialized()
            noderanges = _noderanges(nprocs, length(tci) - 1, tci.localdims[1], estimatedbond; interbond)
            leaders, leaderslist = _leaders(nprocs, noderanges)
            colors = zeros(Int,nprocs)
            count = -1
            for i in 1:length(colors)
                if leaders[i] != -1
                    count += 1
                end
                colors[i] = count
            end
            color = colors[juliarank]
            subcomm = MPI.Comm_split(comm, color, mpirank)
            for bondindex in noderanges[juliarank]
                updatepivots!(
                    tci, bondindex, f, true;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    sweepdirection=:parallel,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    extraIset=extraIset[bondindex+1],
                    extraJset=extraJset[bondindex],
                    estimatedbond=estimatedbond,
                    interbond=interbond,
                    multithread_eval=multithread_eval,
                    subcomm = subcomm
                )
            end
            for b in noderanges[juliarank]
                if leaders[juliarank] != -1
                    if b == noderanges[juliarank][1] && juliarank != leaderslist[1]
                        currentleader = findfirst(isequal(juliarank), leaderslist)
                        if currentleader != nothing
                            tci.Iset[b] = MPI.recv(comm; source=leaderslist[currentleader - 1] - 1, tag=2*b)[1]
                        else
                            println("Error! Missmatch between leaders and leaderslist, unexpected behaviour! Please open an issue")
                        end
                    end
                    if b == noderanges[juliarank][end] && juliarank != leaderslist[end]
                        currentleader = findfirst(isequal(juliarank), leaderslist)
                        if currentleader != nothing
                            tci.Jset[b + 1] = MPI.recv(comm; source=leaderslist[currentleader + 1] - 1, tag=2*(b + 1) + 1)[1]
                        else
                            println("Error! Missmatch between leaders and leaderslist, unexpected behaviour! Please open an issue")
                        end
                    end
                end
            end
        elseif forwardsweep(sweepstrategy, iter) # forward sweep
            for bondindex in 1:n-1
                updatepivots!(
                    tci, bondindex, f, true;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    sweepdirection=:forward,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    extraIset=extraIset[bondindex+1],
                    extraJset=extraJset[bondindex],
                    multithread_eval=multithread_eval
                )
            end
        else # backward sweep
            for bondindex in (n-1):-1:1
                updatepivots!(
                    tci, bondindex, f, false;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    sweepdirection=:backward,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    extraIset=extraIset[bondindex+1],
                    extraJset=extraJset[bondindex],
                    multithread_eval=multithread_eval
                )
            end
        end
    end

    if fillsitetensors && sweepstrategy != :parallel
        fillsitetensors!(tci, f)
    end
    nothing
end

@doc raw"""
    function crossinterpolate2(
        ::Type{ValueType},
        f,
        localdims::Union{Vector{Int},NTuple{N,Int}},
        initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
        kwargs...
    ) where {ValueType,N}

Cross interpolate a function ``f(\mathbf{u})`` using the TCI2 algorithm. Here, the domain of ``f`` is ``\mathbf{u} \in [1, \ldots, d_1] \times [1, \ldots, d_2] \times \ldots \times [1, \ldots, d_{\mathscr{L}}]`` and ``d_1 \ldots d_{\mathscr{L}}`` are the local dimensions.

Arguments:
- `ValueType` is the return type of `f`. Automatic inference is too error-prone.
- `f` is the function to be interpolated. `f` should have a single parameter, which is a vector of the same length as `localdims`. The return type should be `ValueType`.
- `localdims::Union{Vector{Int},NTuple{N,Int}}` is a `Vector` (or `Tuple`) that contains the local dimension of each index of `f`.
- `initialpivots::Vector{MultiIndex}` is a vector of pivots to be used for initialization. Default: `[1, 1, ...]`.

Refer to [`optimize!`](@ref) for other keyword arguments such as `tolerance`, `maxbonddim`, `maxiter`.

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
    nsearch::Int = 100,
    maxnglobalpivot::Int = 5
)::Vector{MultiIndex} where {ValueType}
    if nsearch == 0 || maxnglobalpivot == 0
        return MultiIndex[]
    end

    if !issitetensorsavailable(tci)
        fillsitetensors!(tci, f)
    end

    pivots = Dict{Float64,MultiIndex}()
    ttcache = TTCache(tci)
    for _ in 1:nsearch
        pivot, error = _floatingzone(ttcache, f; earlystoptol = 10 * abstol, nsweeps=100)
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

    return [p for (_,p) in pivots]
end