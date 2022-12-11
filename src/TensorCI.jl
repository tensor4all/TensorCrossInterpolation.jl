# It is necessary to nest vectors here, as these vectors will frequently grow
# by adding elements. Using matrices, the entire matrix would have to be copied.
# These type definitions are mostly to make the code below more readable, as
# Vector{Vector{Vector{...}}} constructions make it difficult to understand
# the meaning of each dimension.
const LocalIndex = Int
const MultiIndex = Vector{LocalIndex}

module SweepStrategies
@enum SweepStrategy begin
    back_and_forth
    forward
    backward
end
end

mutable struct TensorCI{ValueType}
    f::CachedFunction{MultiIndex,ValueType}

    Iset::Vector{IndexSet{MultiIndex}}
    Jset::Vector{IndexSet{MultiIndex}}
    localSet::Vector{Vector{Int}}

    """
    The 3-leg ``T`` tensors from the TCI paper. First and last leg are links to 
    adjacent ``P^{-1}`` matrices, and the second leg is the local index. The
    outermost vector iterates over the site index ``p``, such that
    `T[p]` is the ``p``-th site tensor.
    """
    T::Vector{Array{ValueType,3}}

    """
    The 2-leg ``P`` tensors from the TCI paper.
    """
    P::Vector{Matrix{ValueType}}

    """
    The 4-leg ``\\Pi`` matrices from the TCI paper, who are decomposed to
    obtain ``T`` and ``P``. Have to be kept in memory for efficient updates.
    """
    Pi::Vector{Matrix{ValueType}}
    Pi_Iset::Vector{IndexSet{MultiIndex}}
    Pi_Jset::Vector{IndexSet{MultiIndex}}

    """
    Pivot errors at each site index ``p``.
    """
    pivot_errors::Vector{ValueType}

    function TensorCI(
        f::CachedFunction{MultiIndex,ValueType},
        localdims::Vector{Int}
    ) where {ValueType}
        n = length(localdims)
        new{ValueType}(
            f,
            [IndexSet{MultiIndex}() for _ in 1:n],
            [IndexSet{MultiIndex}() for _ in 1:n],
            [collect(1:d) for d in localdims],
            [zeros(0, d, 0) for d in localdims],
            [zeros(0, 0) for _ in 1:n],
            [zeros(0, 0) for _ in 1:n],
            [IndexSet{MultiIndex}() for _ in 1:n],
            [IndexSet{MultiIndex}() for _ in 1:n],
            fill(Inf, n - 1)
        )
    end
end

function Base.show(io::IO, tci::TensorCI{ValueType}) where {ValueType}
    print(io, "$(typeof(tci)) with ranks $(rank(tci))")
end

function TensorCI(
    f::CachedFunction{MultiIndex,ValueType},
    localdim::Int,
    length::Int
) where {ValueType}
    return TensorCI(f, fill(localdim, length))
end

function TensorCI(
    f::CachedFunction{MultiIndex,ValueType},
    localdims::Vector{Int},
    firstpivot::Vector{Int}
) where {ValueType}
    tci = TensorCI(f, localdims)

    firstpivotval = f(firstpivot)
    if firstpivotval == 0
        throw(ArgumentError("Please provide a first pivot where f(pivot) != 0."))
    end
    if length(localdims) != length(firstpivot)
        throw(ArgumentError("Firstpivot and localdims must have same length."))
    end

    n = length(localdims)
    tci.localSet = [collect(1:localdim) for localdim in localdims]
    tci.Iset = [IndexSet([firstpivot[1:p-1]]) for p in 1:n]
    tci.Jset = [IndexSet([firstpivot[p+1:end]]) for p in 1:n]
    tci.Pi_Iset = [getPiIsetAt(tci, p) for p in 1:n]
    tci.Pi_Jset = [getPiJsetAt(tci, p) for p in 1:n]
    tci.Pi = [buildPiAt(tci, p) for p in 1:n-1]

    for p in 1:n-1
        cross = MatrixCI(
            tci.Pi[p],
            (pos(tci.Pi_Iset[p], tci.Iset[p+1][1]),
                pos(tci.Pi_Jset[p+1], tci.Jset[p][1])))
        if p == 1
            updateT!(tci, 1, cross.pivot_cols)
        end
        updateT!(tci, p + 1, cross.pivot_rows)
        tci.P[p] = pivot_matrix(cross)
    end
    tci.P[end] = ones(ValueType, 1, 1)

    return tci
end

function Base.length(tci::TensorCI{V}) where {V}
    return length(tci.localSet)
end

function Base.broadcastable(tci::TensorCI{V}) where {V}
    return Ref(tci)
end

function rank(tci::TensorCI{V}) where {V}
    return [size(t, 1) for t in tci.T[2:end]]
end

function lastSweepPivotError(tci::TensorCI{V}) where {V}
    return maximum(tci.pivot_errors)
end

function TtimesPinv(tci::TensorCI{V}, p::Int) where {V}
    T = tci.T[p]
    shape = size(T)
    TPinv = AtimesBinv(reshape(T, (shape[1] * shape[2], shape[3])), tci.P[p])
    return reshape(TPinv, shape)
end

function PinvtimesT(tci::TensorCI{V}, p::Int) where {V}
    T = tci.T[p]
    shape = size(T)
    PinvT = AinvtimesB(tci.P[p-1], reshape(shape[1], shape[2] * shape[3]))
    return reshape(PinvT, shape)
end

function tensortrain(tci::TensorCI{V}) where {V}
    return TtimesPinv.(tci, 1:length(tci))
end

function evaluateTCI(
    tci::TensorCI{V}, indexset::AbstractVector{LocalIndex}
) where {V}
    return prod(
        AtimesBinv(tci.T[p][:, indexset[p], :], tci.P[p])
        for p in 1:length(tci))[1, 1]
end

function getPiIsetAt(tci::TensorCI{V}, p::Int) where {V}
    return IndexSet([
        [is..., ups] for is in tci.Iset[p].from_int, ups in tci.localSet[p]
    ][:])
end

function getPiJsetAt(tci::TensorCI{V}, p::Int) where {V}
    return IndexSet([
        [up1s, js...] for up1s in tci.localSet[p], js in tci.Jset[p].from_int
    ][:])
end

"""
    buildPiAt(tci::TensorCrossInterpolation{V}, p::Int)

Build a 4-legged ``\\Pi`` tensor at site `p`. Indices are in the order
``i, u_p, u_{p + 1}, j``, as in the TCI paper.
"""
function buildPiAt(tci::TensorCI{V}, p::Int) where {V}
    iset = tci.Pi_Iset[p]
    jset = tci.Pi_Jset[p+1]
    return [tci.f([is..., js...]) for is in iset.from_int, js in jset.from_int]
end

function buildCrossAt(tci::TensorCI{V}, p::Int) where {V}
    # Translate to order at site p from adjacent sites.
    iset = pos(tci.Pi_Iset[p], tci.Iset[p+1].from_int)
    jset = pos(tci.Pi_Jset[p+1], tci.Jset[p].from_int)
    shape = size(tci.T[p])
    Tp = reshape(tci.T[p], (shape[1] * shape[2], shape[3]))
    shape1 = size(tci.T[p+1])
    Tp1 = reshape(tci.T[p+1], (shape1[1], shape1[2] * shape1[3]))
    return MatrixCI(iset, jset, Tp, Tp1)
end

function updateT!(
    tci::TensorCI{V},
    p::Int,
    new_T::AbstractArray{V}
) where {V}
    tci.T[p] = reshape(
        new_T,
        length(tci.Iset[p]),
        length(tci.localSet[p]),
        length(tci.Jset[p]))
end

function addPivotAt!(
    tci::TensorCI{V},
    p::Int,
    tolerance::V=V()
) where {V}
    if (p < 1) || (p > length(tci) - 1)
        throw(BoundsError(
            "Pi tensors can only be built at sites 1 to length - 1 = $(length(tci) - 1)."))
    end

    cross = buildCrossAt(tci, p)

    if rank(cross) >= minimum(size(tci.Pi[p]))
        tci.pivot_errors[p] = 0.0
        return
    end

    newpivot, newerror = find_new_pivot(tci.Pi[p], cross)
    tci.pivot_errors[p] = newerror
    if newerror < tolerance
        return
    end

    # Update T, P, T that formed Pi
    add_pivot!(tci.Pi[p], cross, newpivot)
    push!(tci.Iset[p+1], tci.Pi_Iset[p][newpivot[1]])
    push!(tci.Jset[p], tci.Pi_Jset[p+1][newpivot[2]])
    updateT!(tci, p, cross.pivot_cols)
    updateT!(tci, p + 1, cross.pivot_rows)
    tci.P[p] = pivot_matrix(cross)

    # Update adjacent Pi matrices, since shared Ts have changed
    # TODO: re-use existing data in old Pi matrices
    if p < length(tci) - 1
        tci.Pi_Iset[p+1] = getPiIsetAt(tci, p + 1)
        tci.Pi[p+1] = buildPiAt(tci, p + 1)
    end

    if p > 1
        tci.Pi_Jset[p] = getPiJsetAt(tci, p)
        tci.Pi[p-1] = buildPiAt(tci, p - 1)
    end
end

function cross_interpolate(
    f::CachedFunction{MultiIndex,ValueType},
    localdims::Vector{Int},
    firstpivot::MultiIndex=ones(Int, length(localdims));
    tolerance::Float64=1e-8,
    maxiter::Int=200,
    sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.back_and_forth
) where {ValueType}
    tci = TensorCI(f, localdims, firstpivot)
    n = length(tci)
    errors = []
    ranks = []

    # Start at two, because the constructor already added a pivot everywhere.
    for iter in 2:maxiter
        foward_sweep = (
            sweepstrategy == SweepStrategies.forward ||
            (sweepstrategy != SweepStrategies.backward && isodd(iter))
        )

        if foward_sweep
            addPivotAt!.(tci, 1:n-1, tolerance)
        else
            addPivotAt!.(tci, (n-1):-1:1, tolerance)
        end

        push!(errors, lastSweepPivotError(tci))
        push!(ranks, maximum(rank(tci)))
        if last(errors) < tolerance
            break
        end
    end

    return tci, ranks, errors
end

function cross_interpolate(
    f::Function,
    localdims::Vector{Int},
    firstpivot::MultiIndex=ones(Int, length(localdims));
    tolerance::Float64=1e-8,
    maxiter::Int=200,
    sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.back_and_forth
)
    firstpivotval = f(firstpivot)
    cf = CachedFunction(f, Dict(firstpivot => firstpivotval))
    return cross_interpolate(
        cf,
        localdims,
        firstpivot,
        tolerance=tolerance,
        maxiter=maxiter,
        sweepstrategy=sweepstrategy
    )
end
