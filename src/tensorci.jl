# It is necessary to nest vectors here, as these vectors will frequently grow
# by adding elements. Using matrices, the entire matrix would have to be copied.
# These type definitions are mostly to make the code below more readable, as
# Vector{Vector{Vector{...}}} constructions make it difficult to understand
# the meaning of each dimension.
const LocalIndex = Int
const MultiIndex = Vector{LocalIndex}

"""
    module SweepStrategies

Wraps the enum SweepStrategy, such that sweep strategies are available as
`SweepStrategies.forward`, etc.
"""
module SweepStrategies
@enum SweepStrategy begin
    back_and_forth
    forward
    backward
end
end

"""
    mutable struct TensorCI{ValueType}

An object that represents a tensor cross interpolation. Users may want to create these using
`crossinterpolate(...)` rather than calling a constructor directly.
"""
mutable struct TensorCI{ValueType}
    f::CachedFunction{MultiIndex,ValueType}

    Iset::Vector{IndexSet{MultiIndex}}
    Jset::Vector{IndexSet{MultiIndex}}
    localset::Vector{Vector{Int}}

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
    PiIset::Vector{IndexSet{MultiIndex}}
    PiJset::Vector{IndexSet{MultiIndex}}

    """
    Pivot errors at each site index ``p``.
    """
    pivoterrors::Vector{Float64}

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
    tci.localset = [collect(1:localdim) for localdim in localdims]
    tci.Iset = [IndexSet([firstpivot[1:p-1]]) for p in 1:n]
    tci.Jset = [IndexSet([firstpivot[p+1:end]]) for p in 1:n]
    tci.PiIset = [getPiIset(tci, p) for p in 1:n]
    tci.PiJset = [getPiJset(tci, p) for p in 1:n]
    tci.Pi = [getPi(tci, p) for p in 1:n-1]

    for p in 1:n-1
        cross = MatrixCI(
            tci.Pi[p],
            (pos(tci.PiIset[p], tci.Iset[p+1][1]),
                pos(tci.PiJset[p+1], tci.Jset[p][1])))
        if p == 1
            updateT!(tci, 1, cross.pivotcols)
        end
        updateT!(tci, p + 1, cross.pivotrows)
        tci.P[p] = pivotmatrix(cross)
    end
    tci.P[end] = ones(ValueType, 1, 1)

    return tci
end

function Base.length(tci::TensorCI{V}) where {V}
    return length(tci.localset)
end

function Base.broadcastable(tci::TensorCI{V}) where {V}
    return Ref(tci)
end

"""
    function linkdims(tci::TensorCI{V}) where {V}

Bond dimensions along the links between ``T`` and ``P`` matrices in the TCI.
"""
function linkdims(tci::TensorCI{V}) where {V}
    return [size(t, 1) for t in tci.T[2:end]]
end

"""
    function rank(tci::TensorCI{V}) where {V}

Maximum link dimension, which is approximately the rank of the tensor being cross
interpolated.
"""
function rank(tci::TensorCI{V}) where {V}
    return maximum(linkdims(tci))
end

function lastsweeppivoterror(tci::TensorCI{V}) where {V}
    return maximum(tci.pivoterrors)
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

"""
    function tensortrain(tci::TensorCI{V}) where {V}

Convert the TCI object into a tensor train, i.e. an MPS. Returns an Array of 3-leg tensors,
where the first and last leg connect to neighboring tensors and the second leg is the local
site index.
"""
function tensortrain(tci::TensorCI{V}) where {V}
    return TtimesPinv.(tci, 1:length(tci))
end

"""
    function evaluate(tci::TensorCI{V}, indexset::AbstractVector{LocalIndex}) where {V}

Evaluate the TCI at a specific set of indices. This method is inefficient; to evaluate
many points, first convert the TCI object into a tensor train using `tensortrain(tci)`.
"""
function evaluate(tci::TensorCI{V}, indexset::AbstractVector{LocalIndex}) where {V}
    return prod(
        AtimesBinv(tci.T[p][:, indexset[p], :], tci.P[p])
        for p in 1:length(tci))[1, 1]
end

function getPiIset(tci::TensorCI{V}, p::Int) where {V}
    return IndexSet([
        [is..., ups] for is in tci.Iset[p].fromint, ups in tci.localset[p]
    ][:])
end

function getPiJset(tci::TensorCI{V}, p::Int) where {V}
    return IndexSet([
        [up1s, js...] for up1s in tci.localset[p], js in tci.Jset[p].fromint
    ][:])
end

"""
    buildPiAt(tci::TensorCrossInterpolation{V}, p::Int)

Build a 4-legged ``\\Pi`` tensor at site `p`. Indices are in the order
``i, u_p, u_{p + 1}, j``, as in the TCI paper.
"""
function getPi(tci::TensorCI{V}, p::Int) where {V}
    iset = tci.PiIset[p]
    jset = tci.PiJset[p+1]
    return [tci.f([is..., js...]) for is in iset.fromint, js in jset.fromint]
end

function getcross(tci::TensorCI{V}, p::Int) where {V}
    # Translate to order at site p from adjacent sites.
    iset = pos(tci.PiIset[p], tci.Iset[p+1].fromint)
    jset = pos(tci.PiJset[p+1], tci.Jset[p].fromint)
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
        length(tci.localset[p]),
        length(tci.Jset[p]))
end

function updatePirows!(tci::TensorCI{V}, p::Int) where {V}
    newIset = getPiIset(tci, p)
    diffIset = setdiff(newIset.fromint, tci.PiIset[p].fromint)
    newPi = Matrix{V}(undef, length(newIset), size(tci.Pi[p], 2))
    for (oldi, imulti) in enumerate(tci.PiIset[p].fromint)
        newi = pos(newIset, imulti)
        newPi[newi, :] = tci.Pi[p][oldi, :]
    end
    for imulti in diffIset
        newi = pos(newIset, imulti)
        newPi[newi, :] = [tci.f([imulti..., js...]) for js in tci.PiJset[p+1].fromint]
    end
    tci.Pi[p] = newPi
    tci.PiIset[p] = newIset
end

function updatePicols!(tci::TensorCI{V}, p::Int) where {V}
    newJset = getPiJset(tci, p + 1)
    diffJset = setdiff(newJset.fromint, tci.PiJset[p+1].fromint)
    newPi = Matrix{V}(undef, size(tci.Pi[p], 1), length(newJset))
    for (oldj, jmulti) in enumerate(tci.PiJset[p+1].fromint)
        newj = pos(newJset, jmulti)
        newPi[:, newj] = tci.Pi[p][:, oldj]
    end
    for jmulti in diffJset
        newj = pos(newJset, jmulti)
        newPi[:, newj] = [tci.f([is..., jmulti...]) for is in tci.PiIset[p].fromint]
    end
    tci.Pi[p] = newPi
    tci.PiJset[p+1] = newJset
end

"""
    function addpivot!(tci::TensorCI{V}, p::Int, tolerance::T=T()) where {V,T<:Real}

Add a pivot to the TCI at site `p`. Do not add a pivot if the error is below tolerance.
"""
function addpivot!(tci::TensorCI{V}, p::Int, tolerance::T=T()) where {V,T<:Real}
    if (p < 1) || (p > length(tci) - 1)
        throw(BoundsError(
            "Pi tensors can only be built at sites 1 to length - 1 = $(length(tci) - 1)."))
    end

    cross = getcross(tci, p)

    if rank(cross) >= minimum(size(tci.Pi[p]))
        tci.pivoterrors[p] = 0.0
        return
    end

    newpivot, newerror = findnewpivot(tci.Pi[p], cross)
    tci.pivoterrors[p] = newerror
    if newerror < tolerance
        return
    end

    # Update T, P, T that formed Pi
    addpivot!(tci.Pi[p], cross, newpivot)
    push!(tci.Iset[p+1], tci.PiIset[p][newpivot[1]])
    push!(tci.Jset[p], tci.PiJset[p+1][newpivot[2]])
    updateT!(tci, p, cross.pivotcols)
    updateT!(tci, p + 1, cross.pivotrows)
    tci.P[p] = pivotmatrix(cross)

    # Update adjacent Pi matrices, since shared Ts have changed
    if p < length(tci) - 1
        updatePirows!(tci, p + 1)
    end

    if p > 1
        updatePicols!(tci, p - 1)
    end
end

"""
    function crossinterpolate(
        f::CachedFunction{MultiIndex,ValueType},
        localdims::Vector{Int},
        firstpivot::MultiIndex=ones(Int, length(localdims));
        tolerance::Float64=1e-8,
        maxiter::Int=200,
        sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.back_and_forth
    ) where {ValueType}

Cross interpolate a function `f`, which can be called with a single argument `u`, where `u`
is an array of the same length as `localdims` and `firstpivot` with each component `u[p]`
between 1 and `localdims[p]`.

Arguments:
- `f::CachedFunction{MultiIndex,ValueType}` is the function to be interpolated, as `CachedFunction` object.
- `localdims::Vector{Int}` is a Vector that contains the local dimension of each external leg.
- `tolerance::Float64` is a float specifying the tolerance for the interpolation. Default: `1e-8`.
- `maxiter::Int` is the maximum number of iterations (i.e. optimization sweeps) before aborting the TCI construction. Default: `200`.
- `sweepstrategy::SweepStrategies.SweepStrategy` specifies whether to sweep forward, backward, or back and forth during optimization. Default: `SweepStrategies.back_and_forth`.
- `pivottolerance::Float64` specifies the tolerance below which no new pivot will be added to each tensor. Default: `1e-12`.
- `errornormalization::Float64` is the value by which the error will be normalized. Set this to `1.0` to get a pure absolute error; set this to `f(firstpivot)` to get the same behaviour as in the C++ library *xfac*. Default: `f(firstpivot)`.
"""
function crossinterpolate(
    f::CachedFunction{MultiIndex,ValueType},
    localdims::Vector{Int},
    firstpivot::MultiIndex=ones(Int, length(localdims));
    tolerance::Float64=1e-8,
    maxiter::Int=200,
    sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.back_and_forth,
    pivottolerance::Float64=1e-12,
    errornormalization=nothing
) where {ValueType}
    tci = TensorCI(f, localdims, firstpivot)
    n = length(tci)
    errors = Float64[]
    ranks = Int[]
    N::Float64 = isnothing(errornormalization) ? f(firstpivot) : 1.0

    # Start at two, because the constructor already added a pivot everywhere.
    for iter in 2:maxiter
        foward_sweep = (
            sweepstrategy == SweepStrategies.forward ||
            (sweepstrategy != SweepStrategies.backward && isodd(iter))
        )

        if foward_sweep
            addpivot!.(tci, 1:n-1, pivottolerance)
        else
            addpivot!.(tci, (n-1):-1:1, pivottolerance)
        end

        push!(errors, lastsweeppivoterror(tci) / N)
        push!(ranks, maximum(rank(tci)))
        if last(errors) < tolerance
            break
        end
    end

    return tci, ranks, errors
end

"""
    function crossinterpolate(
        f::Function,
        localdims::Vector{Int},
        firstpivot::MultiIndex=ones(Int, length(localdims));
        tolerance::Float64=1e-8,
        maxiter::Int=200,
        sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.back_and_forth
    ) where {ValueType}

Cross interpolate a function `f`, which can be called with a single argument `u`, where `u`
is an array of the same length as `localdims` and `firstpivot` with each component `u[p]`
between 1 and `localdims[p]`.

Arguments:
- `f::Function` is the function to be interpolated.
- `localdims::Vector{Int}` is a Vector that contains the local dimension of each external leg.
- `tolerance::Float64` is a float specifying the tolerance for the interpolation. Default: `1e-8`.
- `maxiter::Int` is the maximum number of iterations (i.e. optimization sweeps) before aborting the TCI construction. Default: `200`.
- `sweepstrategy::SweepStrategies.SweepStrategy` specifies whether to sweep forward, backward, or back and forth during optimization. Default: `SweepStrategies.back_and_forth`.
- `pivottolerance::Float64` specifies the tolerance below which no new pivot will be added to each tensor. Default: `1e-12`.
- `errornormalization::Float64` is the value by which the error will be normalized. Set this to `1.0` to get a pure absolute error; set this to `f(firstpivot)` to get the same behaviour as in the C++ library *xfac*. Default: `f(firstpivot)`.
"""
function crossinterpolate(
    f::Function,
    localdims::Vector{Int},
    firstpivot::MultiIndex=ones(Int, length(localdims));
    tolerance::Float64=1e-8,
    maxiter::Int=200,
    sweepstrategy::SweepStrategies.SweepStrategy=SweepStrategies.back_and_forth,
    pivottolerance::Float64=1e-12,
    errornormalization=nothing
)
    # This is necessary to find the return type of f.
    firstpivotval = f(firstpivot)
    cf = CachedFunction(f, Dict(firstpivot => firstpivotval))

    N::Float64 = isnothing(errornormalization) ? abs(firstpivotval) : errornormalization

    return crossinterpolate(
        cf,
        localdims,
        firstpivot,
        tolerance=tolerance,
        maxiter=maxiter,
        sweepstrategy=sweepstrategy,
        pivottolerance=pivottolerance,
        errornormalization=N
    )
end

"""
Optimize firstpivot by a simple deterministic algorithm
"""
function optfirstpivot(
    f,
    localdims::Vector{Int},
    firstpivot::MultiIndex=ones(Int, length(localdims));
    maxsweep=1000
)
    n = length(localdims)
    valf = abs(f(firstpivot))
    pivot = copy(firstpivot)

    for _ in 1:maxsweep
        valf_prev = valf
        for i in 1:n
            for d in 1:localdims[i]
                bak = pivot[i]
                pivot[i] = d
                if abs(f(pivot)) > valf
                    valf = abs(f(pivot))
                else
                    pivot[i] = bak
                end
            end
        end
        if valf_prev == valf
            break
        end
    end

    return pivot
end
