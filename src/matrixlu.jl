function submatrixargmax(
    f::Function, # real valued function
    A::AbstractMatrix{T},
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange};
    colmask::Function=x->true,
    rowmask::Function=x->true
) where {T}
    m = typemin(f(first(A)))
    !isempty(rows) || throw(ArgumentError("rows must not be empty"))
    !isempty(cols) || throw(ArgumentError("cols must not be empty"))
    mr = first(rows)
    mc = first(cols)
    rows ⊆ axes(A, 1) || throw(ArgumentError("rows ⊆ axes(A, 1) must be satified"))
    cols ⊆ axes(A, 2) || throw(ArgumentError("cols ⊆ axes(A, 2) must be satified"))
    @inbounds for c in cols
        if !colmask(c)
            continue
        end
        for r in rows
            if !rowmask(r)
                continue
            end
            v = f(A[r, c])
            newm = v > m
            m = ifelse(newm, v, m)
            mr = ifelse(newm, r, mr)
            mc = ifelse(newm, c, mc)
        end
    end
    return mr, mc
end

function submatrixargmax(
    A::AbstractMatrix,
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange},
)
    return submatrixargmax(identity, A, rows, cols)
end

function submatrixargmax(f::Function, A::AbstractMatrix, rows, cols; colmask::Function=x->true, rowmask::Function=x->true)
    function convertarg(arg::Int, size::Int)
        return [arg]
    end

    function convertarg(arg::Colon, size::Int)
        return 1:size
    end

    function convertarg(arg::Union{AbstractVector,UnitRange}, size::Int)
        return arg
    end

    return submatrixargmax(f, A, convertarg(rows, size(A, 1)), convertarg(cols, size(A, 2)); colmask=colmask, rowmask=rowmask)
end

function submatrixargmax(A::AbstractMatrix, rows, cols)
    submatrixargmax(identity, A::AbstractMatrix, rows, cols)
end


function submatrixargmax(f::Function, A::AbstractMatrix, startindex::Int; colmask::Function=x->true, rowmask::Function=x->true)
    return submatrixargmax(f, A, startindex:size(A, 1), startindex:size(A, 2); colmask=colmask, rowmask=rowmask)
end

function submatrixargmax(A::AbstractMatrix, startindex::Int)
    return submatrixargmax(identity, A, startindex:size(A, 1), startindex:size(A, 2))
end

mutable struct rrLU{T}
    rowpermutation::Vector{Int}
    colpermutation::Vector{Int}
    L::Matrix{T}
    U::Matrix{T}
    leftorthogonal::Bool
    npivot::Int
    error::Float64

    function rrLU{T}(nrows::Int, ncols::Int; leftorthogonal::Bool=true) where {T}
        new{T}(1:nrows, 1:ncols, zeros(nrows, 0), zeros(0, ncols), leftorthogonal, 0, NaN)
    end
end

function rrLU{T}(A::AbstractMatrix{T}; leftorthogonal::Bool=true) where {T}
    rrLU{T}(size(A)...; leftorthogonal=leftorthogonal)
end

function _addpivot!(lu, A, reltol, abstol; colmask::Function=x->true, rowmask::Function=x->true)::Tuple{Bool,Bool} # (break, exactlowrank)
    k = lu.npivot + 1
    newpivot = submatrixargmax(abs2, A, k; colmask=colmask, rowmask=rowmask)
    if A[newpivot...] == 0
        return (true, true)
    end
    addpivot!(lu, A, newpivot)
    if abs(A[k, k]) < reltol * abs(A[1]) || abs(A[k, k]) < abstol
        return (true, false)
    end
    return (false, false)
end

_count_cols_selected(lu, cols)::Int = sum([c ∈ lu.colpermutation[1:lu.npivot] for c ∈ cols])
_count_rows_selected(lu, rows)::Int = sum([r ∈ lu.rowpermutation[1:lu.npivot] for r ∈ rows])

"""
We add at least one column in each entry of `conservedcols` if numerically stable. The same applies to `conservedrows`.
"""
function _optimizerrlu!(
    lu::rrLU{T},
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    conservedcols::Vector{Vector{Int}} = Vector{Int}[],
    conservedrows::Vector{Vector{Int}} = Vector{Int}[]
) where {T}
    if length(conservedcols) > 0
        allunique(vcat(conservedcols...)) || error("Duplicate in conservedcols")
    end
    if length(conservedrows) > 0
        allunique(vcat(conservedrows...)) || error("Duplicate in conservedrows")
    end

    maxrank = min(maxrank, size(A)...)

    exactlowrank = false
    while lu.npivot < maxrank
        terminated, exactlowrank = _addpivot!(lu, A, reltol, abstol)
        if terminated
            break
        end
    end

    # Error estimate must be done additional pivots are forcely added
    if exactlowrank || lu.npivot == min(size(lu)...)
        lu.error = zero(T)
    else
        lu.error = lu.leftorthogonal ? abs(A[lu.npivot, lu.npivot]) : abs(A[lu.npivot, lu.npivot])
    end

    if !exactlowrank && length(conservedcols) > 0
        for cols in conservedcols
            mask = fill(false, size(A, 2))
            mask[cols] .= true
            while lu.npivot < maxrank && _count_cols_selected(lu, cols) == 0
                terminated_, _ = _addpivot!(lu, A, 1e-14, 0.0; colmask=x->mask[x])
                if terminated_
                   break
                end
            end
        end
    end
    if !exactlowrank && length(conservedrows) > 0
        for rows in conservedrows
            mask = fill(false, size(A, 1))
            mask[rows] .= true
            while lu.npivot < maxrank && _count_rows_selected(lu, rows) == 0
                terminated_, _ = _addpivot!(lu, A, 1e-14, 0.0; rowmask=x->mask[x])
                if terminated_
                   break
                end
            end
        end
    end

    lu.L = tril(A[:, 1:lu.npivot])
    lu.U = triu(A[1:lu.npivot, :])
    if any(isnan.(lu.L))
        error("lu.L contains NaNs")
    end
    if any(isnan.(lu.U))
        error("lu.U contains NaNs")
    end
    if lu.leftorthogonal
        lu.L[diagind(lu.L)] .= one(T)
    else
        lu.U[diagind(lu.U)] .= one(T)
    end

    nothing
end

"""
    function rrlu!(
        A::AbstractMatrix{T};
        maxrank::Int=typemax(Int),
        reltol::Number=1e-14,
        abstol::Number=0.0,
        leftorthogonal::Bool=true
    )::rrLU{T} where {T}

Rank-revealing LU decomposition.
"""
function rrlu!(
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true,
    conservedcols::Vector{Vector{Int}} = Vector{Int}[],
    conservedrows::Vector{Vector{Int}} = Vector{Int}[],
)::rrLU{T} where {T}
    lu = rrLU{T}(A, leftorthogonal=leftorthogonal)
    _optimizerrlu!(lu, A; maxrank=maxrank, reltol=reltol, abstol=abstol, conservedcols=conservedcols, conservedrows=conservedrows)
    return lu
end

"""
    function rrlu(
        A::AbstractMatrix{T};
        maxrank::Int=typemax(Int),
        reltol::Number=1e-14,
        abstol::Number=0.0,
        leftorthogonal::Bool=true
    )::rrLU{T} where {T}

Rank-revealing LU decomposition.
"""
function rrlu(
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true,
    conservedcols::Vector{Vector{Int}} = Vector{Int}[],
    conservedrows::Vector{Vector{Int}} = Vector{Int}[],
)::rrLU{T} where {T}
    return rrlu!(copy(A); maxrank, reltol, abstol, leftorthogonal, conservedcols, conservedrows)
end

function arrlu(
    ::Type{ValueType},
    f,
    matrixsize::Tuple{Int,Int},
    I0::AbstractVector{Int}=Int[],
    J0::AbstractVector{Int}=Int[];
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true,
    numrookiter::Int=3,
    usebatcheval::Bool=false
)::rrLU{ValueType} where {ValueType}
    lu = rrLU{ValueType}(matrixsize...; leftorthogonal)
    islowrank = false
    maxrank = min(maxrank, matrixsize...)

    _batchf = usebatcheval ? f : ((x, y) -> f.(x, y'))

    while true
        if leftorthogonal
            pushrandomsubset!(J0, 1:matrixsize[2], max(1, length(J0)))
        else
            pushrandomsubset!(I0, 1:matrixsize[1], max(1, length(I0)))
        end

        for rookiter in 1:numrookiter
            colmove = (iseven(rookiter) == leftorthogonal)
            submatrix = if colmove
                _batchf(I0, lu.colpermutation)
            else
                _batchf(lu.rowpermutation, J0)
            end
            lu.npivot = 0
            _optimizerrlu!(lu, submatrix; maxrank, reltol, abstol)
            islowrank |= npivots(lu) < minimum(size(submatrix))
            if rowindices(lu) == I0 && colindices(lu) == J0
                break
            end

            J0 = colindices(lu)
            I0 = rowindices(lu)
        end

        if islowrank || length(I0) >= maxrank
            break
        end
    end

    if size(lu.L, 1) < matrixsize[1]
        I2 = setdiff(1:matrixsize[1], I0)
        lu.rowpermutation = vcat(I0, I2)
        L2 = _batchf(I2, J0)
        cols2Lmatrix!(L2, lu.U[1:lu.npivot, 1:lu.npivot], leftorthogonal)
        lu.L = vcat(lu.L[1:lu.npivot, 1:lu.npivot], L2)
    end

    if size(lu.U, 2) < matrixsize[2]
        J2 = setdiff(1:matrixsize[2], J0)
        lu.colpermutation = vcat(J0, J2)
        U2 = _batchf(I0, J2)
        rows2Umatrix!(U2, lu.L[1:lu.npivot, 1:lu.npivot], leftorthogonal)
        lu.U = hcat(lu.U[1:lu.npivot, 1:lu.npivot], U2)
    end

    return lu
end

function rrlu(
    ::Type{ValueType},
    f,
    matrixsize::Tuple{Int,Int},
    I0::AbstractVector{Int}=Int[],
    J0::AbstractVector{Int}=Int[];
    pivotsearch=:full,
    kwargs...
)::rrLU{ValueType} where {ValueType}
    if pivotsearch === :rook
        return arrlu(ValueType, f, matrixsize, I0, J0; kwargs...)
    elseif pivotsearch === :full
        A = f.(1:matrixsize[1], collect(1:matrixsize[2])')
        return rrlu!(A; kwargs...)
    else
        throw(ArgumentError("Unknown pivot search strategy $pivotsearch. Choose between :rook and :full."))
    end
end

function cols2Lmatrix!(C::AbstractMatrix, P::AbstractMatrix, leftorthogonal::Bool)
    if size(C, 2) != size(P, 2)
        throw(DimensionMismatch("C and P matrices must have same number of columns in `cols2Lmatrix!`."))
    elseif size(P, 1) != size(P, 2)
        throw(DimensionMismatch("P matrix must be square in `cols2Lmatrix!`."))
    end

    for k in axes(P, 1)
        C[:, k] /= P[k, k]
        C[:, k+1:end] -= C[:, k] * transpose(P[k, k+1:end])
    end
    return C
end

function rows2Umatrix!(R::AbstractMatrix, P::AbstractMatrix, leftorthogonal::Bool)
    if size(R, 1) != size(P, 1)
        throw(DimensionMismatch("R and P matrices must have same number of rows in `rows2Umatrix!`."))
    elseif size(P, 1) != size(P, 2)
        throw(DimensionMismatch("P matrix must be square in `rows2Umatrix!`."))
    end

    for k in axes(P, 1)
        R[k, :] /= P[k, k]
        R[k+1:end, :] -= P[k+1:end, k] * transpose(R[k, :])
    end
    return R
end

function swaprow!(lu::rrLU{T}, A::AbstractMatrix{T}, a, b) where {T}
    lu.rowpermutation[[a, b]] = lu.rowpermutation[[b, a]]
    A[[a, b], :] = A[[b, a], :]
end

function swapcol!(lu::rrLU{T}, A::AbstractMatrix{T}, a, b) where {T}
    lu.colpermutation[[a, b]] = lu.colpermutation[[b, a]]
    A[:, [a, b]] = A[:, [b, a]]
end

function addpivot!(lu::rrLU{T}, A::AbstractMatrix{T}, newpivot) where {T}
    k = lu.npivot += 1
    swaprow!(lu, A, k, newpivot[1])
    swapcol!(lu, A, k, newpivot[2])

    if lu.leftorthogonal
        A[k+1:end, k] /= A[k, k]
    else
        A[k, k+1:end] /= A[k, k]
    end

    # perform BLAS subroutine manually: A <- -x * transpose(y) + A
    x = @view(A[k+1:end, k])
    y = @view(A[k, k+1:end])
    A = @view(A[k+1:end, k+1:end])
    @inbounds for j in eachindex(axes(A, 2), y)
        for i in eachindex(axes(A, 1), x)
            # update `A[k+1:end, k+1:end]`
            A[i, j] -= x[i] * y[j]
        end
    end
    nothing
end

function size(lu::rrLU{T}) where {T}
    return size(lu.L, 1), size(lu.U, 2)
end

function size(lu::rrLU{T}, dim) where {T}
    if dim == 1
        return size(lu.L, 1)
    elseif dim == 2
        return size(lu.U, 2)
    else
        return 1
    end
end

function left(lu::rrLU{T}; permute=true) where {T}
    if permute
        l = Matrix{T}(undef, size(lu.L)...)
        l[lu.rowpermutation, :] = lu.L
        return l
    else
        return lu.L
    end
end

function right(lu::rrLU{T}; permute=true) where {T}
    if permute
        u = Matrix{T}(undef, size(lu.U)...)
        u[:, lu.colpermutation] = lu.U
        return u
    else
        return lu.U
    end
end

function diag(lu::rrLU{T}) where {T}
    if lu.leftorthogonal
        return diag(lu.U[1:lu.npivot, 1:lu.npivot])
    else
        return diag(lu.L[1:lu.npivot, 1:lu.npivot])
    end
end

function rowindices(lu::rrLU{T}) where {T}
    return lu.rowpermutation[1:lu.npivot]
end

function colindices(lu::rrLU{T}) where {T}
    return lu.colpermutation[1:lu.npivot]
end

function npivots(lu::rrLU{T}) where {T}
    return lu.npivot
end

function pivoterrors(lu::rrLU{T}) where {T}
    return abs.(diag(lu))
end

"""
Correct estimate of the last pivot error
A special care is taken for a full-rank matrix: the last pivot error is set to zero.
"""
function lastpivoterror(lu::rrLU{T})::Float64 where {T}
    return lu.error
end
