function submatrixargmax(
    f::Function, # real valued function
    A::AbstractMatrix{T},
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange},
) where {T}
    m = typemin(f(first(A)))
    !isempty(rows) || throw(ArgumentError("rows must not be empty"))
    !isempty(cols) || throw(ArgumentError("cols must not be empty"))
    mr = first(rows)
    mc = first(cols)
    rows ⊆ axes(A, 1) || throw(ArgumentError("rows ⊆ axes(A, 1) must be satified"))
    cols ⊆ axes(A, 2) || throw(ArgumentError("cols ⊆ axes(A, 2) must be satified"))
    @inbounds for c in cols
        for r in rows
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

function submatrixargmax(f::Function, A::AbstractMatrix, rows, cols)
    function convertarg(arg::Int, size::Int)
        return [arg]
    end

    function convertarg(arg::Colon, size::Int)
        return 1:size
    end

    function convertarg(arg::Union{AbstractVector,UnitRange}, size::Int)
        return arg
    end

    return submatrixargmax(f, A, convertarg(rows, size(A, 1)), convertarg(cols, size(A, 2)))
end

function submatrixargmax(A::AbstractMatrix, rows, cols)
    submatrixargmax(identity, A::AbstractMatrix, rows, cols)
end


function submatrixargmax(f::Function, A::AbstractMatrix, startindex::Int)
    return submatrixargmax(f, A, startindex:size(A, 1), startindex:size(A, 2))
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

    function rrLU{T}(nrows::Int, ncols::Int; leftorthogonal::Bool=true) where {T}
        new{T}(1:nrows, 1:ncols, zeros(nrows, 0), zeros(0, ncols), leftorthogonal, 0)
    end
end

function rrLU{T}(A::AbstractMatrix{T}; leftorthogonal::Bool=true) where {T}
    rrLU{T}(size(A)...; leftorthogonal=leftorthogonal)
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
    leftorthogonal::Bool=true
)::rrLU{T} where {T}
    maxrank = min(maxrank, size(A)...)
    lu = rrLU{T}(A, leftorthogonal=leftorthogonal)

    for k in 1:maxrank
        newpivot = submatrixargmax(abs2, A, k)
        if abs(A[newpivot...]) < reltol * abs(A[1]) || abs(A[newpivot...]) < abstol
            break
        end
        addpivot!(lu, A, newpivot)
    end
    lu.L = tril(A[:, 1:lu.npivot])
    lu.U = triu(A[1:lu.npivot, :])
    if lu.leftorthogonal
        lu.L[diagind(lu.L)] .= one(T)
    else
        lu.U[diagind(lu.U)] .= one(T)
    end
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
    leftorthogonal::Bool=true
)::rrLU{T} where {T}
    return rrlu!(copy(A); maxrank, reltol, abstol, leftorthogonal)
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
    numrookiter::Int=3
)::rrLU{ValueType} where {ValueType}
    local lu::rrLU{ValueType}
    islowrank = false
    maxrank = min(maxrank, matrixsize...)

    while true
        if leftorthogonal
            pushrandomsubset!(J0, 1:matrixsize[2], max(1, length(J0)))
        else
            pushrandomsubset!(I0, 1:matrixsize[1], max(1, length(I0)))
        end

        for rookiter in 1:numrookiter
            colmove = (iseven(rookiter) == leftorthogonal)
            submatrix = if colmove
                f.(I0, collect(1:matrixsize[2])')
            else
                f.(1:matrixsize[1], J0')
            end
            lu = rrlu!(submatrix; maxrank, reltol, abstol, leftorthogonal)
            islowrank |= npivots(lu) < minimum(size(submatrix))
            if rowindices(lu) == I0 && colindices(lu) == J0
                break
            end

            if colmove
                J0 = colindices(lu)
            else
                I0 = rowindices(lu)
            end
        end

        if islowrank || length(I0) >= maxrank
            break
        end
    end

    I2 = setdiff(1:matrixsize[1], I0)
    J2 = setdiff(1:matrixsize[2], J0)
    L2 = cols2Lmatrix!(f.(I2, J0'), lu.U[1:lu.npivot, 1:lu.npivot], leftorthogonal)
    U2 = rows2Umatrix!(f.(I0, J2'), lu.L[1:lu.npivot, 1:lu.npivot], leftorthogonal)

    lu.L = vcat(lu.L, L2)
    lu.U = hcat(lu.U, U2)
    lu.rowpermutation = vcat(I0, I2)
    lu.colpermutation = vcat(J0, J2)
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
        if leftorthogonal
            C[:, k] /= P[k, k]
        end
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
        if leftorthogonal
            R[k, :] /= P[k, k]
        end
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
    if lu.npivot == min(size(lu)...)
        return 0.0
    else
        return abs(diag(lu)[lu.npivot])
    end
end
