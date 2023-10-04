function submatrixargmax(
    A::AbstractMatrix,
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange}
)
    result = argmax(A[rows, cols]) |> Tuple
    return rows[result[1]], cols[result[2]]
end

function submatrixargmax(
    A::AbstractMatrix,
    rows,
    cols
)
    function convertarg(arg::Int, size::Int)
        return [arg]
    end

    function convertarg(arg::Colon, size::Int)
        return 1:size
    end

    function convertarg(arg::Union{AbstractVector,UnitRange}, size::Int)
        return arg
    end

    return submatrixargmax(
        A,
        convertarg(rows, size(A, 1)),
        convertarg(cols, size(A, 2))
    )
end

function submatrixargmax(
    A::AbstractMatrix,
    startindex::Int
)
    return submatrixargmax(A, startindex:size(A, 1), startindex:size(A, 2))
end

mutable struct rrLU{T}
    rowpermutation::Vector{Int}
    colpermutation::Vector{Int}
    buffer::Matrix{T}
    leftorthogonal::Bool
    npivot::Int

    function rrLU{T}(nrows::Int, ncols::Int; leftorthogonal::Bool=true) where {T}
        new{T}(1:nrows, 1:ncols, zeros(nrows, ncols), leftorthogonal, 0)
    end

    function rrLU{T}(A::AbstractMatrix{T}; leftorthogonal::Bool=true) where {T}
        new{T}(axes(A, 1), axes(A, 2), A, leftorthogonal, 0)
    end
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
        lu.npivot = k
        newpivot = submatrixargmax(abs.(lu.buffer), k)
        if k >= 1 && (
            abs(lu.buffer[newpivot...]) < reltol * abs(lu.buffer[1]) ||
            abs(lu.buffer[newpivot...]) < abstol
        )
            break
        end
        addpivot!(lu, newpivot)
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
    I0::AbstractVector{Int},
    J0::AbstractVector{Int};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true,
    numrookiter::Int=3
)::rrLU{ValueType} where {ValueType}
    local lu
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

            islowrank = npivots(lu) < minimum(size(submatrix))
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

    P = lu.buffer[1:lu.npivot, 1:lu.npivot]
    I2 = setdiff(1:matrixsize[1], I0)
    J2 = setdiff(1:matrixsize[2], J0)
    L2 = cols2Lmatrix!(f.(I2, J0'), P, leftorthogonal)
    U2 = rows2Umatrix!(f.(I0, J2'), P, leftorthogonal)

    lu.buffer = [
        P U2
        L2 zeros(length(I2), length(J2))
    ]
    lu.rowpermutation = vcat(I0, I2)
    lu.colpermutation = vcat(J0, J2)

    return lu
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

function swaprow!(lu::rrLU{T}, a, b) where {T}
    lu.rowpermutation[[a, b]] = lu.rowpermutation[[b, a]]
    lu.buffer[[a, b], :] = lu.buffer[[b, a], :]
end

function swapcol!(lu::rrLU{T}, a, b) where {T}
    lu.colpermutation[[a, b]] = lu.colpermutation[[b, a]]
    lu.buffer[:, [a, b]] = lu.buffer[:, [b, a]]
end

function addpivot!(lu::rrLU{T}, newpivot) where {T}
    k = lu.npivot
    swaprow!(lu, k, newpivot[1])
    swapcol!(lu, k, newpivot[2])

    if lu.leftorthogonal
        lu.buffer[k+1:end, k] /= lu.buffer[k, k]
    else
        lu.buffer[k, k+1:end] /= lu.buffer[k, k]
    end

    lu.buffer[k+1:end, k+1:end] -= lu.buffer[k+1:end, k] * transpose(lu.buffer[k, k+1:end])
end

function size(lu::rrLU{T}) where {T}
    return size(lu.buffer)
end

function size(lu::rrLU{T}, dim) where {T}
    return size(lu.buffer, dim)
end

function left(lu::rrLU{T}; permute=true) where {T}
    l = tril(lu.buffer[:, 1:lu.npivot])
    if lu.leftorthogonal
        l[diagind(l)] .= one(T)
    end
    if permute
        l[lu.rowpermutation, :] = l
    end
    return l
end

function right(lu::rrLU{T}; permute=true) where {T}
    u = triu(lu.buffer[1:lu.npivot, :])
    if !lu.leftorthogonal
        u[diagind(u)] .= one(T)
    end
    if permute
        u[:, lu.colpermutation] = u
    end
    return u
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
    return abs.(diag(lu.buffer[1:lu.npivot, 1:lu.npivot]))
end

"""
Correct estimate of the last pivot error
A special care is taken for a full-rank matrix: the last pivot error is set to zero.
"""
function lastpivoterror(lu::rrLU{T})::Float64 where {T}
    if lu.npivot == min(size(lu.buffer, 1), size(lu.buffer, 2))
        return 0.0
    else
        return abs(lu.buffer[lu.npivot, lu.npivot])
    end
end
