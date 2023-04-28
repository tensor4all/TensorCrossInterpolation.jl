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

function rrlu(
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    leftorthogonal::Bool=true
)::rrLU{T} where {T}
    maxrank = min(maxrank, size(A)...)
    lu = rrLU{T}(copy(A), leftorthogonal=leftorthogonal)

    for k in 1:maxrank
        lu.npivot = k
        newpivot = submatrixargmax(abs.(lu.buffer), k)
        if k > 1 && abs(lu.buffer[newpivot...]) < reltol * abs(lu.buffer[1])
            break
        end
        addpivot!(lu, newpivot)
    end

    return lu
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
