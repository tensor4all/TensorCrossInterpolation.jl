mutable struct MatrixLUCI{T} <: AbstractMatrixCI{T}
    lu::rrLU{T}
end

function MatrixLUCI(A::AbstractMatrix{T}; kwargs...) where {T}
    MatrixLUCI{T}(rrlu(A; kwargs...))
end

function MatrixLUCI(
    ::Type{ValueType},
    f,
    matrixsize::Tuple{Int,Int},
    I0::AbstractVector{Int}=Int[],
    J0::AbstractVector{Int}=Int[];
    kwargs...
)::rrLU{ValueType} where {ValueType}
    MatrixLUCI{T}(rrlu(ValueType, f, matrixsize, I0, J0; kwargs...))
end

function size(luci::MatrixLUCI{T}) where {T}
    return size(luci.lu)
end

function size(luci::MatrixLUCI{T}, dim) where {T}
    return size(luci.lu, dim)
end

function npivots(luci::MatrixLUCI{T}) where {T}
    return npivots(luci.lu)
end

function rowindices(luci::MatrixLUCI{T}) where {T}
    return rowindices(luci.lu)
end

function colindices(luci::MatrixLUCI{T}) where {T}
    return colindices(luci.lu)
end

function colmatrix(luci::MatrixLUCI{T}) where {T}
    return left(luci.lu) * right(luci.lu, permute=false)[:, 1:npivots(luci)]
end

function rowmatrix(luci::MatrixLUCI{T}) where {T}
    return left(luci.lu, permute=false)[1:npivots(luci), :] * right(luci.lu)
end

function colstimespivotinv(luci::MatrixLUCI{T}) where {T}
    n = npivots(luci)
    result = Matrix{T}(I, size(luci, 1), n)
    if n < size(luci, 1)
        L = left(luci.lu; permute=false)
        result[n+1:end, :] = L[n+1:end, :] / LowerTriangular(L[1:n, :])
    end
    result[luci.lu.rowpermutation, :] = result
    return result
end

function pivotinvtimesrows(luci::MatrixLUCI{T}) where {T}
    n = npivots(luci)
    result = Matrix{T}(I, n, size(luci, 2))
    if n < size(luci, 2)
        U = right(luci.lu; permute=false)
        result[:, n+1:end] = UpperTriangular(U[:, 1:n]) \ U[:, n+1:end]
    end
    result[:, luci.lu.colpermutation] = result
    return result
end

function left(luci::MatrixLUCI{T}) where {T}
    if luci.lu.leftorthogonal
        return colstimespivotinv(luci)
    else
        return colmatrix(luci)
    end
end

function right(luci::MatrixLUCI{T}) where {T}
    if luci.lu.leftorthogonal
        return rowmatrix(luci)
    else
        return pivotinvtimesrows(luci)
    end
end

function pivoterrors(luci::MatrixLUCI{T}) where {T}
    return pivoterrors(luci.lu)
end

function lastpivoterror(luci::MatrixLUCI{T}) where {T}
    return lastpivoterror(luci.lu)
end
