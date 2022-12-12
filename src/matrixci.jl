
"""
    AtimesBinv(A::Matrix, B::Matrix)

Calculates the matrix product ``A B^{-1}``, given a rectangular matrix ``A``
and a square matrix ``B`` in a numerically stable way using QR
decomposition. This is useful in case ``B^{-1}`` is ill-conditioned.
"""
function AtimesBinv(A::AbstractMatrix, B::AbstractMatrix)
    m, n = size(A)
    AB = vcat(A, B)
    decomposition = LinearAlgebra.qr(AB)
    QA = decomposition.Q[1:m, 1:n]
    QB = decomposition.Q[(m+1):end, 1:n]
    return QA * inv(QB)
end

"""
    AtimesBinv(A::Matrix, B::Matrix)

Calculates the matrix product ``A^{-1} B``, given a square matrix ``A``
and a rectangular matrix ``B`` in a numerically stable way using QR
decomposition. This is useful in case ``A^{-1}`` is ill-conditioned.
"""
function AinvtimesB(A::AbstractMatrix, B::AbstractMatrix)
    return AtimesBinv(B', A')'
end

"""
    mutable struct MatrixCrossInterpolation{T}

Represents a cross interpolation of a matrix - or, to be more precise, the data
necessary to evaluate a cross interpolation. This data can be fed into various
methods such as evaluate(c, i, j) to get interpolated values, and be improved by
dynamically adding pivots.
"""
mutable struct MatrixCI{T}
    "Same as `CrossData.Iset` in xfac code, or ``\\mathcal{I}`` in TCI paper."
    rowindices::Vector{Int}
    "Same as `CrossData.Jset` in xfac code, or ``\\mathcal{J}`` in TCI paper."
    colindices::Vector{Int}
    "Same as `CrossData.C` in xfac code, or ``A(\\mathbb{I}, \\mathcal{J})`` in TCI paper."
    pivotcols::Matrix{T}
    "Same as `CrossData.R` in xfac code, or ``A(\\mathcal{I}, \\mathbb{J})`` in TCI paper."
    pivotrows::Matrix{T}

    function MatrixCI{T}(
        nrows::Int, ncols::Int
    ) where {T<:Number}
        return new{T}([], [], zeros(nrows, 0), zeros(0, ncols))
    end

    function MatrixCI(
        rowindices::AbstractVector{Int}, colindices::AbstractVector{Int},
        pivotcols::AbstractMatrix{T}, pivotrows::AbstractMatrix{T}
    ) where {T<:Number}
        return new{T}(rowindices, colindices, pivotcols, pivotrows)
    end
end

function MatrixCI(
    A::AbstractMatrix{T}, firstpivot
) where {T<:Number}
    return MatrixCI(
        [firstpivot[1]], [firstpivot[2]],
        A[:, [firstpivot[2]]], A[[firstpivot[1]], :]
    )
end

function Iset(ci::MatrixCI{T}) where {T}
    return ci.rowindices
end

function Jset(ci::MatrixCI{T}) where {T}
    return ci.colindices
end

function nrows(ci::MatrixCI{T}) where {T}
    return size(ci.pivotcols, 1)
end

function ncols(ci::MatrixCI{T}) where {T}
    return size(ci.pivotrows, 2)
end

function Base.size(ci::MatrixCI{T}) where {T}
    return nrows(ci), ncols(ci)
end

function pivotmatrix(ci::MatrixCI{T}) where {T}
    return ci.pivotcols[ci.rowindices, :]
end

function leftmatrix(ci::MatrixCI{T}) where {T}
    return AtimesBinv(ci.pivotcols, pivotmatrix(ci))
end

function rightmatrix(ci::MatrixCI{T}) where {T}
    return AinvtimesB(pivotmatrix(ci), ci.pivotrows)
end

function availablerows(ci::MatrixCI{T}) where {T}
    return setdiff(1:nrows(ci), ci.rowindices)
end

function availablecols(ci::MatrixCI{T}) where {T}
    return setdiff(1:ncols(ci), ci.colindices)
end

function rank(ci::MatrixCI{T}) where {T}
    return length(ci.rowindices)
end

function isempty(ci::MatrixCI{T}) where {T}
    return Base.isempty(ci.pivotcols)
end

function firstpivotvalue(ci::MatrixCI{T}) where {T}
    return isempty(ci) ? 1.0 : ci.pivotcols[ci.rowindices[1], 1]
end

function evaluate(ci::MatrixCI{T}, i::Int, j::Int) where {T}
    if isempty(ci)
        return T(0)
    else
        return dot(leftmatrix(ci)[i, :], ci.pivotrows[:, j])
    end
end

function _lengthordefault(c::Colon, default)
    return default
end

function _lengthordefault(c, default)
    return length(c)
end

function submatrix(
    ci::MatrixCI{T},
    rows::Union{AbstractVector{Int},Colon,Int},
    cols::Union{AbstractVector{Int},Colon,Int}
) where {T}
    if isempty(ci)
        return zeros(
            T,
            _lengthordefault(rows, nrows(ci)),
            _lengthordefault(cols, ncols(ci)))
    else
        return leftmatrix(ci)[rows, :] * ci.pivotrows[:, cols]
    end
end

function row(
    ci::MatrixCI{T},
    i::Int;
    cols::Union{AbstractVector{Int},Colon}=Colon()) where {T}
    return submatrix(ci, [i], cols)[:]
end

function col(
    ci::MatrixCI{T},
    j::Int;
    rows::Union{AbstractVector{Int},Colon}=Colon()) where {T}
    return submatrix(ci, rows, [j])[:]
end

function Base.getindex(
    ci::MatrixCI{T},
    rows::Union{AbstractVector{Int},Colon},
    cols::Union{AbstractVector{Int},Colon}) where {T}
    return submatrix(ci, rows, cols)
end

function Base.getindex(
    ci::MatrixCI{T},
    i::Int,
    cols::Union{AbstractVector{Int},Colon}) where {T}
    return row(ci, i; cols=cols)
end

function Base.getindex(ci::MatrixCI{T},
    rows::Union{AbstractVector{Int},Colon},
    j::Int) where {T}
    return col(ci, j; rows=rows)
end

function Base.getindex(
    ci::MatrixCI{T}, i::Int, j::Int) where {T}
    return evaluate(ci, i, j)
end

function Base.isapprox(
    lhs::MatrixCI{T}, rhs::MatrixCI{T}
) where {T}
    return (lhs.colindices == rhs.colindices) &&
           (lhs.rowindices == rhs.rowindices) &&
           Base.isapprox(lhs.pivotcols, rhs.pivotcols) &&
           Base.isapprox(lhs.pivotrows, rhs.pivotrows)
end

function Base.Matrix(ci::MatrixCI{T}) where {T}
    return leftmatrix(ci) * ci.pivotrows
end

"""
    localerror(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T},
        row_indices::Union{AbstractVector{Int},Colon,Int}=Colon(),
        col_indices::Union{AbstractVector{Int},Colon,Int}=Colon())

    Returns all local errors of the cross interpolation ci with respect to
    matrix a. To find local errors on a submatrix, specify row_indices or
    col_indices.
"""
function localerror(
    a::AbstractMatrix{T},
    ci::MatrixCI{T},
    rowindices::Union{AbstractVector{Int},Colon,Int}=Colon(),
    colindices::Union{AbstractVector{Int},Colon,Int}=Colon()
) where {T}
    return abs.(a[rowindices, colindices] .- ci[rowindices, colindices])
end

"""
    findnewpivot(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T},
        row_indices::Union{Vector{Int},Colon},
        col_indices::Union{Vector{Int},Colon})

    Finds the pivot that maximizes the local error across all components of
    `a` and `ci` within the subset given by `rowindices` and `colindices`. By
    default, all avalable rows of `ci` will be considered.
"""
function findnewpivot(
    a::AbstractMatrix{T},
    ci::MatrixCI{T},
    rowindices::Union{Vector{Int},Colon}=availablerows(ci),
    colindices::Union{Vector{Int},Colon}=availablecols(ci)
) where {T}
    if rank(ci) == minimum(size(a))
        throw(ArgumentError(
            "Cannot find a new pivot for this MatrixCrossInterpolation, as it is
            already full rank."))
    elseif (length(rowindices) == 0)
        throw(ArgumentError(
            "Cannot find a new pivot in an empty set of rows
            (row_indices = $rowindices)"
        ))
    elseif (length(colindices) == 0)
        throw(ArgumentError(
            "Cannot find a new pivot in an empty set of cols
            (col_indices = $rowindices)"
        ))
    end

    localerrors = localerror(a, ci, rowindices, colindices)
    ijraw = argmax(localerrors)
    return (rowindices[ijraw[1]], colindices[ijraw[2]]), localerrors[ijraw]
end

"""
    function addpivot(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T},
        pivotindices::AbstractArray{T, 2})

    Add a new pivot given by `pivotindices` to the cross interpolation `ci` of
    the matrix `a`.
"""
function addpivot!(
    a::AbstractMatrix{T},
    ci::MatrixCI{T},
    pivotindices::Union{CartesianIndex{2},Tuple{Int,Int},Pair{Int,Int}}
) where {T}
    i = pivotindices[1]
    j = pivotindices[2]

    if size(a) != size(ci)
        throw(DimensionMismatch(
            "This matrix doesn't match the MatrixCrossInterpolation object.
            Their sizes mismatch: $(size(a)) != $(size(ci))."))
    elseif (i < 0) || (i > nrows(ci)) || (j < 0) || (j > ncols(ci))
        throw(BoundsError(
            "Cannot add a pivot at indices ($i, $j): These indices are out of
            bounds for a $(nrows(ci)) * $(ncols(ci)) matrix."))
    elseif i in ci.rowindices
        throw(ArgumentError(
            "Cannot add a pivot at indices ($i, $j) because row $i already has a
            pivot."))
    elseif j in ci.colindices
        throw(ArgumentError(
            "Cannot add a pivot at indices ($i, $j) because col $j already has a
            pivot."))
    end

    row = transpose(a[i, :])
    ci.pivotrows = vcat(ci.pivotrows, row)
    push!(ci.rowindices, i)
    col = a[:, j]
    ci.pivotcols = hcat(ci.pivotcols, col)
    push!(ci.colindices, j)
end

"""
    function addpivot(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T})

    Add a new pivot that maximizes the error to the cross interpolation `ci` of
    the matrix `a`.
"""
function addpivot!(
    a::AbstractMatrix{T},
    ci::MatrixCI{T}
) where {T}
    return addpivot!(a, ci, findnewpivot(a, ci)[1])
end

function crossinterpolate(
    a::AbstractMatrix{T};
    tolerance=1e-6,
    maxiter=200,
    firstpivot=argmax(abs.(a))
) where {T}
    ci = MatrixCI(a, firstpivot)
    for iter in 1:maxiter
        pivoterror, newpivot = findmax(localerror(a, ci))
        if pivoterror < tolerance
            return ci
        end
        addpivot!(a, ci, newpivot)
    end
    return ci
end
