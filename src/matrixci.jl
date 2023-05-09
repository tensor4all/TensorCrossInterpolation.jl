
"""
    AtimesBinv(A::Matrix, B::Matrix)

Calculates the matrix product ``A B^{-1}``, given a rectangular matrix ``A``
and a square matrix ``B`` in a numerically stable way using QR
decomposition. This is useful in case ``B^{-1}`` is ill-conditioned.
"""
function AtimesBinv(A::AbstractVecOrMat, B::AbstractMatrix)
    m, n = size(A)
    AB = vcat(A, B)
    decomposition = LinearAlgebra.qr(AB)
    Q = CuMatrix(decomposition.Q * I)
    QA = Q[1:m, 1:n]
    QB = Q[(m+1):end, 1:n]
    # return QA * inv(QB)
    return QA / QB
end

"""
    AtimesBinv(A::Matrix, B::Matrix)

Calculates the matrix product ``A^{-1} B``, given a square matrix ``A``
and a rectangular matrix ``B`` in a numerically stable way using QR
decomposition. This is useful in case ``A^{-1}`` is ill-conditioned.
"""
function AinvtimesB(A::AbstractMatrix, B::AbstractVecOrMat)
    return AtimesBinv(B', A')'
end

"""
    mutable struct MatrixCrossInterpolation{T}

Represents a cross interpolation of a matrix - or, to be more precise, the data
necessary to evaluate a cross interpolation. This data can be fed into various
methods such as evaluate(c, i, j) to get interpolated values, and be improved by
dynamically adding pivots.
"""
mutable struct MatrixCI{T} <: AbstractMatrixCI{T}
    "Same as `CrossData.Iset` in xfac code, or ``\\mathcal{I}`` in TCI paper."
    rowindices::Vector{Int}
    "Same as `CrossData.Jset` in xfac code, or ``\\mathcal{J}`` in TCI paper."
    colindices::Vector{Int}
    "Same as `CrossData.C` in xfac code, or ``A(\\mathbb{I}, \\mathcal{J})`` in TCI paper."
    pivotcols::CuMatrix{T}
    "Same as `CrossData.R` in xfac code, or ``A(\\mathcal{I}, \\mathbb{J})`` in TCI paper."
    pivotrows::CuMatrix{T}

    function MatrixCI(
        ::Type{T},
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

function nrows(ci::MatrixCI)
    return size(ci.pivotcols, 1)
end

function ncols(ci::MatrixCI)
    return size(ci.pivotrows, 2)
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

function Base.isempty(ci::MatrixCI)
    return Base.isempty(ci.colindices)
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
        return CUDA.zeros(
            T,
            _lengthordefault(rows, nrows(ci)),
            _lengthordefault(cols, ncols(ci)))
    else
        return leftmatrix(ci)[rows, :] * ci.pivotrows[:, cols]
    end
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

function addpivotrow!(
    ci::MatrixCI{T},
    a::AbstractMatrix{T},
    rowindex::Int
) where {T}
    if size(a) != size(ci)
        throw(DimensionMismatch(
            "This matrix doesn't match the MatrixCrossInterpolation object. Their sizes
            mismatch: $(size(a)) != $(size(ci))."))
    elseif (rowindex < 0) || (rowindex > nrows(ci))
        throw(BoundsError(
            "Cannot add row at row index $rowindex: it's out of bounds for a
            $(nrows(ci)) * $(ncols(ci)) matrix."))
    elseif rowindex in ci.rowindices
        throw(ArgumentError(
            "Cannot add row $rowindex: it already has a pivot."))
    end

    row = transpose(a[rowindex, :])
    ci.pivotrows = vcat(ci.pivotrows, row)
    push!(ci.rowindices, rowindex)
end

function addpivotcol!(
    ci::MatrixCI{T},
    a::AbstractMatrix{T},
    colindex::Int
) where {T}
    if size(a) != size(ci)
        throw(DimensionMismatch(
            "This matrix doesn't match the MatrixCrossInterpolation object. Their sizes
            mismatch: $(size(a)) != $(size(ci))."))
    elseif (colindex < 0) || (colindex > ncols(ci))
        throw(BoundsError(
            "Cannot add col at col index $colindex: it's out of bounds for a
            $(nrows(ci)) * $(ncols(ci)) matrix."))
    elseif colindex in ci.colindices
        throw(ArgumentError(
            "Cannot add column $colindex because it already has a pivot."))
    end
    col = a[:, colindex]
    ci.pivotcols = hcat(ci.pivotcols, col)
    push!(ci.colindices, colindex)
end

"""
    function addpivot(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T},
        pivotindices::AbstractArray{T, 2})

Add a new pivot given by `pivotindices` to the cross interpolation `ci` of the matrix `a`.
"""
function addpivot!(
    ci::MatrixCI{T},
    a::AbstractMatrix{T},
    pivotindices::Union{CartesianIndex{2},Tuple{Int,Int},Pair{Int,Int}}
) where {T}
    i = pivotindices[1]
    j = pivotindices[2]

    if size(a) != size(ci)
        throw(DimensionMismatch(
            "This matrix doesn't match the MatrixCrossInterpolation object. Their sizes
            mismatch: $(size(a)) != $(size(ci))."))
    elseif (i < 0) || (i > nrows(ci)) || (j < 0) || (j > ncols(ci))
        throw(BoundsError(
            "Cannot add a pivot at indices ($i, $j): These indices are out of bounds for a
            $(nrows(ci)) * $(ncols(ci)) matrix."))
    elseif i in ci.rowindices
        throw(ArgumentError(
            "Cannot add a pivot at indices ($i, $j) because row $i already has a pivot."))
    elseif j in ci.colindices
        throw(ArgumentError(
            "Cannot add a pivot at indices ($i, $j) because col $j already has a pivot."))
    end


    addpivotrow!(ci, a, pivotindices[1])
    addpivotcol!(ci, a, pivotindices[2])
end

"""
    function addpivot(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T})

Add a new pivot that maximizes the error to the cross interpolation `ci` of the matrix `a`.
"""
function addpivot!(
    ci::MatrixCI{T},
    a::AbstractMatrix{T}
) where {T}
    addpivot!(ci, a, findnewpivot(ci, a)[1])
end

"""
    function crossinterpolate(
        a::AbstractMatrix{T};
        tolerance=1e-6,
        maxiter=200,
        firstpivot=argmax(abs.(a))
    ) where {T}

Cross interpolate a ``m\\times n`` matrix ``a``, i.e. find a set of indices ``I`` and ``J``,
such that
``a(1\\ldots m, 1\\ldots n) \\approx a(1\\ldots m, I) (a(I, J))^{-1} a(J, 1\\ldots m)``.
"""
function crossinterpolate(
    a::CuMatrix{T};
    tolerance=1e-6,
    maxiter=200,
    firstpivot=argmax(abs.(a))
) where {T}
    ci = MatrixCI(a, firstpivot)
    for iter in 1:maxiter
        pivoterror, newpivot = findmax(localerror(ci, a))
        if pivoterror < tolerance
            return ci
        end
        addpivot!(ci, a, newpivot)
    end
    return ci
end

function  crossinterpolate(
    a::AbstractMatrix{T};
    tolerance=1e-6,
    maxiter=200,
    firstpivot=argmax(abs.(a))
) where {T}
    return crossinterpolate(CuMatrix(a), tolerance=tolerance, maxiter=maxiter, firstpivot=firstpivot)
end
