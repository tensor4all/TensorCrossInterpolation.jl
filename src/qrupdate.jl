using LinearAlgebra

function extendrows(A::Matrix, N::Int)
    return [A; zeros(N - size(A, 1), size(A, 2))]
end

function getwide(Q::LinearAlgebra.QRCompactWYQ, R::Matrix)
    # This will convert Q into a wide matrix.
    # Note that the usual convert(Matrix, Q) will result in a thin matrix.
    Qmat = Q * I
    Rmat = extendrows(R, size(Q, 1))
    return Qmat, Rmat
end

function getwide(QR::LinearAlgebra.QRCompactWY)
    return getwide(QR.Q, QR.R)
end

"""
    addcol!(Q::Matrix, R::Matrix, newcol::Vector, colindex::Int)

    Given a QR decomposition of a matrix ``A = QR``, update ``Q`` and ``R`` to be the QR
    decomposition of a matrix with a column `newcol` inserted after the column designated by
    `colindex`.

    This will overwrite `Q` and return `R`.
"""
function addcol!(Q::Matrix, R::Matrix, newcol::Vector, colindex::Int)
    w = Q' * newcol
    R = [R[:, 1:colindex] w R[:, colindex+1:end]]
    for rowindex in (size(R, 1)-1):-1:(colindex+1)
        G, r = givens(R, rowindex, rowindex + 1, colindex + 1)
        lmul!(G, R)
        rmul!(Q, G')
    end
    return R
end

"""
    addcol(Q::Matrix, R::Matrix, newcol::Vector, colindex::Int)

    Given a QR decomposition of a matrix ``A = QR``, update ``Q`` and ``R`` to be the QR
    decomposition of a matrix with a column `newcol` inserted after the column designated by
    `colindex`.

    This is the non-updating version; returns updated `Q` and `R`.
"""
function addcol(Q::Matrix, R::Matrix, newcol::Vector, colindex::Int)
    Qtemp = copy(Q)
    Rtemp = addcol!(Qtemp, extendrows(R, size(Q, 1)), newcol, colindex)
    return Qtemp, Rtemp
end

"""
    addcol(Q::Matrix, R::Matrix, newcol::Vector, colindex::Int)

    Given a QR decomposition of a matrix ``A = QR``, update ``Q`` and ``R`` to be the QR
    decomposition of a matrix with a column `newcol` inserted after the column designated by
    `colindex`.

    This is the non-updating version; returns updated `Q` and `R`.
"""
function addcol(
    Q::LinearAlgebra.QRCompactWYQ,
    R::Matrix,
    newcol::Vector,
    colindex::Int
)
    Qtemp, Rtemp = getwide(Q, R)
    Rtemp = addcol!(Qtemp, Rtemp, newcol, colindex)
    return Qtemp, Rtemp
end

"""
    addrow(Q::Matrix, R::Matrix, newrow::Vector, rowindex::Int)

    Given a QR decomposition of a matrix ``A = QR``, update ``Q`` and ``R`` to be the QR
    decomposition of a matrix with a row `newrow` inserted after the row designated by
    `rowindex`.

    Returns updated `Q` and `R`.
"""
function addrow(Q::Matrix, R::Matrix, newrow::Vector, rowindex::Int)
    Q = [
        1 zeros(size(Q, 2))'
        zeros(size(Q, 1)) Q
    ]
    R = [
        newrow'
        R
    ]
    for colindex in 1:size(R, 2)-1
        G, r = givens(R, colindex, colindex + 1, colindex)
        lmul!(G, R)
        rmul!(Q, G')
    end
    # This is different to 12.5.3 in Golub and van Loan: Matrix computations.
    # Either I'm reading the equation wrong, or there is a mistake in the book.
    Q = [Q[2:rowindex+1, :]; Q[1, :]'; Q[rowindex+2:end, :]]
    return Q, R
end

"""
    addrow(Q::Matrix, R::Matrix, newrow::Vector, rowindex::Int)

    Given a QR decomposition of a matrix ``A = QR``, update ``Q`` and ``R`` to be the QR
    decomposition of a matrix with a row `newrow` inserted after the row designated by
    `rowindex`.

    Returns updated `Q` and `R`.
"""
function addrow(Q::LinearAlgebra.QRCompactWYQ, R::Matrix, newrow::Vector, rowindex::Int)
    return addrow(getwide(Q, R)..., newrow, rowindex)
end
