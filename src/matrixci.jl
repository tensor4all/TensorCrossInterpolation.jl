
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
methods such as eval(c, i, j) to get interpolated values, and be improved by
dynamically adding pivots.
"""
mutable struct MatrixCI{T}
    "Same as `CrossData.Iset` in xfac code, or ``\\mathcal{I}`` in TCI paper."
    row_indices::Vector{Int}
    "Same as `CrossData.Jset` in xfac code, or ``\\mathcal{J}`` in TCI paper."
    col_indices::Vector{Int}
    "Same as `CrossData.C` in xfac code, or ``A(\\mathbb{I}, \\mathcal{J})`` in TCI paper."
    pivot_cols::Matrix{T}
    "Same as `CrossData.R` in xfac code, or ``A(\\mathcal{I}, \\mathbb{J})`` in TCI paper."
    pivot_rows::Matrix{T}

    function MatrixCI{T}(
        n_rows::Int, n_cols::Int
    ) where {T<:Number}
        return new{T}([], [], zeros(n_rows, 0), zeros(0, n_cols))
    end

    function MatrixCI(
        row_indices::AbstractVector{Int}, col_indices::AbstractVector{Int},
        pivot_cols::AbstractMatrix{T}, pivot_rows::AbstractMatrix{T}
    ) where {T<:Number}
        return new{T}(row_indices, col_indices, pivot_cols, pivot_rows)
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
    return ci.row_indices
end

function Jset(ci::MatrixCI{T}) where {T}
    return ci.col_indices
end

function n_rows(ci::MatrixCI{T}) where {T}
    return size(ci.pivot_cols, 1)
end

function n_cols(ci::MatrixCI{T}) where {T}
    return size(ci.pivot_rows, 2)
end

function Base.size(ci::MatrixCI{T}) where {T}
    return n_rows(ci), n_cols(ci)
end

function pivot_matrix(ci::MatrixCI{T}) where {T}
    return ci.pivot_cols[ci.row_indices, :]
end

function left_matrix(ci::MatrixCI{T}) where {T}
    return AtimesBinv(ci.pivot_cols, pivot_matrix(ci))
end

function right_matrix(ci::MatrixCI{T}) where {T}
    return AinvtimesB(pivot_matrix(ci), ci.pivot_rows)
end

function avail_rows(ci::MatrixCI{T}) where {T}
    return setdiff(1:n_rows(ci), ci.row_indices)
end

function avail_cols(ci::MatrixCI{T}) where {T}
    return setdiff(1:n_cols(ci), ci.col_indices)
end

function rank(ci::MatrixCI{T}) where {T}
    return length(ci.row_indices)
end

function isempty(ci::MatrixCI{T}) where {T}
    return Base.isempty(ci.pivot_cols)
end

function first_pivot_value(ci::MatrixCI{T}) where {T}
    return isempty(ci) ? 1.0 : ci.pivot_cols[ci.row_indices[1], 1]
end

function eval(ci::MatrixCI{T}, i::Int, j::Int) where {T}
    if isempty(ci)
        return T(0)
    else
        return dot(left_matrix(ci)[i, :], ci.pivot_rows[:, j])
    end
end

function length_or_default(c::Colon, default)
    return default
end

function length_or_default(c, default)
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
            length_or_default(rows, n_rows(ci)),
            length_or_default(cols, n_cols(ci)))
    else
        return left_matrix(ci)[rows, :] * ci.pivot_rows[:, cols]
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
    return eval(ci, i, j)
end

function Base.isapprox(
    lhs::MatrixCI{T}, rhs::MatrixCI{T}
) where {T}
    return (lhs.col_indices == rhs.col_indices) &&
           (lhs.row_indices == rhs.row_indices) &&
           Base.isapprox(lhs.pivot_cols, rhs.pivot_cols) &&
           Base.isapprox(lhs.pivot_rows, rhs.pivot_rows)
end

function Base.Matrix(ci::MatrixCI{T}) where {T}
    return left_matrix(ci) * ci.pivot_rows
end

"""
    local_error(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T},
        row_indices::Union{AbstractVector{Int},Colon,Int}=Colon(),
        col_indices::Union{AbstractVector{Int},Colon,Int}=Colon())
    
    Returns all local errors of the cross interpolation ci with respect to
    matrix a. To find local errors on a submatrix, specify row_indices or
    col_indices.
"""
function local_error(
    a::AbstractMatrix{T},
    ci::MatrixCI{T},
    row_indices::Union{AbstractVector{Int},Colon,Int}=Colon(),
    col_indices::Union{AbstractVector{Int},Colon,Int}=Colon()
) where {T}
    return abs.(a[row_indices, col_indices] .- ci[row_indices, col_indices])
end

"""
    find_new_pivot(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T},
        row_indices::Union{Vector{Int},Colon},
        col_indices::Union{Vector{Int},Colon})
    
    Finds the pivot that maximizes the local error across all components of
    `a` and `ci` within the subset given by `row_indices` and `col_indices`. By
    default, all avalable rows of `ci` will be considered.
"""
function find_new_pivot(
    a::AbstractMatrix{T},
    ci::MatrixCI{T},
    row_indices::Union{Vector{Int},Colon}=avail_rows(ci),
    col_indices::Union{Vector{Int},Colon}=avail_cols(ci)
) where {T}
    if rank(ci) == minimum(size(a))
        throw(ArgumentError(
            "Cannot find a new pivot for this MatrixCrossInterpolation, as it is
            already full rank."))
    elseif (length(row_indices) == 0)
        throw(ArgumentError(
            "Cannot find a new pivot in an empty set of rows
            (row_indices = $row_indices)"
        ))
    elseif (length(col_indices) == 0)
        throw(ArgumentError(
            "Cannot find a new pivot in an empty set of cols
            (col_indices = $row_indices)"
        ))
    end

    localerrors = local_error(a, ci, row_indices, col_indices)
    ijraw = argmax(localerrors)
    return (row_indices[ijraw[1]], col_indices[ijraw[2]]), localerrors[ijraw]
end

"""
    function add_pivot(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T},
        pivot_indices::AbstractArray{T, 2})
    
    Add a new pivot given by `pivot_indices` to the cross interpolation `ci` of
    the matrix `a`.
"""
function add_pivot!(
    a::AbstractMatrix{T},
    ci::MatrixCI{T},
    pivot_indices::Union{CartesianIndex{2},Tuple{Int,Int},Pair{Int,Int}}
) where {T}
    i = pivot_indices[1]
    j = pivot_indices[2]

    if size(a) != size(ci)
        throw(DimensionMismatch(
            "This matrix doesn't match the MatrixCrossInterpolation object.
            Their sizes mismatch: $(size(a)) != $(size(ci))."))
    elseif (i < 0) || (i > n_rows(ci)) || (j < 0) || (j > n_cols(ci))
        throw(BoundsError(
            "Cannot add a pivot at indices ($i, $j): These indices are out of
            bounds for a $(n_rows(ci)) * $(n_cols(ci)) matrix."))
    elseif i in ci.row_indices
        throw(ArgumentError(
            "Cannot add a pivot at indices ($i, $j) because row $i already has a
            pivot."))
    elseif j in ci.col_indices
        throw(ArgumentError(
            "Cannot add a pivot at indices ($i, $j) because col $j already has a
            pivot."))
    end

    row = transpose(a[i, :])
    ci.pivot_rows = vcat(ci.pivot_rows, row)
    push!(ci.row_indices, i)
    col = a[:, j]
    ci.pivot_cols = hcat(ci.pivot_cols, col)
    push!(ci.col_indices, j)
end

"""
    function add_pivot(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T})
    
    Add a new pivot that maximizes the error to the cross interpolation `ci` of
    the matrix `a`.
"""
function add_pivot!(
    a::AbstractMatrix{T},
    ci::MatrixCI{T}
) where {T}
    return add_pivot!(a, ci, find_new_pivot(a, ci)[1])
end

function cross_interpolate(
    a::AbstractMatrix{T};
    tolerance=1e-6,
    maxiter=200,
    firstpivot=argmax(abs.(a))
) where {T}
    ci = MatrixCI(a, firstpivot)
    for iter in 1:maxiter
        pivoterror, newpivot = findmax(local_error(a, ci))
        if pivoterror < tolerance
            return ci
        end
        add_pivot!(a, ci, newpivot)
    end
    return ci
end
