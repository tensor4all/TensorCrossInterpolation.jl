abstract type AbstractMatrixCI{T} end

function Base.size(ci::AbstractMatrixCI)
    return nrows(ci), ncols(ci)
end

function row(
    ci::AbstractMatrixCI{T},
    i::Int;
    cols::Union{AbstractVector{Int},Colon}=Colon()
) where {T}
    return submatrix(ci, [i], cols)[:]
end

function col(
    ci::AbstractMatrixCI{T},
    j::Int;
    rows::Union{AbstractVector{Int},Colon}=Colon()
) where {T}
    return submatrix(ci, rows, [j])[:]
end

function Base.getindex(
    ci::AbstractMatrixCI{T},
    rows::Union{AbstractVector{Int},Colon},
    cols::Union{AbstractVector{Int},Colon}
) where {T}
    return submatrix(ci, rows, cols)
end

function Base.getindex(
    ci::AbstractMatrixCI{T},
    i::Int,
    cols::Union{AbstractVector{Int},Colon}
) where {T}
    return row(ci, i; cols=cols)
end

function Base.getindex(
    ci::AbstractMatrixCI{T},
    rows::Union{AbstractVector{Int},Colon},
    j::Int
) where {T}
    return col(ci, j; rows=rows)
end

function Base.getindex(ci::AbstractMatrixCI{T}, i::Int, j::Int) where {T}
    return evaluate(ci, i, j)
end

"""
    localerror(
        a::AbstractMatrix{T},
        ci::MatrixCrossInterpolation{T},
        row_indices::Union{AbstractVector{Int},Colon,Int}=Colon(),
        col_indices::Union{AbstractVector{Int},Colon,Int}=Colon())

Returns all local errors of the cross interpolation ci with respect to matrix a. To find
local errors on a submatrix, specify row_indices or col_indices.
"""
function localerror(
    ci::AbstractMatrixCI{T},
    a::CUDA.CuMatrix{T},
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

Finds the pivot that maximizes the local error across all components of `a` and `ci` within
the subset given by `rowindices` and `colindices`. By default, all avalable rows of `ci`
will be considered.
"""
function findnewpivot(
    ci::AbstractMatrixCI{T},
    a::AbstractMatrix{T},
    rowindices::Union{Vector{Int},Colon}=availablerows(ci),
    colindices::Union{Vector{Int},Colon}=availablecols(ci)
) where {T}
    if rank(ci) == minimum(size(a))
        throw(ArgumentError(
            "Cannot find a new pivot for this MatrixCrossInterpolation, as it is
            already full rank."))
    elseif (length(rowindices) == 0)
        throw(ArgumentError(
            "Cannot find a new pivot in an empty set of rows (row_indices = $rowindices)"
        ))
    elseif (length(colindices) == 0)
        throw(ArgumentError(
            "Cannot find a new pivot in an empty set of cols (col_indices = $rowindices)"
        ))
    end

    localerrors = localerror(ci, a, rowindices, colindices)
    ij = argmax(localerrors)
    CUDA.@allowscalar return (rowindices[ij[1]], colindices[ij[2]]), localerrors[ij]
end
