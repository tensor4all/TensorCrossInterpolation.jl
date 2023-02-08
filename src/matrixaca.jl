mutable struct MatrixACA{T} <: AbstractMatrixCI{T}
    rowindices::Vector{Int}
    colindices::Vector{Int}

    u::Matrix{T} # u_k(x): (nrows, npivot)
    v::Matrix{T} # v_k(y): (npivot, ncols)
    alpha::Vector{T} # 1/δ: npivot

    function MatrixACA{T}(
        nrows::Int, ncols::Int
    ) where {T<:Number}
        return new{T}(Int[], Int[], zeros(nrows, 0), zeros(0, ncols), T[])
    end

    function MatrixACA{T}(
        A::AbstractMatrix{T},
        firstpivot::Union{CartesianIndex{2},Tuple{Int,Int},Pair{Int,Int}}
    ) where {T<:Number}
        return new{T}(
            [firstpivot[1]], [firstpivot[2]],
            A[:, [firstpivot[2]]], A[[firstpivot[1]], :],
            [1 / A[firstpivot[1], firstpivot[2]]])
    end
end

ncols(aca::MatrixACA) = size(aca.u, 1)
nrows(aca::MatrixACA) = size(aca.v, 2)
npivots(aca::MatrixACA) = size(aca.u, 2)

function availablerows(aca::MatrixACA{T}) where {T}
    return setdiff(1:nrows(aca), aca.rowindices)
end

function availablecols(aca::MatrixACA{T}) where {T}
    return setdiff(1:ncols(aca), aca.colindices)
end

"""
Compute u_k(x) for all x
"""
function uk(aca::MatrixACA{T}, A) where {T}
    k = length(aca.colindices)
    yk = aca.colindices[end]
    result = copy(A[:, yk])
    u, v = aca.u, aca.v
    for l in 1:k-1
        xl = aca.rowindices[l]
        @. result -= (v[l, yk] / u[xl, l]) * u[:, l]
    end
    return result
end

"""
Compute v_k(y) for all y
"""
function vk(aca::MatrixACA{T}, A) where {T}
    k = length(aca.rowindices)
    xk = aca.rowindices[end]
    result = copy(A[xk, :])
    u, v = aca.u, aca.v
    for l in 1:k-1
        xl = aca.rowindices[l]
        @. result -= (u[xk, l] / u[xl, l]) * v[l, :]
    end
    return result
end

"""
    function addpivot!(a::AbstractMatrix{T}, aca::MatrixACA{T})

Find and add a new pivot according to the ACA algorithm in Kumar 2016 ()
"""
function addpivot!(
    a::AbstractMatrix{T},
    aca::MatrixACA{T},
    pivotindices::Union{CartesianIndex{2},Tuple{Int,Int},Pair{Int,Int},Nothing}=nothing
) where {T}
    xk::Int = 0
    yk::Int = 0

    # Add col
    if isnothing(pivotindices)
        availcols = availablecols(aca)
        yk = availcols[argmax(abs.(aca.v[end, availcols]))]
    else
        yk = pivotindices[2]
    end
    push!(aca.colindices, yk)
    aca.u = hcat(aca.u, uk(aca, a))

    # Add row
    if isnothing(pivotindices)
        availrows = availablerows(aca)
        xk = availrows[argmax(abs.(aca.u[availrows, end]))]
    else
        xk = pivotindices[1]
    end
    push!(aca.rowindices, xk)
    aca.v = vcat(aca.v, transpose(vk(aca, a)))

    push!(aca.alpha, 1 / aca.u[xk, end])
end

function evaluate(aca::MatrixACA{T})::Matrix{T} where {T}
    return aca.u * Diagonal(aca.alpha) * aca.v
end

function evaluate(aca::MatrixACA{T}, i::Int, j::Int)::T where {T}
    if isempty(aca)
        return T(0)
    else
        np = npivots(aca)
        left = reshape(view(aca.u, i, :), 1, np)
        right = reshape(view(aca.v, :, j), np, 1)
        return (left*Diagonal(aca.alpha)*right)[1, 1]
    end
end

function adaptivecrossinterpolate(
    a::AbstractMatrix{T};
    tolerance=1e-6,
    maxiter=200,
    firstpivot=argmax(abs.(a))
) where {T}
    aca = MatrixACA(a, firstpivot)
end
