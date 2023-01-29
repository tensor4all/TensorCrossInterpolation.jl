mutable struct MatrixACA{T} <: AbstractMatrixCI{T}
    rowindices::Vector{Int}
    colindices::Vector{Int}

    u::Matrix{T} # u_k(x): (nrows, npivot)
    v::Matrix{T} # v_k(y): (npivot, ncols)
    alpha::Vector{T}

    function MatrixACA{T}(
        nrows::Int, ncols::Int
    ) where {T<:Number}
        return new{T}(Int[], Int[], zeros(nrows, 0), zeros(0, ncols), T[])
    end
end

ncols(aca::MatrixACA) = size(aca.u, 1)
nrows(aca::MatrixACA) = size(aca.v, 2)
npivots(aca::MatrixACA) = size(aca.u, 2)

function adaptivecrossinterpolate(
    ::Type{T},
    tolerance=1e-6,
    maxiter=200,
    firstpivot=CartesianIndex(1,1)) where T
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
        @. result -= (v[l,yk]/u[xl,l]) * u[:, l]
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
        @. result -= (u[xk,l]/u[xl,l]) * v[l, :]
    end
    return result
end


function addpivot!(
    a::AbstractMatrix{T},
    aca::MatrixACA{T},
    pivotindices::Union{CartesianIndex{2},Tuple{Int,Int},Pair{Int,Int}}) where T
    i, j = pivotindices
    push!(aca.rowindices, i)
    push!(aca.colindices, j)
    aca.u = hcat(aca.u, uk(aca, a))
    aca.v = vcat(aca.v, transpose(vk(aca, a)))
    aca.alpha = vcat(aca.alpha, 1/aca.u[i, end])
    return nothing
end


function evaluate(aca::MatrixACA{T})::Matrix{T} where T
    return aca.u * Diagonal(aca.alpha) * aca.v
end


function evaluate(aca::MatrixACA{T}, i::Int, j::Int)::T where {T}
    if isempty(aca)
        return T(0)
    else
        np = npivots(aca)
        left = reshape(view(aca.u, i, :), 1, np)
        right = reshape(view(aca.v, :, j), np, 1)
        return (left * Diagonal(aca.alpha) * right)[1,1]
    end
end