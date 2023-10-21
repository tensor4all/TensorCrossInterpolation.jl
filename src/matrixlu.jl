function submatrixargmax(
    f::Function, # real valued function
    A::AbstractMatrix{T},
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange},
) where {T}
    m = typemin(f(first(A)))
    !isempty(rows) || throw(ArgumentError("rows must not be empty"))
    !isempty(cols) || throw(ArgumentError("cols must not be empty"))
    mr = first(rows)
    mc = first(cols)
    rows ⊆ axes(A, 1) || throw(ArgumentError("rows ⊆ axes(A, 1) must be satified"))
    cols ⊆ axes(A, 2) || throw(ArgumentError("cols ⊆ axes(A, 2) must be satified"))
    @inbounds for c in cols
        for r in rows
            v = f(A[r, c])
            newm = v > m
            m = ifelse(newm, v, m)
            mr = ifelse(newm, r, mr)
            mc = ifelse(newm, c, mc)
        end
    end
    return mr, mc
end

using LoopVectorization

function submatrixargmax_turbo(
    f::Function, # real valued function
    A::AbstractMatrix{T},
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange},
) where {T}
    m = typemin(f(first(A)))
    !isempty(rows) || throw(ArgumentError("rows must not be empty"))
    !isempty(cols) || throw(ArgumentError("cols must not be empty"))
    mr = first(rows)
    mc = first(cols)
    rows ⊆ axes(A, 1) || throw(ArgumentError("rows ⊆ axes(A, 1) must be satified"))
    cols ⊆ axes(A, 2) || throw(ArgumentError("cols ⊆ axes(A, 2) must be satified"))
    @turbo for c in cols
        for r in rows
            v = f(A[r, c])
            newm = v > m
            m = ifelse(newm, v, m)
            mr = ifelse(newm, r, mr)
            mc = ifelse(newm, c, mc)
        end
    end
    return mr, mc
end

function submatrixargmax_tturbo(
    f::Function, # real valued function
    A::AbstractMatrix{T},
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange},
) where {T}
    m = typemin(f(first(A)))
    !isempty(rows) || throw(ArgumentError("rows must not be empty"))
    !isempty(cols) || throw(ArgumentError("cols must not be empty"))
    mr = first(rows)
    mc = first(cols)
    rows ⊆ axes(A, 1) || throw(ArgumentError("rows ⊆ axes(A, 1) must be satified"))
    cols ⊆ axes(A, 2) || throw(ArgumentError("cols ⊆ axes(A, 2) must be satified"))
    @tturbo for c in cols
        for r in rows
            v = f(A[r, c])
            newm = v > m
            m = ifelse(newm, v, m)
            mr = ifelse(newm, r, mr)
            mc = ifelse(newm, c, mc)
        end
    end
    return mr, mc
end

function submatrixargmax(
    A::AbstractMatrix,
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange},
)
    return submatrixargmax(identity, A, rows, cols)
end

function submatrixargmax(f::Function, A::AbstractMatrix, rows, cols)
    function convertarg(arg::Int, size::Int)
        return [arg]
    end

    function convertarg(arg::Colon, size::Int)
        return 1:size
    end

    function convertarg(arg::Union{AbstractVector,UnitRange}, size::Int)
        return arg
    end

    return submatrixargmax(f, A, convertarg(rows, size(A, 1)), convertarg(cols, size(A, 2)))
end

function submatrixargmax(A::AbstractMatrix, rows, cols)
    submatrixargmax(identity, A::AbstractMatrix, rows, cols)
end


function submatrixargmax(f::Function, A::AbstractMatrix, startindex::Int)
    return submatrixargmax(f, A, startindex:size(A, 1), startindex:size(A, 2))
end

function submatrixargmax_turbo(f::Function, A::AbstractMatrix, startindex::Int)
    return submatrixargmax_turbo(f, A, startindex:size(A, 1), startindex:size(A, 2))
end

function submatrixargmax_tturbo(f::Function, A::AbstractMatrix, startindex::Int)
    return submatrixargmax_tturbo(f, A, startindex:size(A, 1), startindex:size(A, 2))
end

function submatrixargmax(A::AbstractMatrix, startindex::Int)
    return submatrixargmax(identity, A, startindex:size(A, 1), startindex:size(A, 2))
end

using StructArrays

mutable struct rrLU{T, M}
    rowpermutation::Vector{Int}
    colpermutation::Vector{Int}
    buffer::M
    leftorthogonal::Bool
    npivot::Int

    function rrLU{T}(nrows::Int, ncols::Int; leftorthogonal::Bool=true) where {T}
        new{T, Matrix{T}}(1:nrows, 1:ncols, zeros(nrows, ncols), leftorthogonal, 0)
    end

    function rrLU{T}(A::AbstractMatrix{T}; leftorthogonal::Bool=true) where {T}
        new{T, Matrix{T}}(axes(A, 1), axes(A, 2), A, leftorthogonal, 0)
    end

    function rrLU{T}(A::StructArray{T}; leftorthogonal::Bool=true) where {T <: Complex}
        new{T, StructArray{T}}(axes(A, 1), axes(A, 2), A, leftorthogonal, 0)
    end
end

function rrlu(
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true,
    performancemode::Symbol=:tturbo,
)::rrLU{T} where {T}
    maxrank = min(maxrank, size(A)...)
    iscomplexeltype = T <: Complex
    _buffer = iscomplexeltype ? StructArray(A) : copy(A)
    lu = rrLU{T}(_buffer, leftorthogonal=leftorthogonal)
    smfunc = sym2submatriargmax[performancemode]
    apfunc! = sym2addpivot![performancemode]
    for k in 1:maxrank
        lu.npivot = k
        newpivot = if iscomplexeltype
                submatrixargmax(abs2, lu.buffer, k)
            else
                smfunc(abs, lu.buffer, k)
            end
        if k >= 1 &&
            (abs(lu.buffer[newpivot...]) < reltol * abs(lu.buffer[1]) || abs(lu.buffer[newpivot...]) < abstol)
            break
        end
        if iscomplexeltype
            addpivot_turbo!(lu, newpivot)
        else
            apfunc!(lu, newpivot)
        end
    end

    return lu
end

function swaprow!(lu::rrLU{T}, a, b) where {T}
    lu.rowpermutation[[a, b]] = lu.rowpermutation[[b, a]]
    lu.buffer[[a, b], :] .= lu.buffer[[b, a], :]
end

function swapcol!(lu::rrLU{T}, a, b) where {T}
    lu.colpermutation[[a, b]] = lu.colpermutation[[b, a]]
    lu.buffer[:, [a, b]] .= lu.buffer[:, [b, a]]
end

function addpivot!(lu::rrLU{T}, newpivot) where {T}
    k = lu.npivot
    swaprow!(lu, k, newpivot[1])
    swapcol!(lu, k, newpivot[2])

    if lu.leftorthogonal
        lu.buffer[k+1:end, k] ./= lu.buffer[k, k]
    else
        lu.buffer[k, k+1:end] ./= lu.buffer[k, k]
    end

    # perform BLAS subroutine manually: A <- -x * transpose(y) + A
    x = @view(lu.buffer[k+1:end, k])
    y = @view(lu.buffer[k, k+1:end])
    A = @view(lu.buffer[k+1:end, k+1:end])
    @inbounds for j in eachindex(axes(A, 2), y)
        for i in eachindex(axes(A, 1), x)
            # update `lu.buffer[k+1:end, k+1:end]`
            A[i, j] -= x[i] * y[j]
        end
    end
end

function addpivot_turbo!(lu::rrLU{T}, newpivot) where {T}
    k = lu.npivot
    swaprow!(lu, k, newpivot[1])
    swapcol!(lu, k, newpivot[2])

    if lu.leftorthogonal
        lu.buffer[k+1:end, k] ./= lu.buffer[k, k]
    else
        lu.buffer[k, k+1:end] ./= lu.buffer[k, k]
    end

    # perform BLAS subroutine manually: A <- -x * transpose(y) + A
    x = @view(lu.buffer[k+1:end, k])
    y = @view(lu.buffer[k, k+1:end])
    A = @view(lu.buffer[k+1:end, k+1:end])
    @turbo for j in eachindex(axes(A, 2), y)
        for i in eachindex(axes(A, 1), x)
            # update `lu.buffer[k+1:end, k+1:end]`
            A[i, j] -= x[i] * y[j]
        end
    end
end

function addpivot_tturbo!(lu::rrLU{T}, newpivot) where {T}
    k = lu.npivot
    swaprow!(lu, k, newpivot[1])
    swapcol!(lu, k, newpivot[2])

    if lu.leftorthogonal
        lu.buffer[k+1:end, k] ./= lu.buffer[k, k]
    else
        lu.buffer[k, k+1:end] ./= lu.buffer[k, k]
    end

    # perform BLAS subroutine manually: A <- -x * transpose(y) + A
    x = @view(lu.buffer[k+1:end, k])
    y = @view(lu.buffer[k, k+1:end])
    A = @view(lu.buffer[k+1:end, k+1:end])
    @tturbo for j in eachindex(axes(A, 2), y)
        for i in eachindex(axes(A, 1), x)
            # update `lu.buffer[k+1:end, k+1:end]`
            A[i, j] -= x[i] * y[j]
        end
    end
end

function addpivot_turbo!(lu::rrLU{T}, newpivot) where {T<:Complex}
    k = lu.npivot
    swaprow!(lu, k, newpivot[1])
    swapcol!(lu, k, newpivot[2])

    if lu.leftorthogonal
        lu.buffer[k+1:end, k] ./= lu.buffer[k, k]
    else
        lu.buffer[k, k+1:end] ./= lu.buffer[k, k]
    end

    x = @view(lu.buffer[k+1:end, k])
    y = @view(lu.buffer[k, k+1:end])
    A = @view(lu.buffer[k+1:end, k+1:end])
    # perform BLAS subroutine manually: A <- -x * transpose(y) + A
    # for element type T <: Complex
    @turbo for j in eachindex(axes(A.re, 2), axes(A.im, 2), y.re, y.im)
        for i in eachindex(axes(A.re, 1), axes(A.im, 1), x.re, x.im)
            # update `lu.buffer[k+1:end, k+1:end]`
            A.re[i, j] += - x.re[i] * y.re[j] + x.im[i] * y.im[j]
            A.im[i, j] += - x.re[i] * y.im[j] - x.im[i] * y.re[j]
        end
    end
end

function addpivot_tturbo!(lu::rrLU{T}, newpivot) where {T<:Complex}
    k = lu.npivot
    swaprow!(lu, k, newpivot[1])
    swapcol!(lu, k, newpivot[2])

    if lu.leftorthogonal
        lu.buffer[k+1:end, k] /= lu.buffer[k, k]
    else
        lu.buffer[k, k+1:end] /= lu.buffer[k, k]
    end

    x = @view(lu.buffer[k+1:end, k])
    y = @view(lu.buffer[k, k+1:end])
    A = @view(lu.buffer[k+1:end, k+1:end])
    # perform BLAS subroutine manually: A <- -x * transpose(y) + A
    # for element type T <: Complex
    @tturbo for j in eachindex(axes(A.re, 2), axes(A.im, 2), y.re, y.im)
        for i in eachindex(axes(A.re, 1), axes(A.im, 1), x.re, x.im)
            # update `lu.buffer[k+1:end, k+1:end]`
            A.re[i, j] += - x.re[i] * y.re[j] + x.im[i] * y.im[j]
            A.im[i, j] += - x.re[i] * y.im[j] - x.im[i] * y.re[j]
        end
    end
end

const sym2submatriargmax = Dict(
    :inbounds => submatrixargmax,
    :turbo => submatrixargmax_turbo,
    :tturbo => submatrixargmax_tturbo,
)

const sym2addpivot! = Dict(
    :inbounds => addpivot!,
    :turbo => addpivot_turbo!,
    :tturbo => addpivot_tturbo!,
)

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