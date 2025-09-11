function distribute(_batchf, full, fragmented, teamsize, teamrank, stripes, subcomm)
    chunksize, remainder = divrem(length(fragmented), teamsize)
    if stripes == :vertical
        col_chunks = [chunksize + (i <= remainder ? 1 : 0) for i in 1:teamsize]
        row_chunks = [length(full) for _ in 1:teamsize]
        ranges = [(vcat([0],cumsum(col_chunks))[i] + 1):(cumsum(col_chunks)[i]) for i in 1:teamsize]
    else # horizontal
        row_chunks = [chunksize + (i <= remainder ? 1 : 0) for i in 1:teamsize]
        col_chunks = [length(full) for _ in 1:teamsize]
        ranges = [(vcat([0],cumsum(row_chunks))[i] + 1):(cumsum(row_chunks)[i]) for i in 1:teamsize]
    end
    fragment = fragmented[ranges[teamrank]]
    localsubmatrix = if isempty(fragment)
        zeros(length(full), 0)
    else
        if stripes == :vertical
            _batchf(full, fragment)
        else
            _batchf(fragment, full)
        end
    end
    if teamrank == 1
        submatrix = if stripes == :vertical
            zeros(length(full), length(fragmented))
        else
            zeros(length(fragmented), length(full))
        end
        sizes = vcat(row_chunks', col_chunks')
        counts = vec(prod(sizes, dims=1))
        submatrix_vbuf = VBuffer(submatrix, counts)
    else
        submatrix_vbuf = VBuffer(nothing)
    end
    MPI.Gatherv!(localsubmatrix, submatrix_vbuf, 0, subcomm)
    if teamrank != 1
        submatrix = nothing
    end
    submatrix
end


function submatrixargmax(
    f::Function, # real valued function
    A::AbstractMatrix{T},
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange};
    colmask::Function=x->true,
    rowmask::Function=x->true
) where {T}
    m = typemin(f(first(A)))
    !isempty(rows) || throw(ArgumentError("rows must not be empty"))
    !isempty(cols) || throw(ArgumentError("cols must not be empty"))
    mr = first(rows)
    mc = first(cols)
    rows ⊆ axes(A, 1) || throw(ArgumentError("rows ⊆ axes(A, 1) must be satified"))
    cols ⊆ axes(A, 2) || throw(ArgumentError("cols ⊆ axes(A, 2) must be satified"))
    @inbounds for c in cols
        if !colmask(c)
            continue
        end
        for r in rows
            if !rowmask(r)
                continue
            end
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

function submatrixargmax(f::Function, A::AbstractMatrix, rows, cols; colmask::Function=x->true, rowmask::Function=x->true)
    function convertarg(arg::Int, size::Int)
        return [arg]
    end

    function convertarg(arg::Colon, size::Int)
        return 1:size
    end

    function convertarg(arg::Union{AbstractVector,UnitRange}, size::Int)
        return arg
    end

    return submatrixargmax(f, A, convertarg(rows, size(A, 1)), convertarg(cols, size(A, 2)); colmask=colmask, rowmask=rowmask)
end

function submatrixargmax(A::AbstractMatrix, rows, cols)
    submatrixargmax(identity, A::AbstractMatrix, rows, cols)
end


function submatrixargmax(f::Function, A::AbstractMatrix, startindex::Int; colmask::Function=x->true, rowmask::Function=x->true)
    return submatrixargmax(f, A, startindex:size(A, 1), startindex:size(A, 2); colmask=colmask, rowmask=rowmask)
end

function submatrixargmax(A::AbstractMatrix, startindex::Int)
    return submatrixargmax(identity, A, startindex:size(A, 1), startindex:size(A, 2))
end

mutable struct rrLU{T}
    rowpermutation::Vector{Int}
    colpermutation::Vector{Int}
    L::Matrix{T}
    U::Matrix{T}
    leftorthogonal::Bool
    npivot::Int
    error::Float64

    function rrLU(rowpermutation::Vector{Int}, colpermutation::Vector{Int}, L::Matrix{T}, U::Matrix{T}, leftorthogonal, npivot, error) where {T}
        npivot == size(L, 2) || error("L must have the same number of columns as the number of pivots.")
        npivot == size(U, 1) || error("U must have the same number of rows as the number of pivots.")
        length(rowpermutation) == size(L, 1) || error("rowpermutation must have length equal to the number of pivots.")
        length(colpermutation) == size(U, 2) || error("colpermutation must have length equal to the number of pivots.")
        new{T}(rowpermutation, colpermutation, L, U, leftorthogonal, npivot, error)
    end

    function rrLU{T}(nrows::Int, ncols::Int; leftorthogonal::Bool=true) where {T}
        new{T}(1:nrows, 1:ncols, zeros(nrows, 0), zeros(0, ncols), leftorthogonal, 0, NaN)
    end
end


function rrLU{T}(A::AbstractMatrix{T}; leftorthogonal::Bool=true) where {T}
    rrLU{T}(size(A)...; leftorthogonal=leftorthogonal)
end

function swaprow!(lu::rrLU{T}, A::AbstractMatrix{T}, a, b) where {T}
    lurp = lu.rowpermutation
    lurp[a], lurp[b] = lurp[b], lurp[a]
    @inbounds for j in axes(A, 2)
        A[a, j], A[b, j] = A[b, j], A[a, j]
    end
end

function swapcol!(lu::rrLU{T}, A::AbstractMatrix{T}, a, b) where {T}
    lucp = lu.colpermutation
    lucp[a], lucp[b] = lucp[b], lucp[a]
    @inbounds for i in axes(A, 1)
        A[i, a], A[i, b] = A[i, b], A[i, a]
    end
end

function addpivot!(lu::rrLU{T}, A::AbstractMatrix{T}, newpivot) where {T}
    k = lu.npivot += 1
    swaprow!(lu, A, k, newpivot[1])
    swapcol!(lu, A, k, newpivot[2])

    if lu.leftorthogonal
        A[k+1:end, k] ./= A[k, k]
    else
        A[k, k+1:end] ./= A[k, k]
    end

    # perform BLAS subroutine manually: A <- -x * transpose(y) + A
    x = @view(A[k+1:end, k])
    y = @view(A[k, k+1:end])
    A = @view(A[k+1:end, k+1:end])
    @inbounds for j in eachindex(axes(A, 2), y)
        for i in eachindex(axes(A, 1), x)
            # update `A[k+1:end, k+1:end]`
            A[i, j] -= x[i] * y[j]
        end
    end
    nothing
end

_count_cols_selected(lu, cols)::Int = sum([c ∈ lu.colpermutation[1:lu.npivot] for c ∈ cols])
_count_rows_selected(lu, rows)::Int = sum([r ∈ lu.rowpermutation[1:lu.npivot] for r ∈ rows])

function _optimizerrlu!(
    lu::rrLU{T},
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0
) where {T}
    maxrank = min(maxrank, size(A, 1), size(A, 2))
    maxerror = 0.0
    while lu.npivot < maxrank
        k = lu.npivot + 1
        newpivot = submatrixargmax(abs2, A, k)
        lu.error = abs(A[newpivot[1], newpivot[2]])
        # Add at least 1 pivot to get a well-defined L * U
        if (abs(lu.error) <= reltol * maxerror || abs(lu.error) <= abstol) && lu.npivot > 0
            break
        end
        maxerror = max(maxerror, lu.error)
        addpivot!(lu, A, newpivot)
    end

    lu.L = tril(@view A[:, 1:lu.npivot])
    lu.U = triu(@view A[1:lu.npivot, :])
    if any(isnan.(lu.L))
        error("lu.L contains NaNs")
    end
    if any(isnan.(lu.U))
        error("lu.U contains NaNs")
    end
    if lu.leftorthogonal
        lu.L[diagind(lu.L)] .= one(T)
    else
        lu.U[diagind(lu.U)] .= one(T)
    end

    if lu.npivot >= minimum(size(A))
        lu.error = 0
    end

    nothing
end

"""
    function rrlu!(
        A::AbstractMatrix{T};
        maxrank::Int=typemax(Int),
        reltol::Number=1e-14,
        abstol::Number=0.0,
        leftorthogonal::Bool=true
    )::rrLU{T} where {T}

Rank-revealing LU decomposition.
"""
function rrlu!(
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true
)::rrLU{T} where {T}
    lu = rrLU{T}(A, leftorthogonal=leftorthogonal)
    _optimizerrlu!(lu, A; maxrank=maxrank, reltol=reltol, abstol=abstol)
    return lu
end

"""
    function rrlu(
        A::AbstractMatrix{T};
        maxrank::Int=typemax(Int),
        reltol::Number=1e-14,
        abstol::Number=0.0,
        leftorthogonal::Bool=true
    )::rrLU{T} where {T}

Rank-revealing LU decomposition.
"""
function rrlu(
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true
)::rrLU{T} where {T}
    return rrlu!(copy(A); maxrank, reltol, abstol, leftorthogonal)
end

function arrlu(
    ::Type{ValueType},
    f,
    matrixsize::Tuple{Int,Int},
    I0::AbstractVector{Int}=Int[],
    J0::AbstractVector{Int}=Int[];
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true,
    numrookiter::Int=5,
    usebatcheval::Bool=false,
    leaders::Vector=Int[],
    leaderslist::Vector=Int[],
    subcomm=nothing # TODO change: bad practice
)::rrLU{ValueType} where {ValueType}
    lu = rrLU{ValueType}(matrixsize...; leftorthogonal)
    islowrank = false
    maxrank = min(maxrank, matrixsize...)

    _batchf = usebatcheval ? f : ((x, y) -> f.(x, y'))

    if !isempty(leaders)
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        juliarank = rank + 1
        nprocs = MPI.Comm_size(comm)
        teamrank = MPI.Comm_rank(subcomm)
        teamjuliarank = teamrank + 1
        teamsize = MPI.Comm_size(subcomm)
    end

    while true
        if leftorthogonal
            pushrandomsubset!(J0, 1:matrixsize[2], max(1, length(J0)))
        else
            pushrandomsubset!(I0, 1:matrixsize[1], max(1, length(I0)))
        end
        
        for rookiter in 1:numrookiter
            colmove = (iseven(rookiter) == leftorthogonal)
            if !isempty(leaders) # Parallel
                if teamsize > 1 # Parallel and leadership
                    submatrix = if colmove
                        distribute(_batchf, lu.colpermutation, I0, teamsize, teamjuliarank, :horizontal, subcomm)
                    else
                        distribute(_batchf, lu.rowpermutation, J0, teamsize, teamjuliarank, :vertical, subcomm)
                    end
                    MPI.Barrier(subcomm)
                    if teamjuliarank == 1
                        lu.npivot = 0
                        _optimizerrlu!(lu, submatrix; maxrank, reltol, abstol)
                        islowrank |= npivots(lu) < minimum(size(submatrix))
                    end
                else
                    submatrix = if colmove
                        _batchf(I0, lu.colpermutation)
                    else
                        _batchf(lu.rowpermutation, J0)
                    end
                    lu.npivot = 0
                    _optimizerrlu!(lu, submatrix; maxrank, reltol, abstol)
                    islowrank |= npivots(lu) < minimum(size(submatrix))
                end    
            else # Serial
                submatrix = if colmove
                    _batchf(I0, lu.colpermutation)
                else
                    _batchf(lu.rowpermutation, J0)
                end
                lu.npivot = 0
                _optimizerrlu!(lu, submatrix; maxrank, reltol, abstol)
                islowrank |= npivots(lu) < minimum(size(submatrix))
            end
            if !isempty(leaders) # Parallel
                tb = time_ns()
                lu = MPI.bcast([lu], subcomm)[1]
                islowrank = MPI.bcast([islowrank], subcomm)[1]
                tb = (time_ns() - tb)*1e-9
                if rowindices(lu) == I0 && colindices(lu) == J0
                    break
                end
                J0 = colindices(lu)
                I0 = rowindices(lu)
            else # Serial
                if rowindices(lu) == I0 && colindices(lu) == J0
                    break
                end
                J0 = colindices(lu)
                I0 = rowindices(lu)
            end
        end

        
        if islowrank || length(I0) >= maxrank
            break
        end
    end

    
    if size(lu.L, 1) < matrixsize[1]
        I2 = setdiff(1:matrixsize[1], I0)
        lu.rowpermutation = vcat(I0, I2)
        L2 = if !isempty(leaders)
            distribute(_batchf, J0, I2, teamsize, teamjuliarank, :horizontal, subcomm)
        else
            _batchf(I2, J0)
        end
        if isempty(leaders) || teamjuliarank == 1
            cols2Lmatrix!(L2, (@view lu.U[1:lu.npivot, 1:lu.npivot]), leftorthogonal)
            lu.L = vcat((@view lu.L[1:lu.npivot, 1:lu.npivot]), L2)
        end
    end


    if size(lu.U, 2) < matrixsize[2]
        J2 = setdiff(1:matrixsize[2], J0)
        lu.colpermutation = vcat(J0, J2)
        U2 = if !isempty(leaders)
            distribute(_batchf, I0, J2, teamsize, teamjuliarank, :vertical, subcomm)
        else
            U2 = _batchf(I0, J2)
        end
        if isempty(leaders) || teamjuliarank == 1
            rows2Umatrix!(U2, (@view lu.L[1:lu.npivot, 1:lu.npivot]), leftorthogonal)
            lu.U = hcat((@view lu.U[1:lu.npivot, 1:lu.npivot]), U2)
        end
    end

    return lu
end

function rrlu(
    ::Type{ValueType},
    f,
    matrixsize::Tuple{Int,Int},
    I0::AbstractVector{Int}=Int[],
    J0::AbstractVector{Int}=Int[];
    pivotsearch=:full,
    kwargs...
)::rrLU{ValueType} where {ValueType}
    if pivotsearch === :rook
        return arrlu(ValueType, f, matrixsize, I0, J0; kwargs...)
    elseif pivotsearch === :full
        A = f.(1:matrixsize[1], collect(1:matrixsize[2])')
        return rrlu!(A; kwargs...)
    else
        throw(ArgumentError("Unknown pivot search strategy $pivotsearch. Choose between :rook and :full."))
    end
end

function cols2Lmatrix!(C::AbstractMatrix, P::AbstractMatrix, leftorthogonal::Bool)
    if size(C, 2) != size(P, 2)
        throw(DimensionMismatch("C and P matrices must have same number of columns in `cols2Lmatrix!`."))
    elseif size(P, 1) != size(P, 2)
        throw(DimensionMismatch("P matrix must be square in `cols2Lmatrix!`."))
    end

    for k in axes(P, 1)
        C[:, k] ./= P[k, k]
        # C[:, k+1:end] .-= C[:, k] * transpose(P[k, k+1:end])
        x = @view C[:, k]
        y = @view P[k, k+1:end]
        C̃ = @view C[:, k+1:end]
        @inbounds for j in eachindex(axes(C̃, 2), y)
            for i in eachindex(axes(C̃, 1), x)
                # update `C[:, k+1:end]`
                C̃[i, j] -= x[i] * y[j]
            end
        end
    end
    return C
end

function rows2Umatrix!(R::AbstractMatrix, P::AbstractMatrix, leftorthogonal::Bool)
    if size(R, 1) != size(P, 1)
        throw(DimensionMismatch("R and P matrices must have same number of rows in `rows2Umatrix!`."))
    elseif size(P, 1) != size(P, 2)
        throw(DimensionMismatch("P matrix must be square in `rows2Umatrix!`."))
    end

    for k in axes(P, 1)
        R[k, :] ./= P[k, k]
        # R[k+1:end, :] -= P[k+1:end, k] * transpose(R[k, :])
        x = @view P[k+1:end, k]
        y = @view R[k, :]
        R̃ = @view R[k+1:end, :]
        @inbounds for j in eachindex(axes(R̃, 2), y)
            for i in eachindex(axes(R̃, 1), x)
                # update R[k+1:end, :]
                R̃[i, j] -= x[i] * y[j]
            end
        end
    end
    return R
end

function size(lu::rrLU{T}) where {T}
    return size(lu.L, 1), size(lu.U, 2)
end

function size(lu::rrLU{T}, dim) where {T}
    if dim == 1
        return size(lu.L, 1)
    elseif dim == 2
        return size(lu.U, 2)
    else
        return 1
    end
end

function left(lu::rrLU{T}; permute=true) where {T}
    if permute
        l = Matrix{T}(undef, size(lu.L)...)
        l[lu.rowpermutation, :] = lu.L
        return l
    else
        return lu.L
    end
end

function right(lu::rrLU{T}; permute=true) where {T}
    if permute
        u = Matrix{T}(undef, size(lu.U)...)
        u[:, lu.colpermutation] = lu.U
        return u
    else
        return lu.U
    end
end

function diag(lu::rrLU{T}) where {T}
    if lu.leftorthogonal
        return diag(lu.U[1:lu.npivot, 1:lu.npivot])
    else
        return diag(lu.L[1:lu.npivot, 1:lu.npivot])
    end
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
    return vcat(abs.(diag(lu)), lu.error)
end

"""
Correct estimate of the last pivot error
A special care is taken for a full-rank matrix: the last pivot error is set to zero.
"""
function lastpivoterror(lu::rrLU{T})::Float64 where {T}
    return lu.error
end



"""
Solve (LU) x = b

L: lower triangular matrix
U: upper triangular matrix
b: right-hand side vector

Return x

Note: Not optimized for performance
"""
function solve(L::Matrix{T}, U::Matrix{T}, b::Matrix{T}) where{T}
    N1, N2, N3 = size(L, 1), size(L, 2), size(U, 2)
    M = size(b, 2)
    
    # Solve Ly = b
    y = zeros(T, N2, M)
    for i = 1:N2
        y[i, :] .= b[i, :]
        for k in 1:M
            for j in 1:i-1
                y[i, k] -= L[i, j] * y[j, k]
            end
        end
        y[i, :] ./= L[i, i]
    end

    # Solve Ux = y
    x = zeros(T, N3, M)
    for i = N3:-1:1
        x[i, :] .= y[i, :]
        for k in 1:M
            for j = i+1:N3
                x[i, k] -= U[i, j] * x[j, k]
            end
        end
        x[i, :] ./= U[i, i]
    end

    return x
end


"""
Override solving Ax = b using LU decomposition
"""
function Base.:\(A::rrLU{T}, b::AbstractMatrix{T}) where{T}
    size(A, 1) == size(A, 2) || error("Matrix must be square.")
    A.npivot == size(A, 1) || error("rank-deficient matrix is not supportred!")
    b_perm = b[A.rowpermutation, :]
    x_perm = solve(A.L, A.U, b_perm)
    x = similar(x_perm)
    for i in 1:size(x, 1)
        x[A.colpermutation[i], :] .= x_perm[i, :]
    end
    return x
end


function Base.transpose(A::rrLU{T}) where{T}
    return rrLU(
        A.colpermutation, A.rowpermutation,
        Matrix(transpose(A.U)), Matrix(transpose(A.L)), !A.leftorthogonal, A.npivot, A.error)
end