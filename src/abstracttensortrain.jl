# It is necessary to nest vectors here, as these vectors will frequently grow
# by adding elements. Using matrices, the entire matrix would have to be copied.
# These type definitions are mostly to make the code below more readable, as
# Vector{Vector{Vector{...}}} constructions make it difficult to understand
# the meaning of each dimension.
const LocalIndex = Int
const MultiIndex = Vector{LocalIndex}

"""
    abstract type AbstractTensorTrain{V} <: Function end

Abstract type that is a supertype to all tensor train types found in this module. The main purpose of this type is for the definition of functions such as [`rank`](@ref) and [`linkdims`](@ref) that are shared between different tensor train classes.

When iterated over, the tensor train will return each of the tensors in order.

Implementations: [`TensorTrain`](@ref), [`TensorCI2`](@ref), [`TensorCI1`](@ref)
"""
abstract type AbstractTensorTrain{V} <: Function end

function Base.show(io::IO, tt::AbstractTensorTrain{ValueType}) where {ValueType}
    print(io, "$(typeof(tt)) with rank $(rank(tt))")
end

"""
    function linkdims(tt::AbstractTensorTrain{V})::Vector{Int} where {V}

Bond dimensions along the links between ``T`` tensors in the tensor train.

See also: [`rank`](@ref)
"""
function linkdims(tt::AbstractTensorTrain{V})::Vector{Int} where {V}
    return [size(T, 1) for T in tt[2:end]]
end

"""
    function linkdim(tt::AbstractTensorTrain{V}, i::Int)::Int where {V}

Bond dimensions at the link between tensor ``T_i`` and ``T_{i+1}`` in the tensor train.
"""
function linkdim(tt::AbstractTensorTrain{V}, i::Int)::Int where {V}
    return size(sitetensor(tt, i+1), 1)
end

"""
    function sitedims(tt::AbstractTensorTrain{V})::Vector{Vector{Int}} where {V}

Dimensions of the site indices (local indices, physical indices) of the tensor train.
"""
function sitedims(tt::AbstractTensorTrain{V})::Vector{Vector{Int}} where {V}
    return [collect(size(T)[2:end-1]) for T in tt]
end

"""
    function linkdim(tt::AbstractTensorTrain{V}, i::Int)::Int where {V}

Dimension of the `i`th site index along the tensor train.
"""
function sitedim(tt::AbstractTensorTrain{V}, i::Int)::Vector{Int} where {V}
    return collect(size(sitetensor(tt, i))[2:end-1])
end

"""
    function rank(tt::AbstractTensorTrain{V}) where {V}

Rank of the tensor train, i.e. the maximum link dimension.

See also: [`linkdims`](@ref), [`linkdim`](@ref)
"""
function rank(tt::AbstractTensorTrain{V}) where {V}
    return maximum(linkdims(tt))
end

"""
    sitetensors(tt::AbstractTensorTrain{V}) where {V}

The tensors that make up the tensor train.
"""
function sitetensors(tt::AbstractTensorTrain{V}) where {V}
    return tt.sitetensors
end

"""
    sitetensors(tt::AbstractTensorTrain{V}, i) where {V}

The tensor at site i of the tensor train.
"""
function sitetensor(tt::AbstractTensorTrain{V}, i) where {V}
    return tt.sitetensors[i]
end

"""
    function length(tt::AbstractTensorTrain{V}) where {V}

Length of the tensor train, i.e. the number of tensors in the tensor train.
"""
function length(tt::AbstractTensorTrain{V}) where {V}
    return length(sitetensors(tt))
end

function Base.iterate(tt::AbstractTensorTrain{V}) where {V}
    return iterate(sitetensors(tt))
end

function Base.iterate(tt::AbstractTensorTrain{V}, state) where {V}
    return iterate(sitetensors(tt), state)
end

function Base.getindex(tt::AbstractTensorTrain{V}, i) where {V}
    return sitetensor(tt, i)
end

function Base.lastindex(tt::AbstractTensorTrain{V}) where {V}
    return lastindex(sitetensors(tt))
end

"""
    function evaluate(
        tt::TensorTrain{V},
        indexset::Union{AbstractVector{LocalIndex}, NTuple{N, LocalIndex}}
    )::V where {N, V}

Evaluates the tensor train `tt` at indices given by `indexset`.
"""
function evaluate(
    tt::AbstractTensorTrain{V},
    indexset::Union{AbstractVector{LocalIndex},NTuple{N,LocalIndex}}
)::V where {N,V}
    if length(indexset) != length(tt)
        throw(ArgumentError("To evaluate a tt of length $(length(tt)), you have to provide $(length(tt)) indices, but there were $(length(indexset))."))
    end
    return only(prod(T[:, i, :] for (T, i) in zip(tt, indexset)))
end

function evaluate(
    tt::AbstractTensorTrain{V},
    indexset::Union{AbstractVector{LocalIndex},NTuple{N,LocalIndex}},
    jndexset::Union{AbstractVector{LocalIndex},NTuple{N,LocalIndex}}
)::V where {N,V}
    if length(indexset) != length(tt)
        throw(ArgumentError("To evaluate a tt of length $(length(tt)), you have to provide $(length(tt)) indices, but there were $(length(indexset))."))
    end
    return only(prod(T[:, i, j, :] for (T, i, j) in zip(tt, indexset, jndexset)))
end



# Evaluate for left canonical, TODO: create type LCTensorTrain
function evaluate(
    tt::AbstractTensorTrain{V},
    Rs::Vector{Matrix{V}},
    indexset::Union{AbstractVector{LocalIndex},NTuple{N,LocalIndex}},
    jndexset::Union{AbstractVector{LocalIndex},NTuple{N,LocalIndex}}
)::V where {N,V}
    if length(indexset) != length(tt) || length(jndexset) != length(tt) 
        throw(ArgumentError("To evaluate a tt of length $(length(tt)), you have to provide $(length(tt)) indices, but there were $(length(indexset)), $(length(jndexset))."))
    end
    return only(prod(T[:, i, j, :]*Rs[idx] for (idx, (T, i, j)) in enumerate(zip(tt, indexset, jndexset))))
end

"""
    function evaluate(tt::TensorTrain{V}, indexset::CartesianIndex) where {V}

Evaluates the tensor train `tt` at indices given by `indexset`.
"""
function evaluate(tt::AbstractTensorTrain{V}, indexset::CartesianIndex) where {V}
    return evaluate(tt, Tuple(indexset))
end

function evaluate(
    tt::AbstractTensorTrain{V},
    indexset::Union{
        AbstractVector{<:AbstractVector{Int}},
        AbstractVector{NTuple{N,Int}},
        NTuple{N,<:AbstractVector{Int}},
        NTuple{N,NTuple{M,Int}}}
) where {N,M,V}
    if length(indexset) != length(tt)
        throw(ArgumentError("To evaluate a tt of length $(length(tt)), you have to provide $(length(tt)) indices, but there were $(length(indexset))."))
    end
    for (n, (T, i)) in enumerate(zip(tt, indexset))
        length(size(T)) == length(i) + 2 || throw(ArgumentError("The index set $(i) at position $n does not have the correct length for the tensor of size $(size(T))."))
    end
    return only(prod(T[:, i..., :] for (T, i) in zip(tt, indexset)))
end

function (tt::AbstractTensorTrain{V})(indexset) where {V}
    return evaluate(tt, indexset)
end

"""
    function sum(tt::TensorTrain{V}) where {V}

Evaluates the sum of the tensor train approximation over all lattice sites in an efficient
factorized manner.
"""
function sum(tt::AbstractTensorTrain{V}) where {V}
    v = transpose(sum(tt[1], dims=(1, 2))[1, 1, :])
    for T in tt[2:end]
        v *= sum(T, dims=2)[:, 1, :]
    end
    return only(v)
end

"""
    function average(tt::TensorTrain{V}) where {V}

Evaluates the average of the tensor train approximation over all lattice sites in an efficient
factorized manner.
"""
function average(tt::AbstractTensorTrain{V}) where {V}
    v = transpose(sum(tt[1], dims=(1, 2))[1, 1, :]) / length(tt[1][1, :, 1])
    for T in tt[2:end]
        v *= sum(T, dims=2)[:, 1, :] / length(T[1, :, 1])
    end
    return only(v)
end

"""
    function weightedsum(tt::TensorTrain{V}, w::Vector{V}) where {V}

Evaluates the weighted sum of the tensor train approximation over all lattice sites in an efficient
factorized manner, where w is the vector of vector of weights which has the same length and the same sizes as tt.
"""
function weightedsum(tt::AbstractTensorTrain{V}, w::Vector{Vector{V}}) where {V}
    length(tt) == length(w) || throw(DimensionMismatch("The length of the Tensor Train is different from the one of the weight vector ($(length(tt)) and $(length(w)))."))
    size(tt[1])[2] == length(w[1]) || throw(DimensionMismatch("The dimension at site 1 of the Tensor Train is different from the one of the weight vector ($(size(tt[1])[2]) and $(length(w[1])))."))
    v = transpose(sum(tt[1].*w[1]', dims=(1, 2))[1, 1, :])
    for i in 2:length(tt)
        size(tt[i])[2] == length(w[i]) || throw(DimensionMismatch("The dimension at site $(i) of the Tensor Train is different from the one of the weight vector ($(size(tt[i])[2]) and $(length(w[i])))."))
        v *= sum(tt[i].*w[i]', dims=2)[:, 1, :]
    end
    return only(v)
end


function _addtttensor(
    A::Array{V}, B::Array{V};
    factorA=one(V), factorB=one(V),
    lefttensor=false, righttensor=false
) where {V}
    if ndims(A) != ndims(B)
        throw(DimensionMismatch("Elementwise addition only works if both tensors have the same indices, but A and B have different numbers ($(ndims(A)) and $(ndims(B))) of indices."))
    end
    nd = ndims(A)
    offset1 = lefttensor ? 0 : size(A, 1)
    offset3 = righttensor ? 0 : size(A, nd)
    localindices = fill(Colon(), nd - 2)
    C = zeros(V, offset1 + size(B, 1), size(A)[2:nd-1]..., offset3 + size(B, nd))
    C[1:size(A, 1), localindices..., 1:size(A, nd)] = factorA * A
    C[offset1+1:end, localindices..., offset3+1:end] = factorB * B
    return C
end

@doc raw"""
    function add(
        lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V};
        factorlhs=one(V), factorrhs=one(V),
        tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
    ) where {V}

Addition of two tensor trains. If `C = add(A, B)`, then `C(v) ≈ A(v) + B(v)` at each index set `v`. Note that this function increases the bond dimension, i.e. ``\chi_{\text{result}} = \chi_1 + \chi_2`` if the original tensor trains had bond dimensions ``\chi_1`` and ``\chi_2``.

Arguments:
- `lhs`, `rhs`: Tensor trains to be added.
- `factorlhs`, `factorrhs`: Factors to multiply each tensor train by before addition.
- `tolerance`, `maxbonddim`: Parameters to be used for the recompression step.

Returns:
A new `TensorTrain` representing the function `factorlhs * lhs(v) + factorrhs * rhs(v)`.

See also: [`+`](@ref)
"""
function add(
    lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V};
    factorlhs=one(V), factorrhs=one(V),
    tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
) where {V}
    if length(lhs) != length(rhs)
        throw(DimensionMismatch("Two tensor trains with different length ($(length(lhs)) and $(length(rhs))) cannot be added elementwise."))
    end
    L = length(lhs)
    tt = tensortrain(
        [
            _addtttensor(
                lhs[ell], rhs[ell];
                factorA=((ell == L) ? factorlhs : one(V)),
                factorB=((ell == L) ? factorrhs : one(V)),
                lefttensor=(ell==1),
                righttensor=(ell==L)
            )
            for ell in 1:L
        ]
    )
    compress!(tt, :SVD; tolerance, maxbonddim)
    return tt
end

@doc raw"""
    function subtract(
        lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V};
        tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
    )

Subtract two tensor trains `lhs` and `rhs`. See [`add`](@ref).
"""
function subtract(
    lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V};
    tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
) where {V}
    return add(lhs, rhs; factorrhs=-1 * one(V), tolerance, maxbonddim)
end

@doc raw"""
    function (+)(lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V}) where {V}

Addition of two tensor trains. If `c = a + b`, then `c(v) ≈ a(v) + b(v)` at each index set `v`. Note that this function increases the bond dimension, i.e. ``\chi_{\text{result}} = \chi_1 + \chi_2`` if the original tensor trains had bond dimensions ``\chi_1`` and ``\chi_2``. Can be combined with automatic recompression by calling [`add`](@ref).
"""
function Base.:+(lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V}) where {V}
    return add(lhs, rhs)
end

@doc raw"""
    function (-)(lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V}) where {V}

Subtraction of two tensor trains. If `c = a - b`, then `c(v) ≈ a(v) - b(v)` at each index set `v`. Note that this function increases the bond dimension, i.e. ``\chi_{\text{result}} = \chi_1 + \chi_2`` if the original tensor trains had bond dimensions ``\chi_1`` and ``\chi_2``. Can be combined with automatic recompression by calling [`subtract`](@ref) (see documentation for [`add`](@ref)).
"""
function Base.:-(lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V}) where {V}
    return subtract(lhs, rhs)
end

function leftcanonicalize!(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    MPO = tt.sitetensors
    N = length(MPO)  # Number of sites
    for i in 1:N-1
        Q, R = qr(reshape(MPO[i], prod(size(MPO[i])[1:end-1]), size(MPO[i])[end]))
        Q = Matrix(Q)

        MPO[i] = reshape(Q, size(MPO[i])[1:end-1]..., size(Q, 2))  # New bond dimension after Q

        tmpMPO = reshape(MPO[i+1], size(R, 2), :)  # Reshape next tensor
        tmpMPO .= Matrix(R) * tmpMPO
        MPO[i+1] = reshape(tmpMPO, size(MPO[i+1])...)  # Reshape back
    end
end

# This creates a TensorTrain which has every site right-canonical except the last
function rightcanonicalize!(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    MPO = tt.sitetensors
    N = length(MPO)  # Number of sites
    for i in N:-1:2
        # Reshape W_i into a matrix (merging right bond and physical indices)
        W = MPO[i]
        χl, d1, d2, χr = size(W)
        W_mat = reshape(W, χl, d1*d2*χr)

        # Perform RQ decomposition: W_mat = R * Q
        F = lq(reverse(W_mat, dims=1))
        R, Q = reverse(F.L), reverse(Matrix(F.Q), dims=1) # https://discourse.julialang.org/t/rq-decomposition/112795/13

        # Reshape Q back into the MPO tensor
        MPO[i] = reshape(Q, size(Q, 1), d1, d2, χr)  # New bond dimension after Q

        # Update the previous MPO tensor by absorbing R
        tmpMPO = reshape(MPO[i-1], :, size(R, 1))  # Reshape previous tensor
        tmpMPO .= tmpMPO * Matrix(R)
        MPO[i-1] = reshape(tmpMPO, size(MPO[i-1], 1), d1, d2, size(MPO[i-1], 4))  # Reshape back
    end
end


function centercanonicalize!(tt::AbstractTensorTrain{ValueType}, center::Int) where {ValueType}
    MPO = tt.sitetensors
    N = length(MPO)  # Number of sites
    # LEFT
    for i in 1:center-1
        Q, R = qr(reshape(MPO[i], prod(size(MPO[i])[1:end-1]), size(MPO[i])[end]))
        Q = Matrix(Q)

        MPO[i] = reshape(Q, size(MPO[i])[1:end-1]..., size(Q, 2))  # New bond dimension after Q

        tmpMPO = reshape(MPO[i+1], size(R, 2), :)  # Reshape next tensor
        tmpMPO = Matrix(R) * tmpMPO
        MPO[i+1] = reshape(tmpMPO, size(MPO[i+1])...)  # Reshape back
    end
    # RIGHT
    for i in N:-1:center+1
        W = MPO[i]
        χl, d1, d2, χr = size(W)
        W_mat = reshape(W, χl, d1*d2*χr)

        #println(size(W_mat))
        L, Q = lq(W_mat)
        #println(size(Matrix(L)), size(Matrix(Q)))
        Q = Matrix(Q)
        # Reshape Q back into the MPO tensor
        MPO[i] = reshape(Q, size(Q, 1), d1, d2, χr)  # New bond dimension after Q

        # Update the previous MPO tensor by absorbing L
        tmpMPO = reshape(MPO[i-1], :, size(L, 1))  # Reshape previous tensor
        tmpMPO = tmpMPO * Matrix(L)
        MPO[i-1] = reshape(tmpMPO, size(MPO[i-1], 1), d1, d2, size(tmpMPO, 2))  # Reshape back
    end
end

function move_center_right!(tt, i)
    MPO = tt.sitetensors
    A = MPO[i]
    d = size(A)
    A_mat = reshape(A, prod(d[1:end-1]), d[end])
    Q, R = qr(A_mat)
    Q = Matrix(Q)
    MPO[i] = reshape(Q, d[1:end-1]..., size(Q, 2))

    B = MPO[i+1]
    B_mat = reshape(B, size(R, 2), :)
    B_mat .= Matrix(R) * B_mat
    MPO[i+1] = reshape(B_mat, size(B)...)
end

function move_center_left!(tt, i)
    MPO = tt.sitetensors
    A = MPO[i]
    d = size(A)
    A_mat = reshape(A, d[1], prod(d[2:end]))
    L, Q = lq(A_mat)
    Q = Matrix(Q)
    MPO[i] = reshape(Q, size(Q,1), d[2:end]...)

    B = MPO[i-1]
    B_mat = reshape(B, :, size(L,1))
    B_mat .= B_mat * Matrix(L)
    MPO[i-1] = reshape(B_mat, size(B)[1:3]..., size(L,1))
end


function leftcanonicalize(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    tt_ = deepcopy(tt)
    leftcanonicalize!(tt_)
    return tt_
end

# This creates a TensorTrain which has every site right-canonical except the last
function rightcanonicalize(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    tt_ = deepcopy(tt)
    rightcanonicalize!(tt_)
    return tt_
end

# This creates a TensorTrain which has every site right-canonical except the last
function centercanonicalize(tt::AbstractTensorTrain{ValueType}, center::Int) where {ValueType}
    tt_ = deepcopy(tt)
    centercanonicalize!(tt_, center)
    return tt_
end

function checkorthogonality(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    ort = Vector{Symbol}(undef, length(tt))
    for i in 1:length(tt)
        W = tt.sitetensors[i]
        left_check = _contract(permutedims(W, (4,2,3,1,)), W, (2,3,4,),(2,3,1))
        right_check = _contract(W, permutedims(W, (4,2,3,1,)), (2,3,4,),(2,3,1))
        is_left = isapprox(left_check, I, atol=1e-10)
        is_right = isapprox(right_check, I, atol=1e-10)
        ort[i] = if is_left && is_right
            :O # Orthogonal
        elseif is_left
            :L # Left orthogonal
        elseif is_right
            :R # Right orthogonal
        else
            :N # Non orthogonal
        end
    end
    return ort
end

@doc raw"""
    function makesitediagonal(
        tt::AbstractTensorTrain{V}, theory_dims::Vector{Int}, theory_index::Int; local_index::Int
    )

The function will reshape the tensors in to make it diagonal with respect to the specified index.
Warning: This function is not yet tested and may not work as expected.

Arguments:
- `tt`: Tensor train.
- `theory_dims`: Theoretical division of the local index.
- `theory_index`: Which theoretical index with respect to make the tensor diagonal.
- `local_index`: Which local index index with respect to make the tensor diagonal.

"""
# Warning: This function is not yet tested and may not work as expected.
function makesitediagonal!(tt::AbstractTensorTrain{V}, theory_dims::Vector{Int}, theory_index::Int; local_index::Int=1) where {V}
    prod(theory_dims) != prod(size(sitetensor(tt, 1))[2:(end-1)]) && error("Local dims dimension mismatch") # Assume d equal in all sites

    for b in 1:length(tt)
        size_separated = (size(sitetensor(tt, b))[1], theory_dims..., size(sitetensor(tt, b))[end])
        lenA = length(size_separated)
        A = reshape(sitetensor(tt, b), size_separated)
        perm = vcat(1:theory_index-1, theory_index+1:lenA, theory_index)
        A = permutedims(A, perm) # index to duplicate at the end
        size_separated_perm = size_separated[perm]
        new_size_separated_perm = (size_separated_perm..., theory_dims[theory_index])
        B = zeros(eltype(A), new_size_separated_perm...)
        for i in 1:theory_dims[theory_index]
            B[.., i, i] = A[.., i]
        end
        B = permutedims(B, vcat(1:theory_index-1, lenA, lenA+1, theory_index:lenA-1))
        B = reshape(B, (size(sitetensor(tt, b))[1:local_index]..., theory_dims[theory_index]*size(sitetensor(tt, b))[local_index+1] ,size(sitetensor(tt, b))[local_index+2:end]...))
        tt.sitetensors[b] = B
    end
    return
end

@doc raw"""
    function extractdiagonal!(
        tt::AbstractTensorTrain{V}, theory_dims::Vector{Int}, theory_indexes::Union{Vector{Int},Tuple{Int}}, local_index::Int
    )

The function will reshape the tensors to extract the diagonal with respect to the specified indices.
Warning: This function is not yet tested and may not work as expected.

Arguments:
- `tt`: Tensor train.
- `theory_dims`: Theoretical division of the local index.
- `theory_index`: Which theoretical index with respect to make the tensor diagonal.
- `local_index`: Which local index index with respect to make the tensor diagonal.

"""
# Warning: This function is not yet tested and may not work as expected.
function extractdiagonal!(tt::AbstractTensorTrain, theory_dims::Vector{Int}, theory_indexes::Union{Vector{Int},Tuple{Int}}; local_index::Int=1)
    prod(theory_dims) != prod(size(sitetensor(tt, 1))[2:(end-1)]) && error("Local dims dimension mismatch") # Assume d equal in all sites

    theory_indexes = sort(collect(theory_indexes)) # Vector
    for b in 1:length(tt)
        size_separated = (size(sitetensor(tt, b))[1], theory_dims..., size(sitetensor(tt, b))[end])
        lenA = length(size_separated)
        A = reshape(sitetensor(tt, b), size_separated)
        perm = vcat(1:theory_indexes[1]-1, theory_indexes[1]+1:theory_indexes[2]-1, theory_indexes[2]+1:lenA, theory_indexes[1], theory_indexes[2])
        A = permutedims(A, perm) # indexes to merge at the end
        size_separated_perm = size_separated[perm]
        new_size_separated_perm = tuple(size_separated_perm[1:end-1]...)
        B = zeros(eltype(A), new_size_separated_perm...)
        for i in 1:theory_dims[theory_indexes[1]]
            B[.., i] = A[.., i, i]
        end
        B = permutedims(B, vcat(1:theory_indexes[1]-1, lenA-1, theory_indexes[1]:lenA-2))
        B = reshape(B, (size(sitetensor(tt, b))[1:local_index]..., Int(size(sitetensor(tt, b))[local_index+1]/theory_dims[theory_indexes[1]]) ,size(sitetensor(tt, b))[local_index+2:end]...))
        tt.sitetensors[b] = B
    end

    return
end

@doc raw"""
    function mpo2mps(
        tt::AbstractTensorTrain{V}
    )

Converts a Matrix Product Operator (MPO) to a Matrix Product State (MPS) by unifying the physical indices.

Warning: This function is not yet tested and may not work as expected.

Arguments:
- `tt`: Tensor train.

"""
# Warning: This function is not yet tested and may not work as expected.
function mpo2mps(tt::AbstractTensorTrain{V}) where V
    tensors = Vector{Array{V, 3}}(undef, length(tt))
    for b in 1:length(tt)
        site = reshape(tt.sitetensors[b], (size(tt.sitetensors[b])[1], prod(size(tt.sitetensors[b])[2:end-1]), size(tt.sitetensors[b])[end]))
        tensors[b] = site
    end
    return TensorTrain{V, 3}(tensors)
end

@doc raw"""
    function mposwapindex(
        tt::AbstractTensorTrain{V}, theory_dims::Vector{Int}, permute::Vector{Int}, output_dims::Vector{Int}
    )

Swaps the physical indices of a tensor train.
    
Warning: This function is not yet tested and may not work as expected.

Arguments:
- `tt`: Tensor train.
- `theory_dims`: Theoretical division of the local index.
- `permute`: Permutation of the indices.
- `output_dims`: The dimensions of the output tensor train indices.
"""
# For now it assumes N=4
function mposwapindex(tt::AbstractTensorTrain{V}, theory_dims::Vector{Int}, permute::Vector{Int}, output_dims::Vector{Int}) where V
    tensors = Vector{Array{V, 4}}(undef, length(tt))
    for b in 1:length(tt)
        site = reshape(tt.sitetensors[b], (size(tt.sitetensors[b])[1], theory_dims..., size(tt.sitetensors[b])[end]))
        site = permutedims(site, (1, (permute.+1)... ,length(permute)+2,))
        sitedim = reshape(site, (size(tt.sitetensors[b])[1], output_dims..., size(tt.sitetensors[b])[end]))
        tensors[b] = sitedim
    end
    wow = TensorTrain{V, 4}(tensors)
end
"""
Squared Frobenius norm of a tensor train.
"""
function LA.norm2(tt::AbstractTensorTrain{V})::Float64 where {V}
    function _f(n)::Matrix{V}
        t = sitetensor(tt, n)
        t3 = reshape(t, size(t)[1], :, size(t)[end])
        # (lc, s, rc) * (l, s, r) => (lc, rc, l, r)
        tct = _contract(conj.(t3), t3, (2,), (2,))
        tct = permutedims(tct, (1, 3, 2, 4))
        return reshape(tct, size(tct, 1) * size(tct, 2), size(tct, 3) * size(tct, 4))
    end
    return real(only(reduce(*, (_f(n) for n in 1:length(tt)))))
end

"""
Frobenius norm of a tensor train.
"""
function LA.norm(tt::AbstractTensorTrain{V})::Float64 where {V}
    sqrt(LA.norm2(tt))
end