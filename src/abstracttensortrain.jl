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

function left_canonicalize!(tt::AbstractTensorTrain{ValueType}) where {ValueType}
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
function right_canonicalize!(tt::AbstractTensorTrain{ValueType}) where {ValueType}
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
        #Q, R = qr(W_mat')
        #Q = Matrix(Q)

        # Reshape Q back into the MPO tensor
        MPO[i] = reshape(Q, size(Q, 1), d1, d2, χr)  # New bond dimension after Q

        # Update the previous MPO tensor by absorbing R
        tmpMPO = reshape(MPO[i-1], :, size(R, 1))  # Reshape previous tensor
        tmpMPO .= tmpMPO * Diagonal(R)
        MPO[i-1] = reshape(tmpMPO, size(MPO[i-1], 1), d1, d2, size(MPO[i-1], 4))  # Reshape back
    end
end


function center_canonicalize!(tt::AbstractTensorTrain{ValueType}, center::Int) where {ValueType}
    MPO = tt.sitetensors
    N = length(MPO)  # Number of sites
    # LEFT
    for i in 1:center-1
        Q, R = qr(reshape(MPO[i], prod(size(MPO[i])[1:end-1]), size(MPO[i])[end]))
        Q = Matrix(Q)

        MPO[i] = reshape(Q, size(MPO[i])[1:end-1]..., size(Q, 2))  # New bond dimension after Q

        tmpMPO = reshape(MPO[i+1], size(R, 2), :)  # Reshape next tensor
        tmpMPO .= Matrix(R) * tmpMPO
        MPO[i+1] = reshape(tmpMPO, size(MPO[i+1])...)  # Reshape back
    end
    # RIGHT
    for i in N:-1:center+1
        W = MPO[i]
        χl, d1, d2, χr = size(W)
        W_mat = reshape(W, χl, d1*d2*χr)
        F = lq(reverse(W_mat, dims=1))
        R, Q = reverse(F.L), reverse(Matrix(F.Q), dims=1) # https://discourse.julialang.org/t/rq-decomposition/112795/13

        # Reshape Q back into the MPO tensor
        MPO[i] = reshape(Q, size(Q, 1), d1, d2, χr)  # New bond dimension after Q

        # Update the previous MPO tensor by absorbing R
        tmpMPO = reshape(MPO[i-1], :, size(R, 1))  # Reshape previous tensor
        tmpMPO .= tmpMPO * Diagonal(R)
        MPO[i-1] = reshape(tmpMPO, size(MPO[i-1], 1), d1, d2, size(MPO[i-1], 4))  # Reshape back
    end
end


function left_canonicalize(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    tt_ = deepcopy(tt)
    left_canonicalize!(tt_)
    return tt_
end

# This creates a TensorTrain which has every site right-canonical except the last
function right_canonicalize(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    tt_ = deepcopy(tt)
    right_canonicalize!(tt_)
    return tt_
end

# This creates a TensorTrain which has every site right-canonical except the last
function center_canonicalize(tt::AbstractTensorTrain{ValueType}, center::Int) where {ValueType}
    tt_ = deepcopy(tt)
    center_canonicalize!(tt_, center)
    return tt_
end

function extract_sv!(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    MPO = tt.sitetensors
    N = length(MPO)  # Number of sites
    sv = Vector{Vector}(undef, N-1)
    for i in 2:N
        # Extract the tensor at the specified index
        W = MPO[i]
        χl, d1, d2, χr = size(W)
        
        # Reshape the MPO tensor into a matrix for SVD
        W_mat = reshape(W, χl , d1 * d2 * χr)
        
        # Perform SVD to decompose the tensor into U, S, and V
        U, S, V = svd(W_mat)

        sv[i-1] = S
        
        MPO[i] = reshape(U*V', χl, d1, d2, χr)
    end
    return sv
end

function mpo_to_vidal!(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    N = length(tt)
    sv = Vector{Vector{ValueType}}(undef, N-1)
    left_canonicalize!(tt)
    if !check_left_orthogonality(tt)[1]
        println("It's not left_orthogonal")
        return
    end
    for i in N:-1:2
        U, S, V = LinearAlgebra.svd(reshape(tt.sitetensors[i], size(tt.sitetensors[i])[1], prod(size(tt.sitetensors[i])[2:end])))
        sv[i-1] = S
        tt.sitetensors[i] = reshape(V', size(tt.sitetensors[i])...)
        tt.sitetensors[i-1] = _contract(tt.sitetensors[i-1], U*Diagonal(S), (4,), (1,))
    end
    for i in N-1:-1:1
        tt.sitetensors[i] = _contract(tt.sitetensors[i], Diagonal(sv[i].^-1), (4,), (1,))
    end
    return sv
end

function check_orthogonality(tt::AbstractTensorTrain{ValueType}) where {ValueType}
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

# Function to check if a tensor is left-orthogonal (left canonical form)
function check_left_orthogonality(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    for i in 1:length(tt)-1
        W = tt.sitetensors[i]
        left_check = _contract(permutedims(W, (4,2,3,1,)), W, (2,3,4,),(2,3,1))
        if !isapprox(left_check, I, atol=1e-10)
            return false, i  # Fail at tensor i
        end
    end
    return true, nothing  # All tensors passed
end

# Function to check if a tensor is left-orthogonal (left canonical form)
function check_left_orthogonality(tt::AbstractTensorTrain{ValueType}, Ls::Vector{Vector{ValueType}}) where {ValueType}
    for i in 1:length(tt)-1
        if i > 1
            LG = _contract(Diagonal(Ls[i-1]), tt.sitetensors[i], (2,), (1,))
        else
            LG = tt.sitetensors[i]
        end
        left_check = _contract(permutedims(LG, (4,2,3,1,)), LG, (2,3,4,),(2,3,1))
        if !isapprox(left_check, I, atol=1e-10)
            return false, i  # Fail at tensor i
        end
    end
    return true, nothing  # All tensors passed
end

# Function to check if a tensor is right-orthogonal (right canonical form)
function check_right_orthogonality(tt::AbstractTensorTrain)
    for i in length(tt):-1:2
        W = tt.sitetensors[i]
        right_check = _contract(W, permutedims(W, (4,2,3,1,)), (2,3,4,),(2,3,1))
        if !isapprox(right_check, I, atol=1e-10)
            return false, i  # Fail at tensor i
        end
    end
    return true, nothing  # All tensors passed
end

# Function to check if a tensor is right-orthogonal (right canonical form)
function check_right_orthogonality(tt::AbstractTensorTrain{V}, Ls::Vector{Vector{V}}) where {V}
    for i in length(tt):-1:2
        if i < length(tt)
            LG = _contract(tt.sitetensors[i], Diagonal(Ls[i]), (4,), (1,))
        else
            LG = tt.sitetensors[i]
        end
        right_check = _contract(LG, permutedims(LG, (4,2,3,1,)), (2,3,4,),(2,3,1))
        if !isapprox(right_check, I, atol=1e-10)
            return false, i  # Fail at tensor i
        end
    end
    return true, nothing  # All tensors passed
end

function check_vidal(tt::AbstractTensorTrain{ValueType}, Ls::Vector{Vector{ValueType}}) where {ValueType}
    check_l, il = check_left_orthogonality(tt, Ls)
    if !check_l
        println("Not left orthogonal at site $il")
    end
    check_r, ir = check_right_orthogonality(tt, Ls)
    if !check_r
        println("Not right orthogonal at site $ir")
    end
    return check_l && check_r
end

# This creates a LCTensorTrain which has every site left-canonical and the shur components are saved in a Vector of Matrices
function left_canonical_form!(tt::AbstractTensorTrain{V}) where {V}
    Rs = Vector{Matrix{V}}(undef, length(tt)-1)
    for ell in 1:(length(tt) - 1)
        shapel = size(tt.sitetensors[ell])

        Q, R = qr(reshape(tt.sitetensors[ell], prod(shapel[1:end-1]), shapel[end]))  # QR decomposition
        Q = Matrix(Q)
        R = Matrix(R)
        Rs[ell] = R
        # Update sitetensors[i] with reshaped Q
        tt.sitetensors[ell] = reshape(Q, shapel[1:end-1]..., size(Q, 2))
    end
    return Rs
end

function left_canonical_form(tt::AbstractTensorTrain{V}) where {V}
    tt_ = deepcopy(tt)
    Rs = Vector{Matrix{V}}(undef, length(tt_))
    for ell in 1:(length(tt_) - 1)
        shapel = size(tt_.sitetensors[ell])

        Q, R = qr(reshape(tt_.sitetensors[ell], prod(shapel[1:end-1]), shapel[end]))  # QR decomposition
        Q = Matrix(Q)
        R = Matrix(R)
        Rs[ell] = R
        # Update sitetensors[i] with reshaped Q
        tt_.sitetensors[ell] = reshape(Q, shapel[1:end-1]..., size(Q, 2))
    end
    Rs[end] = reshape([1.0], 1, 1)
    return tt_, Rs
end

@doc raw"""
    function makesitediagonal(
        tt::AbstractTensorTrain{V}, theory_dims::Vector{Int},
    )

Subtract two tensor trains `lhs` and `rhs`. See [`add`](@ref).

Arguments:
- `tt`: Tensor train.
- `theory_dims`: Theoretical division of the local index.
- `theory_index`: Which theoretical index with respect to make the tensor diagonal.
- `local_index`: Which local index index with respect to make the tensor diagonal.

"""
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

function mpo2mps(tt::AbstractTensorTrain{V}) where V
    tensors = Vector{Array{V, 3}}(undef, length(tt))
    for b in 1:length(tt)
        site = reshape(tt.sitetensors[b], (size(tt.sitetensors[b])[1], prod(size(tt.sitetensors[b])[2:end-1]), size(tt.sitetensors[b])[end]))
        tensors[b] = site
    end
    return TensorTrain{V, 3}(tensors)
end

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

function trace(mpoa::AbstractTensorTrain{V}, mpob::AbstractTensorTrain{V}) where {V}
    T = ones(Float64, 1, 1, 1)
    for b in 1:length(mpoa)
        T = _contract(T, mpoa[b], (2,), (1,))
        T = _contract(T, mpob[b], (2,3,4,), (1,3,2,))
    end
    return sum(T)
end

function frobenius_norm_diff(mpoa::AbstractTensorTrain{V}, mpob::AbstractTensorTrain{V}) where {V}
    return trace(mpoa, mpoa) + trace(mpob, mpob) - trace(mpoa, mpob) - trace(mpob, mpoa)
end