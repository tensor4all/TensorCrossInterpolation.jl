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

"""
    function initializempi()

Initialize the MPI environment if it has not been initialized yet.
"""
function initializempi(mute::Bool=true)
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    if mute
        if rank != 0
            open("/dev/null", "w") do devnull
                redirect_stdout(devnull)
                redirect_stderr(devnull)
            end
        end
    end
end

"""
    function finalizempi()

Finalize the MPI environment if it has not been finalized yet.
"""
function finalizempi()
    if !MPI.Finalized()
        MPI.Finalize()
    end
end
