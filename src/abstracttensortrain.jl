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

Implementations: [`TensorTrain`](@ref), [`TensorCI2`](@ref), [`TensorCI`](@ref)
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
    return size(tt.T[i+1], 1)
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
    return collect(size(tt.T[i])[2:end-1])
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
    function length(tt::AbstractTensorTrain{V}) where {V}

Length of the tensor train, i.e. the number of tensors in the tensor train.
"""
function length(tt::AbstractTensorTrain{V}) where {V}
    return length(tt.T)
end

function Base.iterate(tt::AbstractTensorTrain{V}) where {V}
    return iterate(tt.T)
end

function Base.iterate(tt::AbstractTensorTrain{V}, state) where {V}
    return iterate(tt.T, state)
end

function Base.getindex(tt::AbstractTensorTrain{V}, i) where {V}
    return getindex(tt.T, i)
end

function Base.lastindex(tt::AbstractTensorTrain{V}) where {V}
    return lastindex(tt.T)
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
    v = sum(tt[1], dims=(1, 2))[1, 1, :]'
    for T in tt[2:end]
        v *= sum(T, dims=2)[:, 1, :]
    end
    return only(v)
end
