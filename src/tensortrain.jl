"""
    struct TensorTrain{ValueType}

Represents a tensor train, also known as MPS.

The tensor train can be evaluated using standard function call notation:
```julia
    tt = TensorTrain(...)
    value = tt([1, 2, 3, 4])
```
The corresponding function is:

    function (tt::TensorTrain{V})(indexset) where {V}

Evaluates the tensor train `tt` at indices given by `indexset`.
"""
mutable struct TensorTrain{ValueType,N} <: AbstractTensorTrain{ValueType}
    T::Vector{Array{ValueType,N}}

    function TensorTrain{ValueType,N}(T::Vector{Array{ValueType,N}}) where {ValueType,N}
        for i in 1:length(T)-1
            if (last(size(T[i])) != size(T[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        new{ValueType,N}(T)
    end
end

"""
    function TensorTrain(T::Vector{Array{V, 3}}) where {V}

Create a tensor train out of a vector of tensors. Each tensor should have links to the
previous and next tensor as dimension 1 and 3, respectively; the local index ("physical
index" for MPS in physics) is dimension 2.
"""
function TensorTrain(T::Vector{Array{V,N}}) where {V,N}
    return TensorTrain{V,N}(T)
end

"""
    function TensorTrain(tci::TensorCI{V}) where {V}

Convert the TCI object into a tensor train, also known as an MPS.

See also: [`crossinterpolate`](@ref), [`TensorCI`](@ref)
"""
function TensorTrain(tci::TensorCI{V})::TensorTrain{V,3} where {V}
    return TensorTrain{V,3}(TtimesPinv.(tci, 1:length(tci)))
end

"""
    function TensorTrain(tci::TensorCI2{V}) where {V}

Convert the TCI2 object into a tensor train, also known as an MPS.

See also: [`crossinterpolate2`](@ref), [`TensorCI2`](@ref)
"""
function TensorTrain(tci::TensorCI2{V})::TensorTrain{V,3} where {V}
    return TensorTrain(tci.T)
end

function tensortrain(tci)
    return TensorTrain(tci)
end


"""
Fitting data with a TensorTrain object.
This may be useful when the interpolated function is noisy.
"""
struct TensorTrainFit{ValueType}
    indexsets::Vector{MultiIndex}
    values::Vector{ValueType}
    tt::TensorTrain{ValueType,3}
    offsets::Vector{Int}
end

function TensorTrainFit{ValueType}(indexsets, values, tt) where {ValueType}
    offsets = [0]
    for n in 1:length(tt.T)
        push!(offsets, offsets[end] + length(tt.T[n]))
    end
    return TensorTrainFit{ValueType}(indexsets, values, tt, offsets)
end

flatten(obj::TensorTrain{ValueType}) where {ValueType} = vcat(vec.(obj.T)...)

function to_tensors(obj::TensorTrainFit{ValueType}, x::Vector{ValueType}) where {ValueType}
    return [
        reshape(
            x[obj.offsets[n]+1:obj.offsets[n+1]],
            size(obj.tt.T[n])
            )
        for n in 1:length(obj.tt)
    ]
end

_evaluate(tt, indexset) = only(prod(T[:, i, :] for (T, i) in zip(tt, indexset)))

function (obj::TensorTrainFit{ValueType})(x::Vector{ValueType}) where {ValueType}
    tensors = to_tensors(obj, x)
    return sum((abs2(_evaluate(tensors, indexset)  - obj.values[i]) for (i, indexset) in enumerate(obj.indexsets)))
end