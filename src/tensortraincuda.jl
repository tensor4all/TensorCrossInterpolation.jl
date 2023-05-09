"""
    struct TensorTrainCUDA{ValueType}

Represents a tensor train, also known as MPS.

The tensor train can be evaluated using standard function call notation:
```julia
    tt = TensorTrainCUDA(...)
    value = tt([1, 2, 3, 4])
```
The corresponding function is:

    function (tt::TensorTrainCUDA{V})(indexset) where {V}

Evaluates the tensor train `tt` at indices given by `indexset`.
"""
struct TensorTrainCUDA{ValueType} <: AbstractTensorTrain{ValueType}
    T::Vector{CuArray{ValueType, 3}}
end

function TensorTrainCUDA(T::AbstractVector{CuArray{ValueType, 3}}) where {ValueType}
    for i in 1:length(T)-1
        if (size(T[i], 3) != size(T[i+1], 1))
            throw(ArgumentError(
                "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
            ))
        end
    end

    new{ValueType}(T)
end

"""
    function TensorTrainCUDA(T::Vector{Array{V, 3}}) where {V}

Create a tensor train out of a vector of tensors. Each tensor should have links to the
previous and next tensor as dimension 1 and 3, respectively; the local index ("physical
index" for MPS in physics) is dimension 2.
"""
function TensorTrainCUDA(T::AbstractVector{Array{V, 3}}) where {V}
    return TensorTrainCUDA{V}(CuArray.(T))
end

"""
    function TensorTrainCUDA(tci::TensorCI{V}) where {V}

Convert the TCI object into a tensor train, also known as an MPS.

See also: [`crossinterpolate`](@ref), [`TensorCI`](@ref)
"""
function TensorTrainCUDA(tci::TensorCI{V})::TensorTrainCUDA{V} where {V}
    return TensorTrainCUDA{V}(TtimesPinv.(tci, 1:length(tci)))
end

"""
    function TensorTrainCUDA(tci::TensorCI2{V}) where {V}

Convert the TCI2 object into a tensor train, also known as an MPS.

See also: [`crossinterpolate`](@ref), [`TensorCI`](@ref)
"""
function TensorTrainCUDA(tci::TensorCI2{V})::TensorTrainCUDA{V} where {V}
    return TensorTrainCUDA{V}(tci.T)
end
