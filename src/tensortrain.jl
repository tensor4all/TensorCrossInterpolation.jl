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
struct TensorTrain{ValueType} <: AbstractTensorTrain{ValueType}
    T::Vector{Array{ValueType, 3}}

    function TensorTrain{ValueType}(T::Vector{Array{ValueType, 3}}) where {ValueType}
        for i in 1:length(T)-1
            if (size(T[i], 3) != size(T[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        new{ValueType}(T)
    end
end

"""
    function TensorTrain(T::Vector{Array{V, 3}}) where {V}

Create a tensor train out of a vector of tensors. Each tensor should have links to the
previous and next tensor as dimension 1 and 3, respectively; the local index ("physical
index" for MPS in physics) is dimension 2.
"""
function TensorTrain(T::Vector{Array{V, 3}}) where {V}
    return TensorTrain{V}(T)
end

"""
    function TensorTrain(tci::TensorCI{V}) where {V}

Convert the TCI object into a tensor train, also known as an MPS.

See also: [`crossinterpolate`](@ref), [`TensorCI`](@ref)
"""
function TensorTrain(tci::TensorCI{V})::TensorTrain{V} where {V}
    return TensorTrain{V}(Array{V,3}.(TtimesPinv.(tci, 1:length(tci))))
end

"""
    function TensorTrain(tci::TensorCI2{V}) where {V}

Convert the TCI2 object into a tensor train, also known as an MPS.

See also: [`crossinterpolate`](@ref), [`TensorCI`](@ref)
"""
function TensorTrain(tci::TensorCI2{V})::TensorTrain{V} where {V}
    return TensorTrain(tci.T)
end

function tensortrain(tci)
    return TensorTrain(tci)
end
