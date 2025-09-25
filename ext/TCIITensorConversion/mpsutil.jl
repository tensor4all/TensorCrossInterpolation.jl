@doc raw"""
    function evaluate(
        mps::Union{ITensorMPS.MPS,ITensorMPS.MPO},
        indexspecs::Vararg{AbstractVector{<:Tuple{ITensorMPS.Index,Int}}}
    )

Evaluates an MPS or MPO for a given set of index values.

- `indexspec` is a list of tuples, where each tuple contains an `ITensorMPS.Index` object specifying an index, and an `Int` corresponding to the value of the specified index.

If many evaluations are necessary, it may be advantageous to convert your MPS to a [`TensorCrossInterpolation.TTCache`](@ref) object first.
"""
function evaluate(
    mps::Union{ITensorMPS.MPS,ITensorMPS.MPO},
    indexspecs::Vararg{AbstractVector{<:Tuple{ITensors.Index,Int}}}
)
    if isempty(indexspecs)
        error("Please specify at which indices you wish to evaluate the MPS.")
    elseif any(length.(indexspecs) .!= length(mps))
        error("Need one index per MPS leg")
    end

    V = ITensor(1.0)
    for j in eachindex(indexspecs[1])
        states = prod(ITensorMPS.state(spec[j]...) for spec in indexspecs)
        V *= mps[j] * states
    end
    return scalar(V)
end
@doc raw"""
    function evaluate(
        mps::Union{ITensorMPS.MPS,ITensorMPS.MPO},
        indices::AbstractVector{<:ITensors.Index},
        indexvalues::AbstractVector{Int}
    )

Evaluates an MPS or MPO for a given set of index values.

- `indices` is a list of `ITensors.Index` objects that specifies the order in which indices are given.
- `indexvalues` is a list of integer values in the same order as `indices`.

If many evaluations are necessary, it may be advantageous to convert your MPS to a [`TensorCrossInterpolation.TTCache`](@ref) object first.
"""
function evaluate(
    mps::Union{ITensorMPS.MPS,ITensorMPS.MPO},
    indices::AbstractVector{<:ITensors.Index},
    indexvalues::AbstractVector{Int}
)
    return evaluate(mps, collect(zip(indices, indexvalues)))
end
