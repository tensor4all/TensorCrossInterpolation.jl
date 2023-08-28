function TCI.evaluate(
    mps::Union{ITensors.MPS,ITensors.MPO},
    indexspecs::Vararg{AbstractVector{<:Tuple{ITensors.Index,Int}}}
)
    if isempty(indexspecs)
        error("Please specify at which indices you wish to evaluate the MPS.")
    elseif any(length.(indexspecs) .!= length(mps))
        error("Need one index per MPS leg")
    end

    V = ITensor(1.0)
    for j in eachindex(indexspecs[1])
        states = prod(state(spec[j]...) for spec in indexspecs)
        V *= mps[j] * states
    end
    return scalar(V)
end

function TCI.evaluate(
    mps::Union{ITensors.MPS,ITensors.MPO},
    indices::AbstractVector{<:ITensors.Index},
    indexvalues::AbstractVector{Int}
)
    return TCI.evaluate(mps, collect(zip(indices, indexvalues)))
end
