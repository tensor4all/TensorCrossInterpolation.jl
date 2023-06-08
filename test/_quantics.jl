"""
From an index to a bit list
"""
function index_to_quantics(i::Int, nbit::Int)::Vector{Int}
    @assert 1 ≤ i ≤ 2^nbit
    mask = 1 << (nbit-1)
    bitlist = ones(Int, nbit)
    for n in 1:nbit
        bitlist[n] = (mask & (i-1)) >> (nbit-n) + 1
        mask = mask >> 1
    end
    return bitlist
end

"""
From a bit lit to an index
"""
function quantics_to_index(bitlist::AbstractArray{Int})::Int
    @assert all(1 .≤ bitlist .≤ 2)
    nbit = length(bitlist)
    i = 1
    tmp = 2^(nbit-1)
    for n in eachindex(bitlist)
        i += tmp * (bitlist[n] -1)
        tmp = tmp >> 1
    end
    return i
end