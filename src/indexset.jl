struct IndexSet{T}
    to_int::Dict{T,Int}
    from_int::Vector{T}

    function IndexSet{T}() where {T}
        return new{T}(Dict{T,Int}(), [])
    end

    function IndexSet(l::Vector{T}) where {T}
        to_int = Dict([(l[i], i) for i in eachindex(l)])
        return new{T}(to_int, l)
    end
end

function Base.getindex(is::IndexSet{T}, i::Int) where {T}
    return is.from_int[i]
end

function pos(is::IndexSet{T}, indices::T) where {T}
    return is.to_int[indices]
end

function pos(is::IndexSet{T}, indicesvec::AbstractVector{T}) where {T}
    return [pos(is, i) for i in indicesvec]
end

function Base.setindex!(is::IndexSet{T}, x::T, i::Int) where {T}
    is.to_int[x] = i
    is.from_int[i] = x
end

function Base.push!(is::IndexSet{T}, x::T) where {T}
    push!(is.from_int, x)
    is.to_int[x] = size(is.from_int, 1)
end

function Base.length(is::IndexSet{T}) where {T}
    return length(is.from_int)
end

function Base.isempty(is::IndexSet{T}) where {T}
    return Base.isempty(is.from_int)
end

function ==(a::IndexSet{T}, b::IndexSet{T}) where {T}
    return a.from_int == b.from_int
end
