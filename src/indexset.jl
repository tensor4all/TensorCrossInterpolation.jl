struct IndexSet{T}
    toint::Dict{T,Int}
    fromint::Vector{T}

    function IndexSet{T}() where {T}
        return new{T}(Dict{T,Int}(), [])
    end

    function IndexSet(l::Vector{T}) where {T}
        toint = Dict([(l[i], i) for i in eachindex(l)])
        return new{T}(toint, l)
    end
end

function Base.getindex(is::IndexSet{T}, i::Int) where {T}
    return is.fromint[i]
end

function pos(is::IndexSet{T}, indices::T) where {T}
    return is.toint[indices]
end

function pos(is::IndexSet{T}, indicesvec::AbstractVector{T}) where {T}
    return [pos(is, i) for i in indicesvec]
end

function Base.setindex!(is::IndexSet{T}, x::T, i::Int) where {T}
    is.toint[x] = i
    is.fromint[i] = x
end

function Base.push!(is::IndexSet{T}, x::T) where {T}
    push!(is.fromint, x)
    is.toint[x] = size(is.fromint, 1)
end

function Base.length(is::IndexSet{T}) where {T}
    return length(is.fromint)
end

function Base.isempty(is::IndexSet{T}) where {T}
    return Base.isempty(is.fromint)
end

function ==(a::IndexSet{T}, b::IndexSet{T}) where {T}
    return a.fromint == b.fromint
end
