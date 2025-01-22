"""
Contraction of two TTOs
Optionally, the contraction can be done with a function applied to the result.
"""
struct Contraction{T} <: BatchEvaluator{T}
    mpo::NTuple{2,TensorTrain{T,4}}
    leftcache::Dict{Vector{Tuple{Int,Int}},Matrix{T}}
    rightcache::Dict{Vector{Tuple{Int,Int}},Matrix{T}}
    f::Union{Nothing,Function}
    sitedims::Vector{Vector{Int}}
end


Base.length(obj::Contraction) = length(obj.mpo[1])

function sitedims(obj::Contraction{T})::Vector{Vector{Int}} where {T}
    return obj.sitedims
end

function Base.lastindex(obj::Contraction{T}) where {T}
    return lastindex(obj.mpo[1])
end

function Base.getindex(obj::Contraction{T}, i) where {T}
    return getindex(obj.mpo[1], i)
end

function Base.show(io::IO, obj::Contraction{T}) where {T}
    print(
        io,
        "$(typeof(obj)) of tensor trains with ranks $(rank(obj.mpo[1])) and $(rank(obj.mpo[2]))",
    )
end

function Contraction(
    a::TensorTrain{T,4},
    b::TensorTrain{T,4};
    f::Union{Nothing,Function}=nothing,
) where {T}
    mpo = a, b
    if length(unique(length.(mpo))) > 1
        throw(ArgumentError("Tensor trains must have the same length."))
    end
    for n = 1:length(mpo[1])
        if size(mpo[1][n], 3) != size(mpo[2][n], 2)
            error("Tensor trains must share the identical index at n=$(n)!")
        end
    end

    localdims1 = [size(mpo[1][n], 2) for n = 1:length(mpo[1])]
    localdims3 = [size(mpo[2][n], 3) for n = 1:length(mpo[2])]

    sitedims = [[x, y] for (x, y) in zip(localdims1, localdims3)]

    return Contraction(
        mpo,
        Dict{Vector{Tuple{Int,Int}},Matrix{T}}(),
        Dict{Vector{Tuple{Int,Int}},Matrix{T}}(),
        f,
        sitedims
    )
end

_localdims(obj::TensorTrain{<:Any,4}, n::Int)::Tuple{Int,Int} =
    (size(obj[n], 2), size(obj[n], 3))
_localdims(obj::Contraction{<:Any}, n::Int)::Tuple{Int,Int} =
    (size(obj.mpo[1][n], 2), size(obj.mpo[2][n], 3))

_getindex(x, indices) = ntuple(i -> x[indices[i]], length(indices))

function _contract(
    a::AbstractArray{T1,N1},
    b::AbstractArray{T2,N2},
    idx_a::NTuple{n1,Int},
    idx_b::NTuple{n2,Int}
) where {T1,T2,N1,N2,n1,n2}
    length(idx_a) == length(idx_b) || error("length(idx_a) != length(idx_b)")
    # check if idx_a contains only unique elements
    length(unique(idx_a)) == length(idx_a) || error("idx_a contains duplicate elements")
    # check if idx_b contains only unique elements
    length(unique(idx_b)) == length(idx_b) || error("idx_b contains duplicate elements")
    # check if idx_a and idx_b are subsets of 1:N1 and 1:N2
    all(1 <= idx <= N1 for idx in idx_a) || error("idx_a contains elements out of range")
    all(1 <= idx <= N2 for idx in idx_b) || error("idx_b contains elements out of range")

    rest_idx_a = setdiff(1:N1, idx_a)
    rest_idx_b = setdiff(1:N2, idx_b)

    amat = reshape(permutedims(a, (rest_idx_a..., idx_a...)), prod(_getindex(size(a), rest_idx_a)), prod(_getindex(size(a), idx_a)))
    bmat = reshape(permutedims(b, (idx_b..., rest_idx_b...)), prod(_getindex(size(b), idx_b)), prod(_getindex(size(b), rest_idx_b)))

    return reshape(amat * bmat, _getindex(size(a), rest_idx_a)..., _getindex(size(b), rest_idx_b)...)
end

function _unfuse_idx(obj::Contraction{T}, n::Int, idx::Int)::Tuple{Int,Int} where {T}
    return reverse(divrem(idx - 1, _localdims(obj, n)[1]) .+ 1)
end

function _fuse_idx(obj::Contraction{T}, n::Int, idx::Tuple{Int,Int})::Int where {T}
    return idx[1] + _localdims(obj, n)[1] * (idx[2] - 1)
end

function _extend_cache(oldcache::Matrix{T}, a_ell::Array{T,4}, b_ell::Array{T,4}, i::Int, j::Int) where {T}
    # (link_a, link_b) * (link_a, s, link_a') => (link_b, s, link_a')
    tmp1 = _contract(oldcache, a_ell[:, i, :, :], (1,), (1,))

    # (link_b, s, link_a') * (link_b, s, link_b') => (link_a', link_b')
    return _contract(tmp1, b_ell[:, :, j, :], (1, 2), (1, 2))
end

# Compute left environment
function evaluateleft(
    obj::Contraction{T},
    indexset::AbstractVector{Tuple{Int,Int}},
)::Matrix{T} where {T}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    a, b = obj.mpo

    if length(indexset) == 0
        return ones(T, 1, 1)
    end

    ell = length(indexset)
    if ell == 1
        i, j = indexset[1]
        return transpose(a[1][1, i, :, :]) * b[1][1, :, j, :]
    end

    key = collect(indexset)
    if !(key in keys(obj.leftcache))
        i, j = indexset[end]
        obj.leftcache[key] = _extend_cache(evaluateleft(obj, indexset[1:ell-1]), a[ell], b[ell], i, j)
    end

    return obj.leftcache[key]
end



# Compute right environment
function evaluateright(
    obj::Contraction{T},
    indexset::AbstractVector{Tuple{Int,Int}},
)::Matrix{T} where {T}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    a, b = obj.mpo

    N = length(obj)

    if length(indexset) == 0
        return ones(T, 1, 1)
    elseif length(indexset) == 1
        i, j = indexset[1]
        return a[end][:, i, :, 1] * transpose(b[end][:, :, j, 1])
    end

    ell = N - length(indexset) + 1

    key = collect(indexset)
    if !(key in keys(obj.rightcache))
        i, j = indexset[1]
        obj.rightcache[key] = _extend_cache(
            evaluateright(obj, indexset[2:end]),
            permutedims(a[ell], (4, 2, 3, 1)),
            permutedims(b[ell], (4, 2, 3, 1)),
            i, j)
    end

    return obj.rightcache[key]
end


function evaluate(obj::Contraction{T}, indexset::AbstractVector{Int})::T where {T}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    indexset_unfused = [_unfuse_idx(obj, n, indexset[n]) for n = 1:length(obj)]
    return evaluate(obj, indexset_unfused)
end

function evaluate(
    obj::Contraction{T},
    indexset::AbstractVector{Tuple{Int,Int}},
)::T where {T}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    midpoint = div(length(obj), 2)
    res = sum(
        evaluateleft(obj, indexset[1:midpoint]) .*
        evaluateright(obj, indexset[midpoint+1:end]),
    )

    if obj.f isa Function
        return obj.f(res)
    else
        return res
    end
end

function _lineari(dims, mi)::Integer
    ci = CartesianIndex(Tuple(mi))
    li = LinearIndices(Tuple(dims))
    return li[ci]
end

function lineari(sitedims::Vector{Vector{Int}}, indexset::Vector{MultiIndex})::Vector{Int}
    return [_lineari(sitedims[l], indexset[l]) for l in 1:length(indexset)]
end


function (obj::Contraction{T})(indexset::AbstractVector{Int})::T where {T}
    return evaluate(obj, indexset)
end

function (obj::Contraction{T})(indexset::AbstractVector{<:AbstractVector{Int}})::T where {T}
    return evaluate(obj, lineari(obj.sitedims, indexset))
end

function (obj::Contraction{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    return batchevaluate(obj, leftindexset, rightindexset, Val(M))
end

function batchevaluate(obj::Contraction{T},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
    projector::Union{Nothing,AbstractVector{<:AbstractVector{<:Integer}}}=nothing)::Array{T,M + 2} where {T,M}
    N = length(obj)
    Nr = length(rightindexset[1])
    s_ = length(leftindexset[1]) + 1
    e_ = N - length(rightindexset[1])
    a, b = obj.mpo

    if projector === nothing
        projector = [fill(0, length(obj.sitedims[n])) for n in s_:e_]
    end
    length(projector) == M || error("Length mismatch: length of projector (=$(length(projector))) must be $(M)")
    for n in s_:e_
        length(projector[n - s_ + 1]) == 2 || error("Invalid projector at $n: $(projector[n - s_ + 1]), the length must be 2")
        all(0 .<= projector[n - s_ + 1] .<= obj.sitedims[n]) || error("Invalid projector: $(projector[n - s_ + 1])")
    end


    # Unfused index
    leftindexset_unfused = [
        [_unfuse_idx(obj, n, idx) for (n, idx) in enumerate(idxs)] for idxs in leftindexset
    ]
    rightindexset_unfused = [
        [_unfuse_idx(obj, N - Nr + n, idx) for (n, idx) in enumerate(idxs)] for
        idxs in rightindexset
    ]

    t1 = time_ns()
    linkdims_a = vcat(1, linkdims(a), 1)
    linkdims_b = vcat(1, linkdims(b), 1)

    left_ = Array{T,3}(undef, length(leftindexset), linkdims_a[s_], linkdims_b[s_])
    for (i, idx) in enumerate(leftindexset_unfused)
        left_[i, :, :] .= evaluateleft(obj, idx)
    end
    t2 = time_ns()

    right_ = Array{T,3}(
        undef,
        linkdims_a[e_+1],
        linkdims_b[e_+1],
        length(rightindexset),
    )
    for (i, idx) in enumerate(rightindexset_unfused)
        right_[:, :, i] .= evaluateright(obj, idx)
    end
    t3 = time_ns()

    # (left_index, link_a, link_b, site[s_] * site'[s_] *  ... * site[e_] * site'[e_])
    leftobj::Array{T,4} = reshape(left_, size(left_)..., 1)
    return_size_siteinds = Int[]
    for n = s_:e_
        slice_ab, shape_ab = projector_to_slice(projector[n - s_ + 1])
        a_n = begin
            a_n_org = obj.mpo[1][n]
            tmp = a_n_org[:, slice_ab[1], :, :]
            reshape(tmp, size(a_n_org, 1), shape_ab[1], size(a_n_org)[3:4]...)
        end
        b_n = begin
            b_n_org = obj.mpo[2][n]
            tmp = b_n_org[:, :, slice_ab[2], :]
            reshape(tmp, size(b_n_org, 1), size(b_n_org, 2), shape_ab[2], size(b_n_org, 4))
        end
        push!(return_size_siteinds, size(a_n, 2) * size(b_n, 3))

        #(left_index, link_a, link_b, S) * (link_a, site[n], shared, link_a')
        #  => (left_index, link_b, S, site[n], shared, link_a')
        tmp1 = _contract(leftobj, a_n, (2,), (1,))

        # (left_index, link_b, S, site[n], shared, link_a') * (link_b, shared, site'[n], link_b')
        #  => (left_index, S, site[n], link_a', site'[n], link_b')
        tmp2 = _contract(tmp1, b_n, (2, 5), (1, 2))

        # (left_index, S, site[n], link_a', site'[n], link_b')
        #  => (left_index, link_a', link_b', S, site[n], site'[n])
        tmp3 = permutedims(tmp2, (1, 4, 6, 2, 3, 5))

        leftobj = reshape(tmp3, size(tmp3)[1:3]..., :)
    end

    return_size = (
        length(leftindexset),
        return_size_siteinds...,
        length(rightindexset),
    )
    t5 = time_ns()

    # (left_index, link_a, link_b, S) * (link_a, link_b, right_index)
    #   => (left_index, S, right_index)
    res = _contract(leftobj, right_, (2, 3), (1, 2))

    if obj.f isa Function
        res .= obj.f.(res)
    end

    return reshape(res, return_size)
end


function _contractsitetensors(a::Array{T,4}, b::Array{T,4})::Array{T,4} where {T}
    # indices: (link_a, s1, s2, link_a') * (link_b, s2, s3, link_b')
    ab::Array{T,6} = _contract(a, b, (3,), (2,))
    # => indices: (link_a, s1, link_a', link_b, s3, link_b')
    abpermuted = permutedims(ab, (1, 4, 2, 5, 3, 6))
    # => indices: (link_a, link_b, s1, s3, link_a', link_b')
    return reshape(abpermuted,
        size(a, 1) * size(b, 1),  # link_a * link_b
        size(a, 2), size(b, 3),  # s1, s3
        size(a, 4) * size(b, 4)   # link_a' * link_b'
    )
end

function contract_naive(
    a::TensorTrain{T,4}, b::TensorTrain{T,4};
    tolerance=0.0, maxbonddim=typemax(Int)
)::TensorTrain{T,4} where {T}
    return contract_naive(Contraction(a, b); tolerance, maxbonddim)
end

function contract_naive(
    obj::Contraction{T};
    tolerance=0.0, maxbonddim=typemax(Int)
)::TensorTrain{T,4} where {T}
    if obj.f isa Function
        error("Naive contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
    end

    a, b = obj.mpo
    tt = TensorTrain{T,4}(_contractsitetensors.(sitetensors(a), sitetensors(b)))
    if tolerance > 0 || maxbonddim < typemax(Int)
        compress!(tt, :SVD; tolerance, maxbonddim)
    end
    return tt
end

function _reshape_fusesites(t::AbstractArray{T}) where {T}
    shape = size(t)
    return reshape(t, shape[1], prod(shape[2:end-1]), shape[end]), shape[2:end-1]
end

function _reshape_splitsites(
    t::AbstractArray{T},
    legdims::Union{AbstractVector{Int},Tuple},
) where {T}
    return reshape(t, size(t, 1), legdims..., size(t, ndims(t)))
end

function _findinitialpivots(f, localdims, nmaxpivots)::Vector{MultiIndex}
    pivots = MultiIndex[]
    for _ in 1:nmaxpivots
        pivot_ = [rand(1:d) for d in localdims]
        pivot_ = optfirstpivot(f, localdims, pivot_)
        if abs(f(pivot_)) == 0.0
            continue
        end
        push!(pivots, pivot_)
    end
    return pivots
end

function contract_TCI(
    A::TensorTrain{ValueType,4},
    B::TensorTrain{ValueType,4};
    initialpivots::Union{Int,Vector{MultiIndex}}=10,
    f::Union{Nothing,Function}=nothing,
    kwargs...
) where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end
    if !all([sitedim(A, i)[2] == sitedim(B, i)[1] for i = 1:length(A)])
        throw(
            ArgumentError(
                "Cannot contract tensor trains with non-matching site dimensions.",
            ),
        )
    end
    matrixproduct = Contraction(A, B; f=f)
    localdims = prod.(matrixproduct.sitedims)
    if initialpivots isa Int
        initialpivots = _findinitialpivots(matrixproduct, localdims, initialpivots)
        if isempty(initialpivots)
            error("No initial pivots found.")
        end
    end

    tci, ranks, errors = crossinterpolate2(
        ValueType,
        matrixproduct,
        localdims,
        initialpivots;
        kwargs...,
    )
    legdims = [_localdims(matrixproduct, i) for i = 1:length(tci)]
    return TensorTrain{ValueType,4}(
        [_reshape_splitsites(t, d) for (t, d) in zip(tci, legdims)]
    )
end

"""
See SVD version:
https://tensornetwork.org/mps/algorithms/zip_up_mpo/
"""
function contract_zipup(
    A::TensorTrain{ValueType,4},
    B::TensorTrain{ValueType,4};
    tolerance::Float64=1e-12,
    method::Symbol=:SVD, # :SVD, :LU, :CI
    maxbonddim::Int=typemax(Int)
) where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end
    R::Array{ValueType,3} = ones(ValueType, 1, 1, 1)

    sitetensors = Vector{Array{ValueType,4}}(undef, length(A))
    for n in 1:length(A)
        # R:     (link_ab, link_an, link_bn)
        # A[n]:  (link_an, s_n, s_n', link_anp1)
        #println("Time RA:")
        RA = _contract(R, A[n], (2,), (1,))

        # RA[n]: (link_ab, link_bn, s_n, s_n' link_anp1)
        # B[n]:  (link_bn, s_n', s_n'', link_bnp1)
        # C:     (link_ab, s_n, link_anp1, s_n'', link_bnp1)
        #  =>    (link_ab, s_n, s_n'', link_anp1, link_bnp1)
        #println("Time C:")
        C = permutedims(_contract(RA, B[n], (2, 4), (1, 2)), (1, 2, 4, 3, 5))
        if n == length(A)
            sitetensors[n] = reshape(C, size(C)[1:3]..., 1)
            break
        end

        # Cmat:  (link_ab * s_n * s_n'', link_anp1 * link_bnp1)

        #lu = rrlu(Cmat; reltol, abstol, leftorthogonal=true)
        # println("Time factorize:")
        left, right, newbonddim = _factorize(
            reshape(C, prod(size(C)[1:3]), prod(size(C)[4:5])),
            method; tolerance, maxbonddim
        )

        # U:     (link_ab, s_n, s_n'', link_ab_new)
        sitetensors[n] = reshape(left, size(C)[1:3]..., newbonddim)

        # R:     (link_ab_new, link_an, link_bn)
        R = reshape(right, newbonddim, size(C)[4:5]...)
    end

    return TensorTrain{ValueType,4}(sitetensors)
end

"""
    function contract(
        A::TensorTrain{V1,4},
        B::TensorTrain{V2,4};
        algorithm::Symbol=:TCI,
        tolerance::Float64=1e-12,
        maxbonddim::Int=typemax(Int),
        f::Union{Nothing,Function}=nothing,
        kwargs...
    ) where {V1,V2}

Contract two tensor trains `A` and `B`.

Currently, two implementations are available:
 1. `algorithm=:TCI` constructs a new TCI that fits the contraction of `A` and `B`.
 2. `algorithm=:naive` uses a naive tensor contraction and subsequent SVD recompression of the tensor train.
 2. `algorithm=:zipup` uses a naive tensor contraction with on-the-fly LU decomposition.

Arguments:
- `A` and `B` are the tensor trains to be contracted.
- `algorithm` chooses the algorithm used to evaluate the contraction.
- `tolerance` is the tolerance of the TCI or SVD recompression.
- `maxbonddim` sets the maximum bond dimension of the resulting tensor train.
- `f` is a function to be applied elementwise to the result. This option is only available with `algorithm=:TCI`.
- `method` chooses the method used for the factorization in the `algorithm=:zipup` case (`:SVD` or `:LU`).
- `kwargs...` are forwarded to [`crossinterpolate2`](@ref) if `algorithm=:TCI`.
"""
function contract(
    A::TensorTrain{V1,4},
    B::TensorTrain{V2,4};
    algorithm::Symbol=:TCI,
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    method::Symbol=:SVD,
    f::Union{Nothing,Function}=nothing,
    kwargs...
)::TensorTrain{promote_type(V1,V2),4} where {V1,V2}
    Vres = promote_type(V1, V2)
    A_ = TensorTrain{Vres,4}(A)
    B_ = TensorTrain{Vres,4}(B)
    if algorithm === :TCI
        return contract_TCI(A_, B_; tolerance=tolerance, maxbonddim=maxbonddim, f=f, kwargs...)
    elseif algorithm === :naive
        if f !== nothing
            error("Naive contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        return contract_naive(A_, B_; tolerance=tolerance, maxbonddim=maxbonddim)
    elseif algorithm === :zipup
        if f !== nothing
            error("Zipup contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        return contract_zipup(A_, B_; tolerance=tolerance, maxbonddim=maxbonddim, method=method)
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))
    end
end

function contract(
    A::Union{TensorCI1{V},TensorCI2{V},TensorTrain{V,3}},
    B::TensorTrain{V2,4};
    kwargs...
)::TensorTrain{promote_type(V,V2),3} where {V,V2}
    tt = contract(TensorTrain{4}(A, [(1, s...) for s in sitedims(A)]), B; kwargs...)
    return TensorTrain{3}(tt, prod.(sitedims(tt)))
end

function contract(
    A::TensorTrain{V,4},
    B::Union{TensorCI1{V2},TensorCI2{V2},TensorTrain{V2,3}};
    kwargs...
)::TensorTrain{promote_type(V,V2),3} where {V,V2}
    tt = contract(A, TensorTrain{4}(B, [(s..., 1) for s in sitedims(B)]); kwargs...)
    return TensorTrain{3}(tt, prod.(sitedims(tt)))
end

function contract(
    A::Union{TensorCI1{V},TensorCI2{V},TensorTrain{V,3}},
    B::Union{TensorCI1{V2},TensorCI2{V2},TensorTrain{V2,3}};
    kwargs...
)::promote_type(V,V2) where {V,V2}
    tt = contract(TensorTrain{4}(A, [(1, s...) for s in sitedims(A)]), TensorTrain{4}(B, [(s..., 1) for s in sitedims(B)]); kwargs...)
    return prod(prod.(tt.sitetensors))
end
