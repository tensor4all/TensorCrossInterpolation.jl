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

# Compute full left environment
function leftenvironment(
    A::Vector{Array{T,4}},
    B::Vector{Array{T,4}},
    X::Vector{Array{T,4}},
    i::Int;
    L::Array{T, 3} = ones(T, 1, 1, 1),
    Li::Int = 1
)::Array{T, 3} where {T}
    for ell in Li:i
        L = _contract(_contract(_contract(L, A[ell], (1,), (1,)), B[ell], (1,4,), (1,2,)), conj(X[ell]), (1,2,4,), (1,2,3,))
    end
    return L
end

# Compute full right environment
function rightenvironment(
    A::Vector{Array{T,4}},
    B::Vector{Array{T,4}},
    X::Vector{Array{T,4}},
    i::Int;
    R::Array{T, 3} = ones(T, 1, 1, 1),
    Ri::Int = length(A),
    )::Array{T, 3} where {T}
    for ell in Ri:-1:i
        R = permutedims(_contract(conj(X[ell]), _contract(B[ell], _contract(A[ell], R, (4,), (1,)), (2,4,), (3,4,)), (2,3,4,), (4,2,5,)), (3,2,1,))
    end
    return R
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
    method::Symbol=:SVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    kwargs...
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
        #println(size(C))
        if n == length(A)
            sitetensors[n] = reshape(C, size(C)[1:3]..., 1)
            break
        end

        # Cmat:  (link_ab * s_n * s_n'', link_anp1 * link_bnp1)

        #lu = rrlu(Cmat; reltol, abstol, leftorthogonal=true)
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

function contract_distr_zipup(
    A::TensorTrain{ValueType,4},
    B::TensorTrain{ValueType,4};
    tolerance::Float64=1e-12,
    method::Symbol=:SVD,
    p::Int=16,
    maxbonddim::Int=typemax(Int),
    estimatedbond::Union{Nothing,Int}=nothing,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    kwargs...
) where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    R::Array{ValueType,3} = ones(ValueType, 1, 1, 1)

    if MPI.Initialized()
        if subcomm != nothing
            comm = subcomm
        else
            comm = MPI.COMM_WORLD
        end
        mpirank = MPI.Comm_rank(comm)
        juliarank = mpirank + 1
	    nprocs = MPI.Comm_size(comm)
        if estimatedbond == nothing
            estimatedbond = maxbonddim == typemax(Int) ? max(maximum(linkdims(A)), maximum(linkdims(B))) : maxbonddim
        end
        noderanges = _noderanges(nprocs, length(A), size(A[1])[2]*size(B[1])[3], estimatedbond; interbond=true, algorithm=:mpompo)
        Us = Vector{Array{ValueType,3}}(undef, length(noderanges[juliarank]))
        Vts = Vector{Array{ValueType,3}}(undef, length(noderanges[juliarank]))
    else
        println("Warning! Distributed zipup has been chosen, but MPI is not initialized, please use initializempi() before using contract() and use finalizempi() afterwards")
        return contract_zipup(A, B; tolerance, method, maxbonddim, kwargs...)
    end

    if nprocs == 1
        println("Warning! Distributed zipup has been chosen, but only one process is running")
        return contract_zipup(A, B; tolerance, method, maxbonddim, kwargs...)
    end
    
    finalsitetensors =  Vector{Array{ValueType,4}}(undef, length(A))

    time = time_ns()
    if method == :SVD || method == :RSVD
        if maxbonddim == typemax(Int)
            println("Warning! distrzipup uses always Randomized SVD, which cannot accept maxbonddim=typemax(Int), it will be set to the maximum bond of A or B")
            maxbonddim = max(maximum(linkdims(A)), maximum(linkdims(B)))
        end
        for (i, n) in enumerate(noderanges[juliarank])
            # Random matrix
            G = randn(size(A[n])[end], size(B[n])[end], maxbonddim+p)
            # Projection on smaller space
            Y = _contract(A[n], G, (4,), (1,))
            Y = _contract(B[n], Y, (2,4,), (3,4))
            Y = permutedims(Y, (3,1,4,2,5,))

            # QR decomposition
            Q = Matrix(LinearAlgebra.qr!(reshape(Y, (prod(size(Y)[1:4]), size(Y)[end]))).Q)
            Q = reshape(Q, (size(Y)[1:4]..., min(prod(size(Y)[1:4]), size(Y)[end])))
            Qt = permutedims(Q, (5, 3, 4, 1, 2))
            # Smaller object to SVD
            to_svd = _contract(Qt, A[n], (2,4,), (2,1,))
            to_svd = _contract(to_svd, B[n], (2,3,4,), (3,1,2,))
            factorization = svd(reshape(to_svd, (size(to_svd)[1], prod(size(to_svd)[2:3]))))
            newbonddimr = min(
                replacenothing(findlast(>(tolerance), Array(factorization.S)), 1),
                maxbonddim
            )

            U = _contract(Q, factorization.U[:, 1:newbonddimr], (5,), (1,))
            US = _contract(U, Diagonal(factorization.S[1:newbonddimr]), (5,), (1,))
            U, Vt, newbonddiml = _factorize(
                reshape(US, prod(size(US)[1:2]), prod(size(US)[3:5])),
                method; tolerance, maxbonddim, leftorthogonal=true
            )
            U = reshape(U, (size(A[n])[1], size(B[n])[1], newbonddiml))
            finalsitetensors[n] = reshape(Vt, newbonddiml, size(US)[3:5]...)

            Us[i] = U
            Vts[i] = reshape(factorization.Vt[1:newbonddimr, :], newbonddimr, size(to_svd)[end-1], size(to_svd)[end])
        end
    elseif method == :CI || method == :LU
        for (i, n) in enumerate(noderanges[juliarank])
            dimsA = size(A[n])
            dimsB = size(B[n])
            function f24(j,k; print=false)
                b1, a1 = Tuple(CartesianIndices((dimsB[1], dimsA[1]))[j])  # Extract (a1, b1) from j
                b4, a4, b3, a2 = Tuple(CartesianIndices((dimsB[4], dimsA[4], dimsB[3], dimsA[2]))[k])  # Extract (a2, b3, a4, b4) from k
                sum(A[n][a1, a2, c, a4] * B[n][b1, c, b3, b4] for c in 1:dimsA[3])  # Summing over the contracted index
            end
            
            # Non Zero element to start PRRLU decomposition
            nz = nothing
            size_A = size(A[n])  # (i, j, k, l)
            size_B = size(B[n])  # (m, k, nn, o)
            
            for i in 1:size_A[1], m in 1:size_B[1], j in 1:size_A[2], nn in 1:size_B[3], l in 1:size_A[4], o in 1:size_B[4]
                sum_val = zero(eltype(A[n]))
                for k in 1:size_A[3]  # k index contraction
                    if A[n][i, j, k, l] != 0 && B[n][m, k, nn, o] != 0
                        sum_val += A[n][i, j, k, l] * B[n][m, k, nn, o]
                    end
                end
                if sum_val != 0
                    nz = [i, m, j, nn, l, o]
                    break
                end
            end
            
            # CartesianIndices are read the other way around
            I0 = [(nz[1]-1)*dimsB[1] + nz[2], (nz[3]-1)*dimsB[3]*dimsA[4]*dimsB[4] + (nz[4]-1)*dimsA[4]*dimsB[4] + (nz[5]-1)*dimsB[4] + nz[6]]
            J0 = [(nz[1]-1)*dimsB[1] + nz[2], (nz[3]-1)*dimsB[3]*dimsA[4]*dimsB[4] + (nz[4]-1)*dimsA[4]*dimsB[4] + (nz[5]-1)*dimsB[4] + nz[6]]
            factorization = arrlu(ValueType, f24, (dimsA[1]*dimsB[1], dimsA[2]*dimsB[3]*dimsA[4]*dimsB[4]), I0, J0; abstol=tolerance, maxrank=maxbonddim)
            L = left(factorization)
            U = right(factorization)
            newbonddiml = npivots(factorization)
            
            U_3_2 = reshape(U, size(U)[1]*dimsA[2]*dimsB[3], dimsA[4]*dimsB[4])
            U, Vt, newbonddimr = _factorize(U_3_2, :SVD; tolerance, maxbonddim)
            Us[i] = reshape(L, dimsA[1], dimsB[1], newbonddiml)
            finalsitetensors[n] = reshape(U, newbonddiml, dimsA[2], dimsB[3], newbonddimr)
            Vts[i] = reshape(Vt, newbonddimr, dimsA[4], dimsB[4])
        end
    end
    
    # Exchange Vt to left
    if MPI.Initialized()
        MPI.Barrier(comm)
        reqs = MPI.Request[]
        if juliarank < nprocs
            push!(reqs, MPI.isend(Vts[end], comm; dest=mpirank+1, tag=mpirank))
        end
        if juliarank > 1
            Vts_l = MPI.recv(comm; source=mpirank-1, tag=mpirank-1)
        end
        MPI.Waitall(reqs)
        MPI.Barrier(comm)
    end

    # contraction
    time = time_ns()
    for (i, n) in enumerate(noderanges[juliarank])
        if i > 1
            VtU = _contract(Vts[i-1], Us[i], (2,3,), (1,2,))
            finalsitetensors[n] = _contract(VtU, finalsitetensors[n], (2,), (1,))
        elseif n > 1
            VtU = _contract(Vts_l, Us[i], (2,3,), (1,2,))
            finalsitetensors[n] = _contract(VtU, finalsitetensors[n], (2,), (1,))
        end
    end
    if juliarank == 1
        finalsitetensors[1] = _contract(_contract(R, Us[1], (2,3,), (1,2,)), finalsitetensors[1], (2,), (1,))
    end
    if juliarank == nprocs
        finalsitetensors[end] = _contract(finalsitetensors[end], _contract(Vts[end], R, (2,3,), (1,2,)), (4,), (1,))
    end

    # All to all exchange
    if MPI.Initialized()
        all_sizes = [length(noderanges[r]) for r in 1:nprocs]
        shapes = [isassigned(finalsitetensors, i) ? size(finalsitetensors[i]) : (0,0,0,0) for i in 1:length(finalsitetensors)]
        shapesbuffer = VBuffer(shapes, all_sizes)
        MPI.Allgatherv!(shapesbuffer, comm)

        lengths = [sum(prod.(shapes)[noderanges[r]]) for r in 1:nprocs]
        all_length = sum(prod.(shapes))
        vec_tensors = zeros(all_length)

        if juliarank == 1
            before_me = 0
        else
            before_me = sum(prod.(shapes[1:noderanges[juliarank][1]-1]))
        end
        me = sum(prod.(shapes[noderanges[juliarank]]))
        vec_tensors[before_me+1:before_me+me] = vcat([vec(finalsitetensors[i]) for i in 1:length(finalsitetensors) if isassigned(finalsitetensors, i)]...)

        sendrecvbuf = VBuffer(vec_tensors, lengths)
        MPI.Allgatherv!(sendrecvbuf, comm)

        idx = 1
        for i in 1:length(finalsitetensors)
            s = shapes[i]
            len = prod(s)
            finalsitetensors[i] = reshape(view(vec_tensors, idx:idx+len-1), s)
            idx += len
        end
    end
    MPI.Barrier(comm)

    return TensorTrain{ValueType,4}(finalsitetensors)
end

# If Ri = i, then R is the right environment until site i+1, which has size (size(A[i])[end], size(B[i])[end], size(X[i]])[end])
# If Li = i, then L is the left environment until site i-1, which has size (size(A[i])[1], size(B[i])[1], size(X[i]])[1])
function updatecore!(A::Vector{Array{ValueType,4}}, B::Vector{Array{ValueType,4}}, X::Vector{Array{ValueType,4}}, i::Int;
    method::Symbol=:SVD, tolerance::Float64=1e-8, maxbonddim::Int=typemax(Int), direction::Symbol=:forward,
    R::Array{ValueType,3}=ones(ValueType, 1, 1, 1), Ri::Int=length(A), L::Array{ValueType,3}=ones(ValueType, 1, 1, 1), Li::Int=1
    ) where {ValueType}
    L = leftenvironment(A, B, X, i-1; L, Li)
    R = rightenvironment(A, B, X, i+2; R, Ri)

    Le = _contract(_contract(L, A[i], (1,), (1,)), B[i], (1, 4), (1, 2))
    Re = _contract(B[i+1], _contract(A[i+1], R, (4,), (1,)), (2, 4), (3, 4))
    Ce = _contract(Le, Re, (3, 5), (3, 1))
    Ce = permutedims(Ce, (1, 2, 3, 5, 4, 6))

    left, right, newbonddim = _factorize(
        reshape(Ce, prod(size(Ce)[1:3]), prod(size(Ce)[4:6])),
        method; tolerance, maxbonddim, leftorthogonal=(direction == :forward ? true : false)
    )

    X_i = reshape(left, :, size(X[i])[2:3]..., newbonddim)
    X_i1 = reshape(right, newbonddim, size(X[i+1])[2:3]..., :)

    if size(X[i]) != size(X_i) || size(X[i+1]) != size(X_i1)
       diff = Inf
    else
        diff = maximum(abs, X[i] .- X_i)
        diff = max(diff, maximum(abs, X[i+1] .- X_i1))
    end
    X[i] = X_i
    X[i+1] = X_i1
    return diff, L, R
end

"""
    function contract_fit(A::TensorTrain{ValueType,4}, B::TensorTrain{ValueType,4})

Conctractos tensor trains A and B using the fit algorithm.

# Keyword Arguments

- `nsweep::Int`: Number of sweeps to perform during the algorithm.
- `initial_guess`: Initial guess of the solution A*B.
- `tolerance::Float64`: Convergence tolerance for the iterative algorithm.
- `method::Symbol`: Algorithm or method to use for the computation :SVD, :RSVD, :LU, :CI.
- `maxbonddim::Int`: Maximum bond dimension allowed during the decomposition.
"""
#- `initial_guess` is an optional initial guess for the tensor train. If not provided, a tensor train of rank one is used. This must have coherent bond dimension (i.e. start and finish with less than [d,d^2,d^3,...,d^3,d^2,d]).
function contract_fit(
    mpoA::TensorTrain{ValueType,4},
    mpoB::TensorTrain{ValueType,4};
    nsweeps::Int=2,
    initial_guess::Union{Nothing,TensorTrain{ValueType,4}}=nothing,
    tolerance::Float64=1e-12,
    method::Symbol=:SVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    kwargs...
) where {ValueType}
    if length(mpoA) != length(mpoB)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    A = [deepcopy(a) for a in mpoA.sitetensors]
    B = [deepcopy(b) for b in mpoB.sitetensors]

    n = length(A)
    Rs = Vector{Array{ValueType, 3}}(undef, n)
    Ls = Vector{Array{ValueType, 3}}(undef, n)

    if !isnothing(initial_guess)
        X = [deepcopy(a) for a in initial_guess.sitetensors]
    else
        X = [ones(ValueType, 1, size(A[i])[2], size(B[i])[3], 1) for i in 1:n]
    end
    
    centercanonicalize!(A, 1)
    centercanonicalize!(B, 1)
    centercanonicalize!(X, 1)
    # Precompute right environments
    for i in n:-1:3
        if i == n
            Rs[i] = rightenvironment(A, B, X, i)
        else
            Rs[i] = rightenvironment(A, B, X, i; R=Rs[i+1], Ri=i)
        end
    end
    
    # It doesn't matter if we repeat update in 1 or n-1, those are negligible
    direction = :forward
    for sweep in 1:nsweeps
        max_diff = 0.0
        # Update cores and store Left or Right environment
        if direction == :forward
            for i in 1:n-1
                centercanonicalize!(A, i)
                centercanonicalize!(B, i)
                centercanonicalize!(X, i)
                if i == 1
                    diff, _, _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction, R=Rs[i+2], Ri=i+1)
                elseif i == 2
                    diff, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction, R=Rs[i+2], Ri=i+1)
                elseif i < n-1
                    diff, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction, L=Ls[i-2], Li=i-1, R=Rs[i+2], Ri=i+1)
                else # i == n-1
                    diff, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction, L=Ls[i-2], Li=i-1)
                end
                max_diff = max(max_diff, diff)
            end
            direction = :backward
        elseif direction == :backward
            for i in n-1:-1:1
                centercanonicalize!(A, i)
                centercanonicalize!(B, i)
                centercanonicalize!(X, i)
                if i == n-1
                    diff, _, _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction, L=Ls[i-1], Li=i)
                elseif i == n-2
                    diff, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction, L=Ls[i-1], Li=i)
                elseif i > 1
                    diff, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction, L=Ls[i-1], Li=i, R=Rs[i+3], Ri=i+2)
                else # i == 1
                    diff, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction, R=Rs[i+3], Ri=i+2)
                end
                max_diff = max(max_diff, diff)
            end
            direction = :forward
        end
        if max_diff < tolerance
            break
        end
    end
    return TensorTrain{ValueType,4}(X)
end

"""
    function contract_distr_fit(A::TensorTrain{ValueType,4}, B::TensorTrain{ValueType,4})

Conctractos tensor trains A and B using the fit algorithm.

# Keyword Arguments

- `nsweep::Int`: Number of sweeps to perform during the algorithm.
- `initial_guess`: Initial guess of the solution A*B.
- `subcomm::Union{Nothing, MPI.Comm}`: Subcommunicator to use for the distributed computation. If nothing, the global communicator is used.
- `tolerance::Float64`: Convergence tolerance for the iterative algorithm.
- `method::Symbol`: Algorithm or method to use for the computation :SVD, :RSVD, :LU, :CI.
- `maxbonddim::Int`: Maximum bond dimension allowed during the decomposition.
- `synchedmpo::Bool`: Tells whether the input MPOs are synchronized. Use "true" if you know that the input MPOs are exactly the same across different nodes.
"""
function contract_distr_fit(
    mpoA::TensorTrain{ValueType,4},
    mpoB::TensorTrain{ValueType,4};
    nsweeps::Int=2,
    initial_guess::Union{Nothing,TensorTrain{ValueType,4}}=nothing,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    noderanges::Union{Nothing,Vector{UnitRange{Int}}}=nothing,
    tolerance::Float64=1e-12,
    method::Symbol=:SVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    synchedmpo::Bool=false,
    kwargs...
) where {ValueType}
    if length(mpoA) != length(mpoB)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    if !synchedmpo
        synchronize_tt!(mpoA)
        synchronize_tt!(mpoB)
        synchronize_tt!(initial_guess)
    end

    A = deepcopy(mpoA.sitetensors)
    B = deepcopy(mpoB.sitetensors)

    n = length(A)

    if MPI.Initialized()
        if subcomm != nothing
            comm = subcomm
        else
            comm = MPI.COMM_WORLD
        end
        mpirank = MPI.Comm_rank(comm)
        juliarank = mpirank + 1
        nprocs = MPI.Comm_size(comm)
        if noderanges == nothing
            if n < 4
                println("Warning! The TT is too small to be parallelized.")
                return contract_fit(A, B; nsweeps, initial_guess, tolerance, method, maxbonddim, kwargs...)
            end
            if n < 6
                if nprocs > 2
                    println("Warning! The TT is too small to be parallelized with more than 2 nodes. Some nodes will not be used.")
                end
                nprocs = 2
                noderanges = [1:2,3:n]
            elseif n == 6
                if nprocs == 2
                    noderanges = [1:3,4:6]
                elseif nprocs == 3
                    noderanges = [1:2,3:4,5:6]
                else
                    println("Warning! The TT is too small to be parallelized with more than 3 nodes. Some nodes will not be used.")
                    nprocs = 3
                    noderanges = [1:2,3:4,5:6]
                end
            else 
                extra1 = 3 # "Hyperparameter"
                extraend = 3
                if nprocs >= div(n - extra1 - extraend, 2) # It's just one update per node.
                    println("Warning! A TT of lenght L can use be parallelized with up to (L-$(extra1 + extraend))/2 nodes. Some nodes will not be used.")
                    # Each node has 2 cores. Except first and last who have 2+extra1 and 2+extraend+n%2
                    if n % 2 == 0
                        noderanges = vcat([1:extra1+1],[extra1+i+1:extra1+i+2 for i in 1:2:n-extra1-extraend-2],[n-extraend:n])
                    else
                        noderanges = vcat([1:extra1+1],[extra1+i+1:extra1+i+2 for i in 1:2:n-extra1-extraend-3],[n-1-extraend:n])
                    end
                    for _ in div(n - extra1 - extraend, 2)+1:nprocs
                        push!(noderanges, 1:-1)
                    end
                    noderanges[1] += extra1
                    noderanges[end] += extraend
                    nprocs = div(n - extra1 - extraend, 2)
                else
                    number_of_sites, rem = divrem(n-extra1-extraend, nprocs)
                    sites = [number_of_sites for i in 1:nprocs]
                    sites[1] += extra1
                    sites[end] += extraend
                    # Distribute the remainder as evenly as possible
                    for i in 1:rem
                        if i % 2 == 1
                            sites[div(i,2)+2] += 1
                        else
                            sites[end-div(i,2)] += 1
                        end
                    end
                    noderanges = [sum(sites[1:i-1])+1:sum(sites[1:i]) for i in 1:nprocs]
                end
            end
        end
    else
        println("Warning! Distributed strategy has been chosen, but MPI is not initialized, please use TCI.initializempi() before contract() and use TCI.finalizempi() afterwards")
        return contract_fit(A, B; nsweeps, initial_guess, tolerance, method, maxbonddim, kwargs...)
    end

    if nprocs == 1
        println("Warning! Distributed strategy has been chosen, but only one process is running")
        return contract_fit(A, B; nsweeps, initial_guess, tolerance, method, maxbonddim, kwargs...)
    end

    if !isnothing(initial_guess)
        X = deepcopy(initial_guess.sitetensors)
    else
        X = [ones(ValueType, 1, size(A[i])[2], size(B[i])[3], 1) for i in 1:n]
    end

    Ls = Vector{Array{ValueType, 3}}(undef, n)
    Rs = Vector{Array{ValueType, 3}}(undef, n)

    first = noderanges[juliarank][1]
    last = noderanges[juliarank][end]

    #Ls are left enviroments and Rs are right environments
    if first != 1
        Ls[first-1] = ones(ValueType, size(A[first])[1], size(B[first])[1], size(X[first])[1])
    end
    if last != n
        Rs[last+1] = ones(ValueType, size(A[last])[end], size(B[last])[end], size(X[last])[end])
    end

    
    if juliarank % 2 == 1 # Precompute right environment if going forward
        centercanonicalize!(A, first)
        centercanonicalize!(B, first)
        centercanonicalize!(X, first)
        for i in last:-1:first+2
            if i == n
                Rs[i] = rightenvironment(A, B, X, i)
            else
                Rs[i] = rightenvironment(A, B, X, i; R=Rs[i+1], Ri=i)
            end
        end
    else # Precompute left environment if going backward
        centercanonicalize!(A, last-1)
        centercanonicalize!(B, last-1)
        centercanonicalize!(X, last-1)
        for i in first:last-2
            if i == 1
                Ls[i] = leftenvironment(A, B, X, i)
            else
                Ls[i] = leftenvironment(A, B, X, i; L=Ls[i-1], Li=i)
            end
        end
    end

    for sweep in 1:nsweeps
        max_diff = 0.0
        if sweep % 2 == juliarank % 2
            for i in first:last-1
                centercanonicalize!(A, i)
                centercanonicalize!(B, i)
                if i == 1
                    diff, _, _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:forward, R=Rs[i+2], Ri=i+1)
                elseif i == 2
                    diff, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:forward, R=Rs[i+2], Ri=i+1)
                elseif i == first
                    diff, _, _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:forward, L=Ls[i-1], Li=i, R=Rs[i+2], Ri=i+1)
                elseif i < n-1
                    diff, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:forward, L=Ls[i-2], Li=i-1, R=Rs[i+2], Ri=i+1)
                else
                    diff, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:forward, L=Ls[i-2], Li=i-1)
                end
                max_diff = max(max_diff, diff)
            end
        else
            for i in last-1:-1:first
                centercanonicalize!(A, i)
                centercanonicalize!(B, i)
                if i == n-1
                    diff, _, _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:backward, L=Ls[i-1], Li=i)
                elseif i == n-2
                    diff, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:backward, L=Ls[i-1], Li=i)
                elseif i == last-1
                    diff, _, _ = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:backward, L=Ls[i-1], Li=i, R=Rs[i+2], Ri=i+1)
                elseif i > 1
                    diff, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:backward, L=Ls[i-1], Li=i, R=Rs[i+3], Ri=i+2)
                else
                    diff, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance, maxbonddim, direction=:backward, R=Rs[i+3], Ri=i+2)
                end
                max_diff = max(max_diff, diff)
            end
        end

        if MPI.Initialized()
            if juliarank % 2 == sweep % 2 # Left
                if juliarank != nprocs
                    centercanonicalize!(A, last)
                    centercanonicalize!(B, last)

                    Ls[last-1] = leftenvironment(A, B, X, last-1; L=Ls[last-2], Li=last-1)

                    sizes = Vector{Int}(undef, 3)
                    reqs = MPI.Isend(collect(size(Ls[last-1])), comm; dest=mpirank+1, tag=juliarank)
                    reqr = MPI.Irecv!(sizes, comm; source=mpirank+1, tag=juliarank+1)

                    MPI.Waitall([reqs, reqr])

                    Rs[last+2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                    
                    reqs = MPI.Isend(Ls[last-1], comm; dest=mpirank+1, tag=juliarank)
                    reqr = MPI.Irecv!(Rs[last+2], comm; source=mpirank+1, tag=juliarank+1)

                    MPI.Waitall([reqs, reqr])
                    
                    diff, _, _ = updatecore!(A, B, X, last; method, tolerance, maxbonddim, direction=:backward, L=Ls[last-1], Li=last, R=Rs[last+2], Ri=last+1)
                    
                    Rs[last+1] = rightenvironment(A, B, X, last+1; R=Rs[last+2], Ri=last+1)
                end
            else # Right
                if juliarank != 1
                    centercanonicalize!(A, first)
                    centercanonicalize!(B, first)

                    Rs[first+1] = rightenvironment(A, B, X, first+1; R=Rs[first+2], Ri=first+1)

                    sizes = Vector{Int}(undef, 3)
                    reqs = MPI.Isend(collect(size(Rs[first+1])), comm; dest=mpirank-1, tag=juliarank)
                    reqr = MPI.Irecv!(sizes, comm; source=mpirank-1, tag=juliarank-1)
                    
                    MPI.Waitall([reqs, reqr])

                    Ls[first-2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                    reqs = MPI.Isend(Rs[first+1], comm; dest=mpirank-1, tag=juliarank)
                    reqr = MPI.Irecv!(Ls[first-2], comm; source=mpirank-1, tag=juliarank-1)
                    
                    MPI.Waitall([reqs, reqr])
                    
                    diff, _, _ = updatecore!(A, B, X, first-1; method, tolerance, maxbonddim, direction=:forward, L=Ls[first-2], Li=first-1, R=Rs[first+1], Ri=first) #center on first
                    
                    Ls[first-1] = leftenvironment(A, B, X, first-1; L=Ls[first-2], Li=first-1)
                end
            end
        end

        max_diff = max(max_diff, diff) # Check last update on edge
        converged = max_diff < tolerance
        global_converged = MPI.Allreduce(converged, MPI.LAND, comm)
        if global_converged
            break
        end
    end

    # Merge togheter all the sub tensor trains
    if juliarank == 1
        for j in 2:nprocs
            centercanonicalize!(A, noderanges[j-1][end])
            centercanonicalize!(B, noderanges[j-1][end])
            centercanonicalize!(X, noderanges[j-1][end])
            X[noderanges[j]] = MPI.recv(comm; source=j-1, tag=2*j)
            if j != nprocs
                R = MPI.recv(comm; source=j-1, tag=2*j+1)
            else
                R = ones(ValueType, 1, 1, 1)
            end
            diff, _, _ = updatecore!(A, B, X, noderanges[j-1][end]; method, tolerance, maxbonddim, direction=:forward, R, Ri=noderanges[j][end])
        end
    else
        centercanonicalize!(X, first)
        MPI.send(X[noderanges[juliarank]], comm; dest=0, tag=2*juliarank)
        if juliarank != nprocs
            MPI.send(Rs[noderanges[juliarank][end]+1], comm; dest=0, tag=2*juliarank+1)
        end
    end

    nprocs = MPI.Comm_size(comm) # In case not all processes where used to compute

    # Redistribute the tensor train among the processes.
    if juliarank == 1
        for j in 2:nprocs
            MPI.send(X, comm; dest=j-1, tag=1)
        end
    else
        X = MPI.recv(comm; source=0, tag=1)
    end

    return TensorTrain{ValueType,4}(X)
end

"""
    function contract(
        A::TensorTrain{V1,4},
        B::TensorTrain{V2,4};
        algorithm::Symbol=:TCI,
        tolerance::Float64=1e-12,
        maxbonddim::Int=typemax(Int),
        f::Union{Nothing,Function}=nothing,
        subcomm::Union{Nothing, MPI.Comm}=nothing,
        kwargs...
    ) where {V1,V2}

Contract two tensor trains `A` and `B`.

Currently, two implementations are available:
 1. `algorithm=:TCI` constructs a new TCI that fits the contraction of `A` and `B`.
 2. `algorithm=:naive` uses a naive tensor contraction and subsequent SVD recompression of the tensor train.
 3. `algorithm=:zipup` uses a naive tensor contraction with on-the-fly LU decomposition.
 4. `algorithm=:distrzipup` uses a naive tensor contraction with on-the-fly LU decomposition, distributed on multiple nodes.
 5. `algorithm=:fit` uses a variational approach to minimize the difference between `A`*`B` and the solution.
 6. `algorithm=:distrfit` uses a variational approach to minimize the difference between `A`*`B` and the solution, distributed on multiple nodes.


Arguments:
- `A` and `B` are the tensor trains to be contracted.
- `algorithm` chooses the algorithm used to evaluate the contraction.
- `tolerance` is the tolerance of the TCI or SVD recompression.
- `maxbonddim` sets the maximum bond dimension of the resulting tensor train.
- `f` is a function to be applied elementwise to the result. This option is only available with `algorithm=:TCI`.
- `method` chooses the method used for the factorization in the `algorithm=:zipup` case (`:SVD` or `:LU`).
- `subcomm` is an optional MPI communicator for distributed algorithms. If not provided, the default communicator is used.
- `kwargs...` are forwarded to [`crossinterpolate2`](@ref) if `algorithm=:TCI` or to [`contract_fit`](@ref) if `algorithm=:fit` or `algorithm=:distrfit`.
"""
function contract(
    A::TensorTrain{V1,4},
    B::TensorTrain{V2,4};
    algorithm::Symbol=:TCI,
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    method::Symbol=:SVD,
    f::Union{Nothing,Function}=nothing,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
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
        return contract_zipup(A_, B_; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    elseif algorithm === :distrzipup
        if f !== nothing
            error("Zipup contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        return contract_distr_zipup(A_, B_; tolerance=tolerance, maxbonddim=maxbonddim, method=method, subcomm=subcomm, kwargs...)
    elseif algorithm === :fit
        if f !== nothing
            error("Fit contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        return contract_fit(A_, B_; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    elseif algorithm === :distrfit
        if f !== nothing
            error("Fit contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        return contract_distr_fit(A_, B_; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
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
