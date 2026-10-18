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

# Without useless allocations
#=
function _contract(
    a::AbstractArray,
    b::AbstractArray,
    idx_a::NTuple{n1,Int},
    idx_b::NTuple{n2,Int}
) where {n1,n2}
    N1 = ndims(a); N2 = ndims(b)
    length(idx_a) == length(idx_b) || error("length(idx_a) != length(idx_b)")
    length(unique(idx_a)) == length(idx_a) || error("idx_a contains duplicates")
    length(unique(idx_b)) == length(idx_b) || error("idx_b contains duplicates")
    all(1 .<= idx .<= N1 for idx in idx_a) || error("idx_a out of range")
    all(1 .<= idx .<= N2 for idx in idx_b) || error("idx_b out of range")

    rest_idx_a = Tuple(setdiff(1:N1, idx_a))
    rest_idx_b = Tuple(setdiff(1:N2, idx_b))

    # create lazy permuted-wrappers (no copy)
    Aperm = PermutedDimsArray(a, (rest_idx_a..., idx_a...))
    Bperm = PermutedDimsArray(b, (idx_b..., rest_idx_b...))

    # sizes
    sizes_Aperm = size(Aperm)
    sizes_Bperm = size(Bperm)

    M = prod(sizes_Aperm[1:length(rest_idx_a)])           # rows of amat
    K = prod(sizes_Aperm[length(rest_idx_a)+1:end])       # contracted dim
    # For Bperm, first dims are contracted dims, rest are rest_idx_b
    N = prod(sizes_Bperm[length(idx_b)+1:end])            # cols of bmat

    # reshape wrappers into 2-D views (these are views/wrappers whenever strides permit)
    amat = reshape(Aperm, M, K)   # M×K view
    bmat = reshape(Bperm, K, N)   # K×N view

    # preallocate output matrix and perform in-place multiply
    Cmat = similar(amat, promote_type(eltype(a), eltype(b)), M, N)
    mul!(Cmat, amat, bmat)   # will call BLAS.gemm when strides are suitable

    # reshape back to the expected tensor shape
    return reshape(Cmat, _getindex(size(a), rest_idx_a)..., _getindex(size(b), rest_idx_b)...)
end
=#
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

function svd_project_right(A::Array{T,4}, max_r::Int; p::Int=2) where {T}
    A_ = reshape(A, prod(size(A)[1:3]), size(A)[4])
    _, A_proj, _, disc = _factorize(A_, :SVD; maxbonddim=max_r+p, tolerance=0.0, leftorthogonal=false)
    # println("Discarded proj_right: $disc")

    return A_proj'
end

function svd_project_left(A::Array{T,4}, max_l::Int; p::Int=2) where {T}
    A_ = reshape(A, size(A)[1], prod(size(A)[2:4]))
    A_proj, _, _, disc = _factorize(A_, :SVD, maxbonddim=max_l+p, tolerance=0.0, leftorthogonal=true)
    # println("Discarded proj_left: $disc")

    return A_proj'
end

function random_project_right(A::Array{T,4}, max_r::Int; p::Int=2) where {T}
    A_ = reshape(A, prod(size(A)[1:3]), size(A)[4])
    A_proj = Matrix(qr(A_'*randn(size(A_)[1], max_r+p)).Q)

    # println("From $(size(A_proj)[1]) to $(size(A_proj)[2])")

    return A_proj
end

function random_project_left(A::Array{T,4}, max_l::Int; p::Int=2) where {T}
    A_ = reshape(A, size(A)[1], prod(size(A)[2:4]))
    A_proj = Matrix(qr(A_*randn(size(A_)[2], max_l+p)).Q)'

    # println("From $(size(A_proj)[2]) to $(size(A_proj)[1])")

    return A_proj
end

# Compute full left environment
function leftenvironment(
    A::Vector{Array{T,4}},
    B::Vector{Array{T,4}},
    X::Vector{Array{T,4}},
    i::Int;
    # L::Array{T, 3} = ones(T, 1, 1, 1),
    L::Union{Nothing, Array{T, 3}} = nothing,
    Li::Int = 1,
    random_env::Bool = false
)::Array{T, 3} where {T}
    L === nothing && (L = ones(T, 1, 1, 1))
    # println("Left environment ABC $i, init: $(size(L))")
    for ell in Li:i
        A_ = B_ = nothing
        if random_env
            A_proj_left = random_project_left(A[ell], Int(ceil(sqrt(maximum(linkdims(A))))); p=0)
            B_proj_left = random_project_left(B[ell], Int(ceil(sqrt(maximum(linkdims(B))))); p=0)
            
            L = _contract(L, A_proj_left', (1,), (1,))
            L = _contract(L, B_proj_left', (1,), (1,))
            L = permutedims(L, (2,3,1,))

            A_ = _contract(A_proj_left, A[ell], (2,), (1,))
            B_ = _contract(B_proj_left, B[ell], (2,), (1,))

            # A_reconstructed = _contract(A_, A_proj_right', (4,), (1,))
            # B_reconstructed = _contract(B_, B_proj_right', (4,), (1,))

            # println("checkorthogonality: $(checkorthogonality(A)), $(checkorthogonality(B)), $(checkorthogonality(X))")
            # println("errors reconstructions at $i in A,B,C: $(maximum(abs.(A_reconstructed - A[ell]))), $(maximum(abs.(B_reconstructed - B[ell])))")
        else
            A_ = A[ell]
            B_ = B[ell]
        end

        # time_l = time_ns()
        L = _contract(L, A_, (1,), (1,))
        L = _contract(L, B_, (1,4,), (1,2,))
        L = _contract(L, conj(X[ell]), (1,2,4,), (1,2,3,))
        # time_l = (time_ns() - time_l)*1e-9
        # println("ABX random=$random_env in $time_l at L[$ell]: A=$(size(A_)), B=$(size(B_)), X=$(size(X[ell]))")
    end
    if i!=0
        # println("Left environment ABC $i, fine: $(size(L))=?=$(size(A[i])[4]), $(size(B[i])[4]), $(size(X[i])[4])")
    else
        # println("Left environment ABC $i fine: $(size(L))=?=(1,1,1)")
    end
    return L
end

# Compute left environment
function leftenvironment(
    Psia::Vector{Array{T,4}},
    Va::Vector{Matrix{T}},
    Psib::Vector{Array{T,4}},
    Vb::Vector{Matrix{T}},
    Psic::Vector{Array{T,4}},
    Vc::Vector{Matrix{T}},
    i::Int;
    L::Union{Nothing, Array{T, 3}} = nothing,
    Li::Int = 1,
    random_env::Bool = false
    )::Array{T, 3} where {T}
    L === nothing && (L = ones(T, 1, 1, 1))
    # println("Left environment PSI $i, init: $(size(L))")
    for ell in Li:i
        Psia_ = Psib_ = Va_ = Vb_ = nothing
        if random_env
            A_proj_left = random_project_left(Psia[ell], Int(ceil(sqrt(maximum(linkdims(Psia))))); p=0)
            B_proj_left = random_project_left(Psib[ell], Int(ceil(sqrt(maximum(linkdims(Psib))))); p=0)
            
            L = _contract(L, A_proj_left', (1,), (1,))
            L = _contract(L, B_proj_left', (1,), (1,))
            L = permutedims(L, (2,3,1,))

            Psia_ = _contract(A_proj_left, Psia[ell], (2,), (1,))
            Psib_ = _contract(B_proj_left, Psib[ell], (2,), (1,))

            Psia_ = _contract(Psia_, Va[ell], (4,), (1,))
            Psib_ = _contract(Psib_, Vb[ell], (4,), (1,))

            # A_reconstructed = _contract(Psia_, A_proj_right', (4,), (1,))
            # B_reconstructed = _contract(Psib_, B_proj_right', (4,), (1,))

            # println("checkorthogonality: $(checkorthogonality(A)), $(checkorthogonality(B)), $(checkorthogonality(X))")
            # println("errors reconstructions at $i in Psi: $(maximum(abs.(A_reconstructed - Psia[ell]))), $(maximum(abs.(B_reconstructed - Psib[ell])))")
        else
            Psia_ = _contract(Psia[ell], Va[ell], (4,), (1,))
            Psib_ = _contract(Psib[ell], Vb[ell], (4,), (1,))

        end
        Psic_ = _contract(Psic[ell], Vc[ell], (4,), (1,))
        
        # time_l = time_ns()
        L = _contract(L, Psia_, (1,), (1,))
        L = _contract(L, Psib_, (1,4,), (1,2,))
        L = _contract(L, conj(Psic_), (1,2,4,), (1,2,3,))
        # time_l = (time_ns() - time_l)*1e-9
        # println("INV random=$random_env in $time_l at L[$ell]: Psia=$(size(Psia_)), Psib=$(size(Psib_)), Psic=$(size(Psic_))")
    end
    if i!=0
        # println("Left environment PSI $i, fine: $(size(L))=?=$(size(Psia[i])[4]), $(size(Psib[i])[4]), $(size(Psic[i])[4])")
    else
        # println("Left environment PSI $i fine: $(size(L))=?=(1,1,1)")
    end
    return L
end

# Compute full right environment
function rightenvironment(
    A::Vector{Array{T,4}},
    B::Vector{Array{T,4}},
    X::Vector{Array{T,4}},
    i::Int;
    R::Union{Nothing, Array{T, 3}} = nothing,
    Ri::Int = length(A),
    random_env::Bool = false
    )::Array{T, 3} where {T}
    R === nothing && (R = ones(T, 1, 1, 1))
    # println("Right environment ABC $i, init: $(size(R))")
    for ell in Ri:-1:i
        A_ = B_ = nothing
        if random_env
            A_proj_right = random_project_right(A[ell], Int(ceil(sqrt(maximum(linkdims(A))))); p=0)
            B_proj_right = random_project_right(B[ell], Int(ceil(sqrt(maximum(linkdims(B))))); p=0)

            A_ = _contract(A[ell], A_proj_right, (4,), (1,))
            B_ = _contract(B[ell], B_proj_right, (4,), (1,))

            # A_reconstructed = _contract(A_proj_left', A_, (2,), (1,))
            # B_reconstructed = _contract(B_proj_left', B_, (2,), (1,))

            # println("checkorthogonality: $(checkorthogonality(A)), $(checkorthogonality(B)), $(checkorthogonality(X))")
            # println("errors reconstructions at $i in A,B,C: $(maximum(abs.(A_reconstructed - A[ell]))), $(maximum(abs.(B_reconstructed - B[ell])))")

            R = _contract(B_proj_right', R, (2,), (2,))
            R = _contract(A_proj_right', R, (2,), (2,))
        else
            A_ = A[ell]
            B_ = B[ell]
        end

        # println("ABC with random=$random_env, dims at R[$ell]: A=$(size(A_)), B=$(size(B_)), X=$(size(X[ell]))")

        # time_r = time_ns()
        # println("Contracting A=$(size(A_)), B=$(size(B_)), X=$(size(X[ell])) with R=$(size(R))")
        # println("Contracting C[$ell]=$(size(X[ell])) with R=$(size(R))")
        R = _contract(conj(X[ell]), R, (4,), (3,))
        R = _contract(B_, R, (3,4,), (3,5,)) # TODO controlla anche PSI
        R =  _contract(A_, R, (2,3,4), (4,2,5,))
        # time_r = (time_ns() - time_r)*1e-9
        # println("ABX random=$random_env in $time_r at R[$ell]: A=$(size(A_)), B=$(size(B_)), X=$(size(X[ell]))")

    end
    if i!=length(A)+1
        # println("Right environment ABC $i, fine: $(size(R))=?=$(size(A[i])[1]), $(size(B[i])[1]), $(size(X[i])[1])")
    else
        # println("Right environment ABC $i fine: $(size(R))=?=(1,1,1)")
    end
    return R
end

# Compute right environment
function rightenvironment(
    Psia::Vector{Array{T,4}},
    Va::Vector{Matrix{T}},
    Psib::Vector{Array{T,4}},
    Vb::Vector{Matrix{T}},
    Psic::Vector{Array{T,4}},
    Vc::Vector{Matrix{T}},
    i::Int;
    R::Union{Nothing, Array{T, 3}} = nothing,
    Ri::Int = length(Psia),
    random_env::Bool = false
    )::Array{T, 3} where {T}
    R === nothing && (R = ones(T, 1, 1, 1))
    # println("R init max: $(maximum(abs.(R)))")
    for ell in Ri:-1:i
        Psia_ = Psib_ = Va_ = Vb_ = nothing
        if random_env
            A_proj_right = random_project_right(Psia[ell], Int(ceil(sqrt(maximum(linkdims(Psia))))); p=0)
            B_proj_right = random_project_right(Psib[ell], Int(ceil(sqrt(maximum(linkdims(Psib))))); p=0)

            Psia_ = _contract(Psia[ell], A_proj_right, (4,), (1,))
            Psib_ = _contract(Psib[ell], A_proj_right, (4,), (1,))

            Psia_ = _contract(Va[ell-1], Psia_, (2,), (1,))
            Psib_ = _contract(Vb[ell-1], Psib_, (2,), (1,))

            R = _contract(B_proj_right', R, (2,), (2,))
            R = _contract(A_proj_right', R, (2,), (2,))
            # A_reconstructed = _contract(A_proj_left', A_, (2,), (1,))
            # B_reconstructed = _contract(B_proj_left', B_, (2,), (1,))

            # println("checkorthogonality: $(checkorthogonality(A)), $(checkorthogonality(B)), $(checkorthogonality(X))")
            # println("errors reconstructions at $i in Psi: $(maximum(abs.(A_reconstructed - Psia[ell]))), $(maximum(abs.(B_reconstructed - Psib[ell])))")
        else
            Psia_ = _contract(Va[ell-1], Psia[ell], (2,), (1,))
            Psib_ = _contract(Vb[ell-1], Psib[ell], (2,), (1,))
        end
        Psic_ = _contract(Vc[ell-1], Psic[ell], (2,), (1,))

        # println("Psia_ max: $(maximum(abs.(Psia_)))")
        # println("Psib_ max: $(maximum(abs.(Psib_)))")
        # println("Psic_ max: $(maximum(abs.(Psic_)))")

        # println("Random=$random_env, dims at R[$ell]: Psia=$(size(Psia_)), Psib=$(size(Psib_)), Psic=$(size(Psic_))")


        # time_r = time_ns()
        R = _contract(conj(Psic_), R, (4,), (3,))
        R = _contract(Psib_, R, (3,4,), (3,5,))
        R =  _contract(Psia_, R, (2,3,4), (4,2,5,))
        # time_r = (time_ns() - time_r)*1e-9
        # println("INV random=$random_env in $time_r at R[$ell]: Psia=$(size(Psia_)), Psib=$(size(Psib_)), Psic=$(size(Psic_))")


        # println("R at step $ell max: $(maximum(abs.(R)))")
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
)::TensorTrain{ValueType,4} where {ValueType}
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
)::TensorTrain{ValueType,4} where {ValueType}
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
)::TensorTrain{ValueType,4} where {ValueType}
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
            U, Vt, newbonddiml, _ = _factorize(
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
            U, Vt, newbonddimr, _ = _factorize(U_3_2, :SVD; tolerance, maxbonddim)
            Us[i] = reshape(L, dimsA[1], dimsB[1], newbonddiml)
            finalsitetensors[n] = reshape(U, newbonddiml, dimsA[2], dimsB[3], newbonddimr)
            Vts[i] = reshape(Vt, newbonddimr, dimsA[4], dimsB[4])
        end
    end
    
    # Exchange Vt to right
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
function updatecore!(A::Vector{Array{T,4}}, B::Vector{Array{T,4}}, X::Vector{Array{T,4}}, i::Int;
    method::Symbol=:SVD, tolerance::Float64=1e-8, maxbonddim::Int=typemax(Int), direction::Symbol=:forward, random_update::Bool=false, random_env::Bool=false,
    R::Union{Nothing, Array{T,3}}=nothing, Ri::Int=length(A),
    L::Union{Nothing, Array{T,3}}=nothing, Li::Int=1
    )::Tuple{Float64, Array{T,3}, Array{T,3}} where {T}
    L = leftenvironment(A, B, X, i-1; L, Li, random_env)
    R = rightenvironment(A, B, X, i+2; R, Ri, random_env)
    
    if !random_update
        Le = _contract(_contract(L, A[i], (1,), (1,)), B[i], (1, 4), (1, 2))
        Re = _contract(B[i+1], _contract(A[i+1], R, (4,), (1,)), (2, 4), (3, 4))
    else
        A_proj = random_project_right(A[i], Int(ceil(sqrt(maximum(linkdims(A))))); p=0)
        B_proj = random_project_right(B[i], Int(ceil(sqrt(maximum(linkdims(A))))); p=0)
        Ai = _contract(A[i], A_proj, (4,), (1,))
        Bi = _contract(B[i], B_proj, (4,), (1,))

        # A_reconstructed = _contract(Ai, A_proj', (4,), (1,))
        # B_reconstructed = _contract(Bi, B_proj', (4,), (1,))

        # println("Error proj A up: ", maximum(abs, A[i] - A_reconstructed))
        # println("Error proj B up: ", maximum(abs, B[i] - B_reconstructed))

        Aip1 = _contract(A_proj, A[i+1], (1,), (1,))
        Bip1 = _contract(B_proj, B[i+1], (1,), (1,))

        Le = _contract(_contract(L, Ai, (1,), (1,)), Bi, (1, 4), (1, 2))
        Re = _contract(Bip1, _contract(Aip1, R, (4,), (1,)), (2, 4), (3, 4))
    end


    # time_ce = time_ns()
    Ce = _contract(Le, Re, (3, 5), (3, 1))
    Ce = permutedims(Ce, (1, 2, 3, 5, 4, 6))
    # time_ce = (time_ns() - time_ce)*1e-9
    # println("ABX random=$random_update in $time_ce at site $i, (Le=", size(Le), ", Re=", size(Re), ")")
    
    left, right, newbonddim, disc = _factorize(
        reshape(Ce, prod(size(Ce)[1:3]), prod(size(Ce)[4:6])),
        method; tolerance, maxbonddim, leftorthogonal=(direction == :forward ? true : false)
        )
        
    X[i] = reshape(left, :, size(X[i])[2:3]..., newbonddim)
    X[i+1] = reshape(right, newbonddim, size(X[i+1])[2:3]..., :)
    return disc, L, R
end

function updatecore!(Psia::Vector{Array{T,4}}, Va::Vector{Matrix{T}}, Psib::Vector{Array{T,4}}, Vb::Vector{Matrix{T}}, Psic::Vector{Array{T,4}}, Vc::Vector{Matrix{T}}, i::Int;
    method::Symbol=:SVD, tolerance::Float64=1e-8, maxbonddim::Int=typemax(Int), direction::Symbol=:forward, random_update::Bool=false, random_env::Bool=false,
    R::Union{Nothing, Array{T,3}}=nothing, Ri::Int=length(Psia),
    L::Union{Nothing, Array{T,3}}=nothing, Li::Int=1
    )::Tuple{Float64, Array{T,3}, Array{T,3}} where {T}

    L = leftenvironment(Psia, Va, Psib, Vb, Psic, Vc, i-1; L, Li, random_env)
    R = rightenvironment(Psia, Va, Psib, Vb, Psic, Vc, i+2; R, Ri, random_env)

    if !random_update
        Ai = _contract(Psia[i], Va[i], (4,), (1,))
        Bi = _contract(Psib[i], Vb[i], (4,), (1,))

        Le = _contract(_contract(L, Ai, (1,), (1,)), Bi, (1, 4), (1, 2))
        Re = _contract(Psib[i+1], _contract(Psia[i+1], R, (4,), (1,)), (2, 4), (3, 4))
    else
        Aip1 = _contract(Va[i], Psia[i+1], (2,), (1,))
        Bip1 = _contract(Vb[i], Psib[i+1], (2,), (1,))
        A_proj = random_project_right(Psia[i], Int(ceil(sqrt(maximum(linkdims(Psia))))); p=0)
        B_proj = random_project_right(Psib[i], Int(ceil(sqrt(maximum(linkdims(Psib))))); p=0)

        Ai_projected = _contract(Psia[i], A_proj, (4,), (1,))
        Bi_projected = _contract(Psib[i], B_proj, (4,), (1,))
        # A_reconstructed = _contract(Ai_projected, A_proj', (4,), (1,))
        # B_reconstructed = _contract(Bi_projected, B_proj', (4,), (1,))

        # println("Error proj A up: ", maximum(abs, Psia[i] - A_reconstructed))
        # println("Error proj B up: ", maximum(abs, Psib[i] - B_reconstructed))

        Aip1_projected = _contract(A_proj', Aip1, (2,), (1,))
        Bip1_projected = _contract(B_proj', Bip1, (2,), (1,))

        Le = _contract(_contract(L, Ai_projected, (1,), (1,)), Bi_projected, (1, 4), (1, 2))
        Re = _contract(Bip1_projected, _contract(Aip1_projected, R, (4,), (1,)), (2, 4), (3, 4))
    end

    time_ce = time_ns()
    Ce = _contract(Le, Re, (3, 5), (3, 1))
    Ce = permutedims(Ce, (1, 2, 3, 5, 4, 6))
    time_ce = (time_ns() - time_ce)*1e-9
    # println("INV random=$random_update at site $i in $time_ce, (Le=", size(Le), ", Re=", size(Re), ")")

    left, diamond, right, newbonddim, disc = _factorize(
        reshape(Ce, prod(size(Ce)[1:3]), prod(size(Ce)[4:6])),
        method; tolerance, maxbonddim, diamond=true
    )
    
    Psic[i] = reshape(left*Diagonal(diamond), :, size(Psic[i])[2:3]..., newbonddim)
    Psic[i+1] = reshape(Diagonal(diamond)*right, newbonddim, size(Psic[i+1])[2:3]..., :)
    Vc[i] = Diagonal(diamond.^-1)
    
    return disc, L, R
end

"""
    function contract_fit(A::TensorTrain{ValueType,4}, B::TensorTrain{ValueType,4})

Conctractos tensor trains A and B using the fit algorithm.

# Keyword Arguments

- `nsweeps::Int`: Number of sweeps to perform during the algorithm.
- `initial_guess`: Optional initial guess for the tensor train A*B. If not provided, a tensor train of rank one is used. This must have coherent bond dimension (i.e. start and finish with less than [d,d^2,d^3,...,d^3,d^2,d]).
- `tolerance::Float64`: Convergence tolerance for the iterative algorithm.
- `method::Symbol`: Algorithm or method to use for the computation :SVD, :RSVD, :LU, :CI.
- `maxbonddim::Int`: Maximum bond dimension allowed during the decomposition.
"""
function contract_fit(mpoA::TensorTrain{ValueType,4},
    mpoB::TensorTrain{ValueType,4};
    nsweeps::Int=2,
    initial_guess::Union{Nothing,TensorTrain{ValueType,4}}=nothing,
    debug::Union{Nothing,TensorTrain{ValueType,4}}=nothing,
    tolerance::Float64=1e-12,
    method::Symbol=:SVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    random_update::Bool=false,
    random_env::Bool=false,
    stable::Bool=true,
    abgauge::Symbol=:percopy, # serial: :percopy = original (re-canonicalise A,B each update), :twosided = precomputed
    kwargs...)::TensorTrain{ValueType,4} where {ValueType}
    if length(mpoA) != length(mpoB)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    n = length(mpoA)

    A = Vector{Array{ValueType, 4}}(undef, n)
    B = Vector{Array{ValueType, 4}}(undef, n)
    X = Vector{Array{ValueType, 4}}(undef, n)
    for i in 1:n
        A[i] = deepcopy(mpoA.sitetensors[i])
        B[i] = deepcopy(mpoB.sitetensors[i])
        if !isnothing(initial_guess)
            X[i] = deepcopy(initial_guess.sitetensors[i])
        else
            if size(mpoA[i])[2] == size(mpoB[i])[3]
                X[i] = deepcopy(mpoB.sitetensors[i])
            else
                X[i] = ones(ValueType, size(mpoB[i])[1], size(mpoA[i])[2], size(mpoB[i])[3], size(mpoB[i])[4])
            end
        end
    end
    
    Rs = Vector{Array{ValueType, 3}}(undef, n)
    Ls = Vector{Array{ValueType, 3}}(undef, n)
    
    # println("Pre tutto: $(checkorthogonality(A))")
    tot_time_qr = 0.
    tot_time_env = 0.
    tot_time_update = 0.

    time_qr = time_ns()
    abprepA = abgauge === :twosided ? _twosided_prep(A) : nothing
    abprepB = abgauge === :twosided ? _twosided_prep(B) : nothing
    if !random_env && stable
        _center_inputs!(A, abprepA, abgauge, 1)
        _center_inputs!(B, abprepB, abgauge, 1)
    end
    centercanonicalize!(X, 1)
    tot_time_qr += (time_ns() - time_qr)*1e-9

    # println("check A: $(checkorthogonality(A)), B: $(checkorthogonality(B)), X: $(checkorthogonality(X)) using $method, $random_update, $random_env")

    time_env = time_ns()
    # Precompute right environments
    for i in n:-1:3
        if i == n
            Rs[i] = rightenvironment(A, B, X, i; random_env)
        else
            Rs[i] = rightenvironment(A, B, X, i; R=Rs[i+1], Ri=i, random_env)
        end
    end
    tot_time_env += (time_ns() - time_env)*1e-9

    # time_pre = (time_ns() - time_pre)*1e-9
    # println("ABC time_pre=$time_pre with random=$random_env")
    # println([maximum(abs.(Rs[i])) for i in 3:n])
    
    # It doesn't matter if we repeat update in 1 or n-1, those are negligible
    # println("Error init: $(mynorm(debug-TensorTrain{ValueType,4}(X)))")
    direction = :forward
    for sweep in 1:nsweeps
        tot_disc = 0.0
        time_sweep = time_ns()

        # Update cores and store Left or Right environment
        if direction == :forward
            for i in 1:n-1
                # time_center = time_ns()

                time_qr = time_ns()
                if !random_env && stable
                    _center_inputs!(A, abprepA, abgauge, i)
                    _center_inputs!(B, abprepB, abgauge, i)
                end
                centercanonicalize!(X, i)
                tot_time_qr += (time_ns() - time_qr)*1e-9

                # time_center = (time_ns() - time_center)*1e-9

                time_update = time_ns()

                if i == 1
                    disc, _, _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, R=Rs[i+2], Ri=i+1)
                elseif i == 2
                    disc, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, R=Rs[i+2], Ri=i+1)
                elseif i < n-1
                    disc, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-2], Li=i-1, R=Rs[i+2], Ri=i+1)
                else # i == n-1
                    disc, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-2], Li=i-1)
                end
                # println("Sweep $sweep site $i: disc = $disc")

                # max_disc = max(max_disc, disc)
                tot_disc += disc
                tot_time_update += (time_ns() - time_update)*1e-9
                # println("Error site $i: $(mynorm(debug-TensorTrain{ValueType,4}(X)))")
                # println("ABC time_site=$time_update with random=$random_env")
                # println("Time site $i A,B,C,rupd=$random_update,renv=$random_env center: $time_center, update: $time_update")
            end
            direction = :backward
        elseif direction == :backward
            for i in n-1:-1:1
                # time_center = time_ns()
                
                time_qr = time_ns()
                if !random_env && stable
                    _center_inputs!(A, abprepA, abgauge, i)
                    _center_inputs!(B, abprepB, abgauge, i)
                end
                centercanonicalize!(X, i)

                tot_time_qr += (time_ns() - time_qr)*1e-9
                # time_center = (time_ns() - time_center)*1e-9

                time_update = time_ns()
                if i == n-1
                    disc, _, _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-1], Li=i)
                elseif i == n-2
                    disc, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-1], Li=i)
                elseif i > 1
                    disc, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-1], Li=i, R=Rs[i+3], Ri=i+2)
                else # i == 1
                    disc, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, R=Rs[i+3], Ri=i+2)
                end
                # println("Sweep $sweep site $i: disc = $disc")

                # max_disc = max(max_disc, disc)
                tot_disc += disc
                tot_time_update += (time_ns() - time_update)*1e-9
                # println("Error site $i: $(mynorm(debug-TensorTrain{ValueType,4}(X)))")
                # time_update = (time_ns() - time_update)*1e-9
                # println("ABC time_site=$time_update with random=$random_env")
                # println("Time site $i A,B,C,rupd=$random_update,renv=$random_env center: $time_center, update: $time_update")
                # println("Time site $i A,B,C: $time_site")
            end
            direction = :forward    
        end

        
        # time_sweep = (time_ns() - time_sweep)*1e-9
        # println("ABC time_sweep=$time_sweep with random=$random_env")
        # println("Sweep $sweep: ", [maximum(abs.(X[i])) for i in 1:n])
        # println("Sweep $sweep: ", checkorthogonality(X))
        # println("Time sweep $sweep A,B,C,rupd=$random_update,renv=$random_env: $time_sweep")

        # println("Sweep $sweep: tot_disc = $tot_disc")
        # println("Sweep $sweep: tot_disc = $tot_disc using $method, $random_update, $random_env")
        if tot_disc < tolerance
            break
        end
    end
    # println("ENV: $tot_time_env, UP: $tot_time_update, QR: $tot_time_qr for random=$random_update")
    return TensorTrain{ValueType,4}(X)
end

# TODO remove this
function mynorm(tt)
    res = permutedims(_contract(tt[1], conj(tt[1]), (2,3,), (2,3,)), (1,3,2,4,))
    for i in 2:length(tt)
        # println("Pre step $i, res is $(size(res)), tt[$i] is $(size(tt[i]))")
        res = _contract(res, permutedims(_contract(tt[i], conj(tt[i]), (2,3,), (2,3,)), (1,3,2,4,)), (3,4,), (1,2,))
    end
    if res[1][1] < 0.
        println("Error, norm^2 is $(res[1][1])")
    end
    return sqrt(abs(res[1][1]))
end

function contract_fit(Psia::Vector{Array{ValueType,4}},
    Va::Vector{Array{ValueType,2}},
    Psib::Vector{Array{ValueType,4}},
    Vb::Vector{Array{ValueType,2}};
    nsweeps::Int=2,
    Psi_init::Union{Nothing,Vector{Array{ValueType,4}}}=nothing,
    V_init::Union{Nothing,Vector{Array{ValueType,2}}}=nothing,
    tolerance::Float64=1e-12,
    method::Symbol=:RSVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    random_update::Bool=false,
    random_env::Bool=false,
    stable::Bool=true,
    debug=nothing,
    kwargs...)::TensorTrain{ValueType,4} where {ValueType}
    if length(Psia) != length(Psib)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end


    # time_init = time_ns()
    n = length(Psia)

    Psic = Vector{Array{ValueType, 4}}(undef, n)
    Vc = Vector{Array{ValueType, 2}}(undef, n)

    for i in 1:n
        if !isnothing(Psi_init)
            Psic[i] = deepcopy(Psi_init[i])
        else
            Psic[i] = deepcopy(Psia[i])
        end
    end

    for i in 1:n-1
        if !isnothing(V_init)
            Vc[i] = deepcopy(V_init[i])
        else
            Vc[i] = deepcopy(Va[i])
        end
    end
        
    Rs = Vector{Array{ValueType, 3}}(undef, n)
    Ls = Vector{Array{ValueType, 3}}(undef, n)

    # time_init = (time_ns() - time_init)*1e-9
    # println("time_init=$time_init")

    tot_time_env = 0.
    tot_time_update = 0.
    # time_pre = time_ns()

    time_env = time_ns()
    # Precompute right environments
    for i in n:-1:3
        if i == n
            Rs[i] = rightenvironment(Psia, Va, Psib, Vb, Psic, Vc, i; random_env)
        else
            Rs[i] = rightenvironment(Psia, Va, Psib, Vb, Psic, Vc, i; R=Rs[i+1], Ri=i, random_env)
        end
    end

    tot_time_env += (time_ns() - time_env)*1e-9

    # time_pre = (time_ns() - time_pre)*1e-9
    # println("time_pre=$time_pre with random=$random_env")

    # println("Time pre Psi,rupd=$random_update,renv=$random_env: $time_pre")

    # println("Rs pre tutto: $([maximum(abs.(Rs[i])) for i in 3:n])")
    # It doesn't matter if we repeat update in 1 or n-1, those are negligible

    #=
    PsiVa = [_contract(Psia[i], Va[i], (4,), (1,)) for i in 1:n-1]
    PsiVb = [_contract(Psib[i], Vb[i], (4,), (1,)) for i in 1:n-1]
    PsiVc = [_contract(Psic[i], Vc[i], (4,), (1,)) for i in 1:n-1]
    push!(PsiVa, Psia[n])
    push!(PsiVb, Psib[n])
    push!(PsiVc, Psic[n])

    VPsia = [Psia[1]]
    VPsib = [Psib[1]]
    VPsic = [Psic[1]]
    for i in 1:n-1
        push!(VPsia, _contract(Va[i], Psia[i+1], (2,), (1,)))
        push!(VPsib, _contract(Vb[i], Psib[i+1], (2,), (1,)))
        push!(VPsic, _contract(Vc[i], Psic[i+1], (2,), (1,)))
    end

    println("Post: PsiVa = $(checkorthogonality(PsiVa)), VPsia = $(checkorthogonality(VPsia))")
    println("Post: PsiVb = $(checkorthogonality(PsiVb)), VPsib = $(checkorthogonality(VPsib))")
    println("Post: PsiVc = $(checkorthogonality(PsiVc)), VPsic = $(checkorthogonality(VPsic))")
    =#
    direction = :forward
    for sweep in 1:nsweeps
        tot_disc = 0.0
        # Update cores and store Left or Right environment
        time_sweep = time_ns()
        if direction == :forward
            for i in 1:n-1
                # println("Maximum Psic: ", [maximum(abs.(Psic[i])) for i in 1:n])
                # println("Maximum Vc: ", [maximum(abs.(Vc[i])) for i in 1:n-1])
                # println("At site $i: Psia $(size(Psia[i]))-$(size(Psia[i+1])), Psib $(size(Psib[i]))-$(size(Psib[i+1])), Psic $(size(Psic[i]))-$(size(Psic[i+1]))")

                # time_site = time_ns()
                time_update = time_ns()

                if i == 1
                    disc, _, _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, R=Rs[i+2], Ri=i+1)
                elseif i == 2
                    disc, Ls[i-1], _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, R=Rs[i+2], Ri=i+1)
                elseif i < n-1
                    disc, Ls[i-1], _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-2], Li=i-1, R=Rs[i+2], Ri=i+1)
                else # i == n-1
                    disc, Ls[i-1], _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-2], Li=i-1)
                end
                # println("Sweep $sweep site $i: disc = $disc")

                # max_disc = max(max_disc, disc)
                # time_site = (time_ns() - time_site)*1e-9
                tot_time_update += (time_ns() - time_update)*1e-9
                # println("site $i time_site=$time_site with random=$random_env")
                # println("Time site $i Psi,rupd=$random_update,renv=$random_env: $time_site")
                tot_disc += disc
            #   println("Time update $i = $time_update")
                # println("At step $i direction $direction")
                # tmp = [_contract(Psic[i], Vc[i], (4,), (1,)) for i in 1:n-1]
                # push!(tmp, Psic[n])
                # println("Error: $(mynorm(debug-TensorTrain{ValueType,4}(tmp)))")
            end
            direction = :backward
        elseif direction == :backward
            for i in n-1:-1:1
                # println("Maximum Psic: ", [maximum(abs.(Psic[i])) for i in 1:n])
                # println("Maximum Vc: ", [maximum(abs.(Vc[i])) for i in 1:n-1])
                # println("At site $i: Psia $(size(Psia[i]))-$(size(Psia[i+1])), Psib $(size(Psib[i]))-$(size(Psib[i+1])), Psic $(size(Psic[i]))-$(size(Psic[i+1]))")

                time_update = time_ns()
                if i == n-1
                    disc, _, _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-1], Li=i)
                elseif i == n-2
                    disc, _, Rs[i+2] = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-1], Li=i)
                elseif i > 1
                    disc, _, Rs[i+2] = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, L=Ls[i-1], Li=i, R=Rs[i+3], Ri=i+2)
                else # i == 1
                    disc, _, Rs[i+2] = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction, random_update, random_env, R=Rs[i+3], Ri=i+2)
                end

                tot_time_update += (time_ns() - time_update)*1e-9
                # println("Sweep $sweep site $i: disc = $disc")

                # time_site = (time_ns() - time_site)*1e-9
                # println("site $i time_site=$time_site with random=$random_env")
                # println("Time site $i Psi,rupd=$random_update,renv=$random_env: $time_site")
                tot_disc += disc
                # println("Time update $i = $time_update")
                # println("At step $i direction $direction")
                # tmp = [_contract(Psic[i], Vc[i], (4,), (1,)) for i in 1:n-1]
                # push!(tmp, Psic[n])
                # println("Error: $(mynorm(debug-TensorTrain{ValueType,4}(tmp)))")

            end
            direction = :forward
        end

        time_sweep = (time_ns() - time_sweep)*1e-9
        # println("time_sweep=$time_sweep with random=$random_env")
        # println("Time sweep $sweep Psi,rupd=$random_update,renv=$random_env: $time_sweep")

        # println("Rs: $([maximum(abs.(Rs[i])) for i in 3:n])")
        # println("Ls: $([maximum(abs.(Ls[i])) for i in 1:n-2])")
        # println("Maximum Psic*Vc: ", [maximum(abs.(_contract(Psic[i], Vc[i], (4,), (1,)))) for i in 1:n-1])


        if tot_disc < tolerance
            break
        end
    end

    # println("Orthogonality Psic: ", checkorthogonality(Psic))
    # println("Maximum Psic: ", [maximum(abs.(Psic[i])) for i in 1:n])
    # println("Maximum Vc: ", [maximum(abs.(Vc[i])) for i in 1:n-1])
    # println("Maximum Psic*Vc: ", [maximum(abs.(_contract(Psic[i], Vc[i], (4,), (1,)))) for i in 1:n-1])

    time_final = time_ns()
    for i in 1:n-1
        Psic[i] = _contract(Psic[i], Vc[i], (4,), (1,)) # Dispari giusto
    end

    time_final = (time_ns() - time_final)*1e-9
    # println("time_final=$time_final")
    # println("Time final = $time_final")
    # X[end] = Psic[end]
    # println("Orthogonality X: ", checkorthogonality(X))
    # println("Maximum X: ", [maximum(abs.(X[i])) for i in 1:n])

    # println("ENV: $tot_time_env, UPDATE: $tot_time_update for random=$random_update")
    return TensorTrain{ValueType,4}(Psic)
end

function contract_distr_fit(
    Psia::Vector{Array{ValueType,4}},
    Va::Vector{Array{ValueType,2}},
    Psib::Vector{Array{ValueType,4}},
    Vb::Vector{Array{ValueType,2}};
    nsweeps::Int=8,
    Psi_init::Union{Nothing,Vector{Array{ValueType,4}}}=nothing,
    V_init::Union{Nothing,Vector{Array{ValueType,2}}}=nothing,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    noderanges::Union{Nothing,Vector{UnitRange{Int}}}=nothing,
    tolerance::Float64=1e-12,
    method::Symbol=:RSVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    random_update::Bool=false,
    random_env::Bool=false,
    synchedinput::Bool=false,
    synchedoutput::Bool=true,
    stable::Bool=true,
    kwargs...
)::TensorTrain{ValueType,4} where {ValueType}
    if length(Psia) != length(Psib)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    time_init = time_ns()
    # println("We started")
    # flush(stdout)

    # println("La tolerance usata e': $tolerance")
    if !synchedinput
        synchronize_tt!(Psia)
        synchronize_tt!(Psib)
        synchronize_tt!(Psi_init)
    end
    n = length(Psia)

    Psic = Vector{Array{ValueType, 4}}(undef, n)
    Vc = Vector{Array{ValueType, 2}}(undef, n)
    for i in 1:n
        if !isnothing(Psi_init)
            Psic[i] = deepcopy(Psi_init[i])
        else
            Psic[i] = deepcopy(Psia[i])
        end
    end
    
    for i in 1:n-1
        if !isnothing(V_init)
            Vc[i] = deepcopy(V_init[i])
        else
            Vc[i] = deepcopy(Va[i])
        end
    end
    

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
                return contract_fit(Psia, Va, Psib, Vb; nsweeps, Psi_init, V_init, tolerance, method, maxbonddim, kwargs...)
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
                if nprocs > div(n - extra1 - extraend, 2) # It's just one update per node.
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
        return contract_fit(Psia, Va, Psib, Vb; nsweeps, Psi_init, V_init, tolerance, method, maxbonddim, kwargs...)
    end

    if nprocs == 1
        println("Warning! Distributed strategy has been chosen, but only one process is running")
        return contract_fit(Psia, Va, Psib, Vb; nsweeps, Psi_init, V_init, tolerance, method, maxbonddim, kwargs...)
    end

    time_init = (time_ns() - time_init)*1e-9

    # println("Node $juliarank: time_init=$time_init s")
    if juliarank <= nprocs

        time_precompute = time_ns()

        Ls = Vector{Array{ValueType, 3}}(undef, n)
        Rs = Vector{Array{ValueType, 3}}(undef, n)

        # println("Juliarank $juliarank start maximum Psic: $([maximum(abs.(Psic[i])) for i in 1:n])")
        # println("Juliarank $juliarank start maximum Vc: $([maximum(abs.(Vc[i])) for i in 1:n-1])")

        first = noderanges[juliarank][1]
        last = noderanges[juliarank][end]

        #Ls are left enviroments and Rs are right environments
        if first != 1
            init_env = rand(ValueType, size(Psia[first])[1], size(Psib[first])[1], size(Psic[first])[1])
            Ls[first-1] = init_env ./ sqrt(sum(init_env.^2)) 
            # Ls[first-1] = ones(ValueType, size(Psia[first])[1], size(Psib[first])[1], size(Psic[first])[1]) # This shouldn't slow down but could
        end
        if last != n
            init_env = rand(ValueType, size(Psia[last])[end], size(Psib[last])[end], size(Psic[last])[end])
            Rs[last+1] = init_env ./ sqrt(sum(init_env.^2))
            # Rs[last+1] = ones(ValueType, size(Psia[last])[end], size(Psib[last])[end], size(Psic[last])[end]) # This shouldn't slow down but could
        end

        # println("Juliarank $juliarank prepre max L: $([isassigned(Ls, i) ? maximum(abs.(Ls[i])) : "not" for i in 1:n])")
        # println("Juliarank $juliarank prepre max R: $([isassigned(Rs, i) ? maximum(abs.(Rs[i])) : "not" for i in 1:n])")

        # TODO make so that you can either precompute all or only local environments
        if juliarank % 2 == 1 # Precompute right environment if going forward
            for i in last:-1:first+2
                if i == n
                    Rs[i] = rightenvironment(Psia, Va, Psib, Vb, Psic, Vc, i; random_env)
                else
                    Rs[i] = rightenvironment(Psia, Va, Psib, Vb, Psic, Vc, i; R=Rs[i+1], Ri=i, random_env)
                end
            end
        else # Precompute left environment if going backward
            for i in first:last-2 # i is never 1
                Ls[i] = leftenvironment(Psia, Va, Psib, Vb, Psic, Vc, i; L=Ls[i-1], Li=i, random_env)
            end
        end

        # println("Juliarank $juliarank postpre max L: $([isassigned(Ls, i) ? maximum(abs.(Ls[i])) : "not" for i in 1:n])")
        # println("Juliarank $juliarank postpre max R: $([isassigned(Rs, i) ? maximum(abs.(Rs[i])) : "not" for i in 1:n])")

        time_precompute = (time_ns() - time_precompute)*1e-9
        # println("Node $juliarank: time_precompute=$time_precompute s")

        for sweep in 1:nsweeps
            tot_disc = 0.0
            time_sweep = time_ns()

            if sweep % 2 == juliarank % 2 # Forward
                for i in first:last-1
                    # println("At site $i: Psia $(size(Psia[i]))-$(size(Psia[i+1])), Psib $(size(Psib[i]))-$(size(Psib[i+1])), Psic $(size(Psic[i]))-$(size(Psic[i+1]))")
                    pre_sizesi = size(Psic[i])
                    pre_sizesip1 = size(Psic[i+1])
                    time_update = time_ns()

                    if i == 1
                        disc, _, _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:forward, random_update, random_env, R=Rs[i+2], Ri=i+1)
                    elseif i == 2
                        disc, Ls[i-1], _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:forward, random_update, random_env, R=Rs[i+2], Ri=i+1)
                    elseif i == first
                        disc, _, _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:forward, random_update, random_env, L=Ls[i-1], Li=i, R=Rs[i+2], Ri=i+1)
                    elseif i < n-1
                        disc, Ls[i-1], _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:forward, random_update, random_env, L=Ls[i-2], Li=i-1, R=Rs[i+2], Ri=i+1)
                    else
                        disc, Ls[i-1], _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:forward, random_update, random_env, L=Ls[i-2], Li=i-1)
                    end

                    time_update = (time_ns() - time_update)*1e-9

                    # println("Juliarank $juliarank step $i time_update=$time_update s, from sizes $(pre_sizesi)-$(pre_sizesip1) to sizes $(size(Psic[i]))-$(size(Psic[i+1]))")

                    # max_disc = max(max_disc, disc)
                    # println("Juliarank $juliarank step $i Psic: $([maximum(abs.(Psic[i])) for i in 1:n])")
                    # println("Juliarank $juliarank step $i Vc: $([maximum(abs.(Vc[i])) for i in 1:n-1])")

                    tot_disc += disc
                end
            else
                for i in last-1:-1:first
                    # println("At site $i: Psia $(size(Psia[i]))-$(size(Psia[i+1])), Psib $(size(Psib[i]))-$(size(Psib[i+1])), Psic $(size(Psic[i]))-$(size(Psic[i+1]))")
                    pre_sizesi = size(Psic[i])
                    pre_sizesip1 = size(Psic[i+1])

                    time_update = time_ns()
                    if i == n-1
                        disc, _, _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:backward, random_update, random_env, L=Ls[i-1], Li=i)
                    elseif i == n-2
                        disc, _, Rs[i+2] = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:backward, random_update, random_env, L=Ls[i-1], Li=i)
                    elseif i == last-1
                        disc, _, _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:backward, random_update, random_env, L=Ls[i-1], Li=i, R=Rs[i+2], Ri=i+1)
                    elseif i > 1
                        disc, _, Rs[i+2] = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:backward, random_update, random_env, L=Ls[i-1], Li=i, R=Rs[i+3], Ri=i+2)
                    else
                        disc, _, Rs[i+2] = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, i; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:backward, random_update, random_env, R=Rs[i+3], Ri=i+2)
                    end

                    time_update = (time_ns() - time_update)*1e-9
                    # println("Juliarank $juliarank step $i time_update=$time_update s, from sizes $(pre_sizesi)-$(pre_sizesip1) to sizes $(size(Psic[i]))-$(size(Psic[i+1]))")
                    # println("Juliarank $juliarank step $i Psic: $([maximum(abs.(Psic[i])) for i in 1:n])")
                    # println("Juliarank $juliarank step $i Vc: $([maximum(abs.(Vc[i])) for i in 1:n-1])")

                    # max_disc = max(max_disc, disc)
                    tot_disc += disc
                end
            end

            time_sweep = (time_ns() - time_sweep)*1e-9

            # println("Node $juliarank: sweep $sweep time_sweep=$time_sweep s")

            # println("Juliarank $juliarank after sweep $sweep Xmax: ", [maximum(X[i]) for i in 1:length(X)])

            # println("Juliarank $juliarank sweep $sweep: max_disc = $max_disc")


            time_communication = time_ns()
            if MPI.Initialized()
                if juliarank % 2 == sweep % 2 # Left
                    if juliarank != nprocs
                        # println("Prima di comm sx, $juliarank ha $(size(Psic[last])), $(size(Vc[last])), $(size(Psic[last+1]))")

                        Ls[last-1] = leftenvironment(Psia, Va, Psib, Vb, Psic, Vc, last-1; L=Ls[last-2], Li=last-1, random_env)

                        sizes = Vector{Int}(undef, 3)
                        reqs = MPI.Isend(collect(size(Ls[last-1])), comm; dest=mpirank+1, tag=juliarank)
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank+1, tag=juliarank+1)

                        MPI.Waitall([reqs, reqr])

                        Rs[last+2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                        
                        reqs = MPI.Isend(Ls[last-1], comm; dest=mpirank+1, tag=juliarank)
                        reqr = MPI.Irecv!(Rs[last+2], comm; source=mpirank+1, tag=juliarank+1)

                        MPI.Waitall([reqs, reqr])
                        disc, _, _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, last; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:backward, random_update, random_env, L=Ls[last-1], Li=last, R=Rs[last+2], Ri=last+1)

                        reqs = MPI.Isend(collect(size(Psic[last])), comm; dest=mpirank+1, tag=3*juliarank)
                        reqs1 = MPI.Isend(collect(size(Vc[last])), comm; dest=mpirank+1, tag=3*juliarank+1)
                        reqs2 = MPI.Isend(collect(size(Psic[last+1])), comm; dest=mpirank+1, tag=3*juliarank+2)

                        MPI.Waitall([reqs, reqs1, reqs2])

                        # println("Durante sx, $juliarank invia $(size(Psic[last])), $(size(Vc[last])), $(size(Psic[last+1]))")
                        reqs = MPI.Isend(Psic[last], comm; dest=mpirank+1, tag=3*juliarank)
                        reqs1 = MPI.Isend(Vc[last], comm; dest=mpirank+1, tag=3*juliarank+1)
                        reqs2 = MPI.Isend(Psic[last+1], comm; dest=mpirank+1, tag=3*juliarank+2)

                        MPI.Waitall([reqs, reqs1, reqs2])

                        Rs[last+1] = rightenvironment(Psia, Va, Psib, Vb, Psic, Vc, last+1; R=Rs[last+2], Ri=last+1, random_env)

                        # println("Dopo comm sx, $juliarank ha $(size(Psic[last])), $(size(Vc[last])), $(size(Psic[last+1]))")
                    end
                else # Right
                    if juliarank != 1
                        # println("Prima di comm dx, $juliarank ha $(size(Psic[first-1])), $(size(Vc[first-1])), $(size(Psic[first]))")

                        Rs[first+1] = rightenvironment(Psia, Va, Psib, Vb, Psic, Vc, first+1; R=Rs[first+2], Ri=first+1, random_env)

                        sizes = Vector{Int}(undef, 3)
                        reqs = MPI.Isend(collect(size(Rs[first+1])), comm; dest=mpirank-1, tag=juliarank)
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank-1, tag=juliarank-1)
                        
                        MPI.Waitall([reqs, reqr])

                        Ls[first-2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                        reqs = MPI.Isend(Rs[first+1], comm; dest=mpirank-1, tag=juliarank)
                        reqr = MPI.Irecv!(Ls[first-2], comm; source=mpirank-1, tag=juliarank-1)
                        
                        MPI.Waitall([reqs, reqr])
                        # disc, _, _ = updatecore!(Psia, Va, Psib, Vb, Psic, Vc, first-1; method, tolerance=tolerance/((n-1)*nprocs), maxbonddim, direction=:forward, L=Ls[first-2], Li=first-1, R=Rs[first+1], Ri=first) #center on first

                        sizes = Vector{Int}(undef, 4)
                        sizes1 = Vector{Int}(undef, 2)
                        sizes2 = Vector{Int}(undef, 4)

                        
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank-1, tag=3*(juliarank-1))
                        reqr1 = MPI.Irecv!(sizes1, comm; source=mpirank-1, tag=3*(juliarank-1)+1)
                        reqr2 = MPI.Irecv!(sizes2, comm; source=mpirank-1, tag=3*(juliarank-1)+2)
                        
                        MPI.Waitall([reqr, reqr1, reqr2])
                        
                        # println("Durante dx, $juliarank pronto a ricevere $sizes, $sizes1, $sizes2")

                        Psic[first-1] = ones(ValueType, sizes[1], sizes[2], sizes[3], sizes[4])
                        Vc[first-1] = ones(ValueType, sizes1[1], sizes1[2])
                        Psic[first] = ones(ValueType, sizes2[1], sizes2[2], sizes2[3], sizes2[4])

                        reqr = MPI.Irecv!(Psic[first-1], comm; source=mpirank-1, tag=3*(juliarank-1))
                        reqr1 = MPI.Irecv!(Vc[first-1], comm; source=mpirank-1, tag=3*(juliarank-1)+1)
                        reqr2 = MPI.Irecv!(Psic[first], comm; source=mpirank-1, tag=3*(juliarank-1)+2)

                        MPI.Waitall([reqr, reqr1, reqr2])

                        Ls[first-1] = leftenvironment(Psia, Va, Psib, Vb, Psic, Vc, first-1; L=Ls[first-2], Li=first-1, random_env)

                        # println("Dopo comm dx, $juliarank ha $(size(Psic[first-1])), $(size(Vc[first-1])), $(size(Psic[first]))")
                    end
                end
            end

            time_communication = (time_ns() - time_communication)*1e-9
            # println("Node $juliarank: sweep $sweep time_communication=$time_communication s")

            # println("Juliarank $juliarank post comm Psic: $([maximum(abs.(Psic[i])) for i in 1:n])")
            # println("Juliarank $juliarank post comm Vc: $([maximum(abs.(Vc[i])) for i in 1:n-1])")

            # max_disc = max(max_disc, disc) # Check last update on edge
            tot_disc += disc

            # println("Juliarank $juliarank after comm maximum Psic: $([maximum(abs.(Psic[i])) for i in 1:n])")
            # println("Juliarank $juliarank after comm maximum Vc: $([maximum(abs.(Vc[i])) for i in 1:n-1])")
            # println("juliarank $juliarank after communication sweep $sweep tot_disc = $tot_disc")


            # println("Juliarank $juliarank after comm sweep $sweep Xmax: ", [maximum(X[i]) for i in 1:length(X)])

            time_communication = (time_ns() - time_communication)*1e-9
            # println("Rank $juliarank Sweep $sweep of length $(length(first:last-1)): tot_disc = $tot_disc, time_sweep = $time_sweep, time_communication = $time_communication")

            converged = tot_disc < tolerance
            global_converged = MPI.Allreduce(converged, MPI.LAND, comm)
            if global_converged && sweep >= nprocs
                break
            end
        end

        # println("Juliarank $juliarank pre X maximum Psic: $([maximum(abs.(Psic[i])) for i in 1:n])")
        # println("Juliarank $juliarank pre X maximum Vc: $([maximum(abs.(Vc[i])) for i in 1:n-1])")

        # println("Per curiosità: $juliarank ha Vc: $([diag(Vc[i]) for i in 1:n-1])")

        time_final = time_ns()
        for i in noderanges[juliarank]
            if i != n
                Psic[i] = _contract(Psic[i], Vc[i], (4,), (1,))
            end
        end

        # println("Juliarank $juliarank X maximum X: $([maximum(abs.(X[i])) for i in 1:n])")

        if juliarank == 1
            for j in 2:nprocs
                Psic[noderanges[j][1]:noderanges[j][end]] = MPI.recv(comm; source=j-1, tag=1)
            end
        else
            MPI.send(Psic[noderanges[juliarank][1]:noderanges[juliarank][end]], comm; dest=0, tag=1)
        end

        time_final = (time_ns() - time_final)*1e-9
        # println("Node $juliarank: time_final=$time_final s")

        # println("Time final = $time_final")
        # println("Juliarank $juliarank synched X maximum X: $([maximum(abs.(X[i])) for i in 1:n])")

    end
    
    nprocs = MPI.Comm_size(comm) # In case not all processes where used to compute

    # Redistribute the tensor train among the processes.
    if synchedoutput
        if juliarank == 1
            for j in 2:nprocs
                MPI.send(Psic, comm; dest=j-1, tag=1)
            end
        else
            Psic = MPI.recv(comm; source=0, tag=1)
        end
    end

    if !synchedoutput && juliarank > 1
        return TensorTrain{ValueType,4}(Psia) # Anything is fine
    end

    # println("Noderanges: $noderanges")
    # println("My Psic sizes are: $([size(Psic[i]) for i in 1:n])")

    # println("Juliarank $juliarank return Psic: $([maximum(abs.(Psic[i])) for i in 1:n])")

    return TensorTrain{ValueType,4}(Psic)
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
- `synchedinput::Bool`: Whether the input MPOs are synchronized. Use "true" if you know that the input MPOs are exactly the same across different nodes.
- `synchedoutput::Bool`: Whether the output MPOs must be synchronized. Use "true" if you want that all the nodes have the correct MPOs as output.
"""

"""
    _twosided_prep(T)

Left-canonical copy `AL`, right-canonical copy `AR` and bond matrices `Cb[i]` of a fixed input
tensor train, with `AL[1:i] * Cb[i] * AR[i+1:n] == T` for every bond `i`.  `Cb[i]` is the overlap
`AL[1:i]^dagger AR[1:i]` -- a product, never an inverse.  Computed once, identically on every rank.
"""
function _twosided_prep(T::Vector{Array{V,4}}) where {V}
    AL = deepcopy(T); centercanonicalize!(AL, length(AL))
    AR = deepcopy(T); centercanonicalize!(AR, 1)
    Cb = Vector{Matrix{V}}(undef, length(T))
    E = ones(V, 1, 1)
    for i in eachindex(T)
        tmp = _contract(E, conj(AL[i]), (1,), (1,))
        E = _contract(tmp, AR[i], (1, 2, 3), (1, 2, 3))
        Cb[i] = E
    end
    return (AL, AR, Cb)
end

"""
    _center_inputs!(T, prep, abgauge, i)

Put the fixed input `T` in mixed-canonical form centred at site `i`.  `:twosided` uses the
precomputed data: `AL` left of `i`, `AL[i]*Cb[i]` at `i`, `AR` right of `i`.  Every rank therefore
holds bit-identical tensors on every bond left/right of its centre, so environments exchanged
between ranks are always contracted in matching bases.  `:percopy` is the original behaviour
(each rank re-canonicalises its private copy), which desynchronises the gauges across ranks.
"""
function _center_inputs!(T, prep, abgauge::Symbol, i::Int)
    if abgauge === :twosided
        AL, AR, Cb = prep
        for j in eachindex(T)
            T[j] = j < i ? AL[j] : (j > i ? AR[j] : _contract(AL[j], Cb[j], (4,), (1,)))
        end
    else
        centercanonicalize!(T, i)
    end
    return T
end

function contract_distr_fit(
    mpoA::TensorTrain{ValueType,4},
    mpoB::TensorTrain{ValueType,4};
    nsweeps::Int=8,
    initial_guess::Union{Nothing,TensorTrain{ValueType,4}}=nothing,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    noderanges::Union{Nothing,Vector{UnitRange{Int}}}=nothing,
    tolerance::Float64=1e-12,
    method::Symbol=:RSVD, # :SVD, :RSVD, :LU, :CI
    maxbonddim::Int=typemax(Int),
    synchedinput::Bool=false,
    synchedoutput::Bool=true,
    random_update::Bool=false,
    random_env::Bool=false,
    stable::Bool=true,
    abgauge::Symbol=:twosided, # :twosided (consistent, default) or :percopy (original per-rank re-canonicalisation)
    kwargs...
)::TensorTrain{ValueType,4} where {ValueType}
    if length(mpoA) != length(mpoB)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end

    # println("La tolerance usata e': $tolerance")
    if !synchedinput
        synchronize_tt!(mpoA)
        synchronize_tt!(mpoB)
        if initial_guess != nothing
            synchronize_tt!(initial_guess)
        end
    end

    n = length(mpoA)

    A = Vector{Array{ValueType, 4}}(undef, n)
    B = Vector{Array{ValueType, 4}}(undef, n)
    X = Vector{Array{ValueType, 4}}(undef, n)
    for i in 1:n
        A[i] = deepcopy(mpoA.sitetensors[i])
        B[i] = deepcopy(mpoB.sitetensors[i])
        if !isnothing(initial_guess)
            X[i] = deepcopy(initial_guess.sitetensors[i])
        else
            if size(mpoA[i])[2] == size(mpoB[i])[3]
                X[i] = deepcopy(mpoB.sitetensors[i])
            else
                X[i] = ones(ValueType, size(mpoB[i])[1], size(mpoA[i])[2], size(mpoB[i])[3], size(mpoB[i])[4])
            end
        end
    end

    # One-time, rank-identical canonical data for the inputs (see _twosided_prep).
    abprepA = abgauge === :twosided ? _twosided_prep(A) : nothing
    abprepB = abgauge === :twosided ? _twosided_prep(B) : nothing

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
                return contract_fit(mpoA, mpoB; nsweeps, initial_guess, tolerance, method, maxbonddim, kwargs...)
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
                if nprocs > div(n - extra1 - extraend, 2) # It's just one update per node.
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
        return contract_fit(mpoA, mpoB; nsweeps, initial_guess, tolerance, method, maxbonddim, kwargs...)
    end

    if nprocs == 1
        println("Warning! Distributed strategy has been chosen, but only one process is running")
        return contract_fit(mpoA, mpoB; nsweeps, initial_guess, tolerance, method, maxbonddim, kwargs...)
    end

    if juliarank <= nprocs

        time_precompute = time_ns()

        Ls = Vector{Array{ValueType, 3}}(undef, n)
        Rs = Vector{Array{ValueType, 3}}(undef, n)

        first = noderanges[juliarank][1]
        last = noderanges[juliarank][end]

        #Ls are left enviroments and Rs are right environments
        if first != 1
            init_env = rand(ValueType, size(A[first])[1], size(B[first])[1], size(X[first])[1])
            Ls[first-1] = init_env ./ sqrt(sum(init_env.^2))
        end
        if last != n
            init_env = rand(ValueType, size(A[last])[end], size(B[last])[end], size(X[last])[end])
            Rs[last+1] = init_env ./ sqrt(sum(init_env.^2))
        end

        # println("Juliarank $juliarank my A at the start is $([A[i][1,1,1,1] for i in 1:n]), $([A[i][end,end,end,end] for i in 1:n])")
        # println("Juliarank $juliarank my B at the start is $([B[i][1,1,1,1] for i in 1:n]), $([B[i][end,end,end,end] for i in 1:n])")

        if juliarank % 2 == 1 # Precompute right environment if going forward
            if !random_env && stable
                _center_inputs!(A, abprepA, abgauge, first)
                # _center_inputs!(A, abprepA, abgauge, 1)
                _center_inputs!(B, abprepB, abgauge, first)
                # _center_inputs!(B, abprepB, abgauge, 1)
            end
            centercanonicalize!(X, first)
            for i in last:-1:first+2#n:-1:3#last:-1:first+2
                if i == n
                    Rs[i] = rightenvironment(A, B, X, i; random_env)
                else
                    Rs[i] = rightenvironment(A, B, X, i; R=Rs[i+1], Ri=i, random_env)
                end
            end
        else # Precompute left environment if going backward
            if !random_env && stable
                _center_inputs!(A, abprepA, abgauge, last-1)
                # _center_inputs!(A, abprepA, abgauge, n)
                _center_inputs!(B, abprepB, abgauge, last-1)
                # _center_inputs!(B, abprepB, abgauge, n)
            end
            centercanonicalize!(X, last)
            for i in first:last-2#1:n-2#first:last-2
                if i == 1
                    Ls[i] = leftenvironment(A, B, X, i; random_env)
                else
                    Ls[i] = leftenvironment(A, B, X, i; L=Ls[i-1], Li=i, random_env)
                end
            end
        end

        # println("Juliarank $juliarank my A after envs is $([A[i][1,1,1,1] for i in 1:n]), $([A[i][end,end,end,end] for i in 1:n])")
        # println("Juliarank $juliarank my B after envs is $([B[i][1,1,1,1] for i in 1:n]), $([B[i][end,end,end,end] for i in 1:n])")


        time_precompute = (time_ns() - time_precompute)*1e-9
        # println("Node $juliarank: Precomputation time: $time_precompute s")

        # TODO AT THE END IS NOT CENTERCANONICAL!
        for sweep in 1:nsweeps
            tot_disc = 0.0
            time_update = time_ns()

            # println("Juliarank $juliarank pre sweep $sweep Xmax: ", [maximum(X[i]) for i in 1:length(X)])
            # println("Juliarank $juliarank starting sweep $sweep, X=$(checkorthogonality(X))")

            if sweep % 2 == juliarank % 2
                for i in first:last-1
                    if !random_env && stable
                        _center_inputs!(A, abprepA, abgauge, i)
                        _center_inputs!(B, abprepB, abgauge, i)
                    end
                    # println("Max sweep $sweep: ", [maximum(A[i]) for i in 1:length(A)])
                    # println("Max sweep $sweep: ", [maximum(B[i]) for i in 1:length(B)])

                    # println("Juliarank $juliarank updating site $i forward with X=$(checkorthogonality(X))")

                    if i == 1
                        disc, _, _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:forward, random_update, random_env, R=Rs[i+2], Ri=i+1)
                    elseif i == 2
                        disc, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:forward, random_update, random_env, R=Rs[i+2], Ri=i+1)
                    elseif i == first
                        disc, _, _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:forward, random_update, random_env, L=Ls[i-1], Li=i, R=Rs[i+2], Ri=i+1)
                    elseif i < n-1
                        disc, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:forward, random_update, random_env, L=Ls[i-2], Li=i-1, R=Rs[i+2], Ri=i+1)
                    else
                        disc, Ls[i-1], _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:forward, random_update, random_env, L=Ls[i-2], Li=i-1)
                    end
                    # println("Juliarank $juliarank sweep $sweep site $i: disc = $disc")
                    # max_disc = max(max_disc, disc)
                    tot_disc += disc
                    # println("Juliarank $juliarank after updating $i, X=$(checkorthogonality(X))")

                end
            else
                for i in last-1:-1:first
                    if !random_env && stable
                        _center_inputs!(A, abprepA, abgauge, i)
                        _center_inputs!(B, abprepB, abgauge, i)
                    end

                    # println("Juliarank $juliarank updating site $i backward with X=$(checkorthogonality(X))")

                    if i == n-1
                        disc, _, _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:backward, random_update, random_env, L=Ls[i-1], Li=i)
                    elseif i == n-2
                        disc, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:backward, random_update, random_env, L=Ls[i-1], Li=i)
                    elseif i == last-1
                        disc, _, _ = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:backward, random_update, random_env, L=Ls[i-1], Li=i, R=Rs[i+2], Ri=i+1)
                    elseif i > 1
                        disc, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:backward, random_update, random_env, L=Ls[i-1], Li=i, R=Rs[i+3], Ri=i+2)
                    else
                        disc, _, Rs[i+2] = updatecore!(A, B, X, i; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:backward, random_update, random_env, R=Rs[i+3], Ri=i+2)
                    end
                    # println("Juliarank $juliarank sweep $sweep site $i: disc = $disc")
                    # println("Juliarank $juliarank after updating $i, X=$(checkorthogonality(X))")

                    # max_disc = max(max_disc, disc)
                    tot_disc += disc
                end
            end

            # println("Juliarank $juliarank my A after sweep is $([A[i][1,1,1,1] for i in 1:n]), $([A[i][end,end,end,end] for i in 1:n])")
            # println("Juliarank $juliarank my B after sweep is $([B[i][1,1,1,1] for i in 1:n]), $([B[i][end,end,end,end] for i in 1:n])")

            # println("Juliarank $juliarank after sweep $sweep, A=$(checkorthogonality(A)), B=$(checkorthogonality(B))")


            time_update = (time_ns() - time_update)*1e-9

            # println("Juliarank $juliarank after sweep $sweep Xmax: ", [maximum(X[i]) for i in 1:length(X)])


            # println("Juliarank $juliarank sweep $sweep: max_disc = $max_disc")
            time_communication = time_ns()
            if MPI.Initialized()
                if juliarank % 2 == sweep % 2 # Left
                    if juliarank != nprocs
                        if !random_env && stable
                            _center_inputs!(A, abprepA, abgauge, last)
                            _center_inputs!(B, abprepB, abgauge, last)
                        end

                        Ls[last-1] = leftenvironment(A, B, X, last-1; L=Ls[last-2], Li=last-1, random_env)

                        sizes = Vector{Int}(undef, 3)
                        reqs = MPI.Isend(collect(size(Ls[last-1])), comm; dest=mpirank+1, tag=juliarank)
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank+1, tag=juliarank+1)

                        MPI.Waitall([reqs, reqr])

                        Rs[last+2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                        
                        reqs = MPI.Isend(Ls[last-1], comm; dest=mpirank+1, tag=juliarank)
                        reqr = MPI.Irecv!(Rs[last+2], comm; source=mpirank+1, tag=juliarank+1)

                        MPI.Waitall([reqs, reqr])
                        disc, _, _ = updatecore!(A, B, X, last; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:backward, random_update, random_env, L=Ls[last-1], Li=last, R=Rs[last+2], Ri=last+1)
                        
                        Rs[last+1] = rightenvironment(A, B, X, last+1; R=Rs[last+2], Ri=last+1, random_env)
                    end
                else # Right
                    if juliarank != 1
                        if !random_env && stable
                            _center_inputs!(A, abprepA, abgauge, first)
                            _center_inputs!(B, abprepB, abgauge, first)
                        end

                        Rs[first+1] = rightenvironment(A, B, X, first+1; R=Rs[first+2], Ri=first+1, random_env)

                        sizes = Vector{Int}(undef, 3)
                        reqs = MPI.Isend(collect(size(Rs[first+1])), comm; dest=mpirank-1, tag=juliarank)
                        reqr = MPI.Irecv!(sizes, comm; source=mpirank-1, tag=juliarank-1)
                        
                        MPI.Waitall([reqs, reqr])

                        Ls[first-2] = ones(ValueType, sizes[1], sizes[2], sizes[3])
                        reqs = MPI.Isend(Rs[first+1], comm; dest=mpirank-1, tag=juliarank)
                        reqr = MPI.Irecv!(Ls[first-2], comm; source=mpirank-1, tag=juliarank-1)
                        
                        MPI.Waitall([reqs, reqr])
                        disc, _, _ = updatecore!(A, B, X, first-1; method, tolerance=tolerance/((n-1)), maxbonddim, direction=:forward, random_update, random_env, L=Ls[first-2], Li=first-1, R=Rs[first+1], Ri=first) #center on first

                        Ls[first-1] = leftenvironment(A, B, X, first-1; L=Ls[first-2], Li=first-1, random_env)
                    end
                end
            end
            # max_disc = max(max_disc, disc) # Check last update on edge
            tot_disc += disc


            time_communication = (time_ns() - time_communication)*1e-9
            # println("Rank $juliarank Sweep $sweep: max_disc = $max_disc, time_update = $time_update s, time_communication = $time_communication s")

            converged = tot_disc < tolerance
            global_converged = MPI.Allreduce(converged, MPI.LAND, comm)
            if global_converged
                break
            end
        end

        rounds = ceil(Int, log2(nprocs))

        my_first = partner_first = my_last = partner_last = my_old_first = partner_old_first = my_old_last = partner_old_last = 0

        # println("Juliarank $juliarank before starting with X=$(checkorthogonality(X)), X=$([size(X[i]) for i in 1:length(X)])")


        # println("Juliarank $juliarank starting merging phase with $rounds rounds. X=$(checkorthogonality(X)), X=$([size(X[i]) for i in 1:length(X)])")
        for s in 1:rounds
            my_old_first, partner_old_first, my_old_last, partner_old_last = my_first, partner_first, my_last, partner_last
            delta = 2^(s-1)
            
            if ((juliarank - 1) % (2*delta)) == 0 # I'm a receiver
                juliapartner = juliarank + delta
                # println("Out of $nprocs, I am juliarank $juliarank at round $s receiving from $juliapartner")
                my_first = noderanges[juliarank][1]
                partner_first = noderanges[juliapartner][1]
                my_last  = partner_first - 1
                partner_last  = noderanges[juliapartner+delta-1][end]
                # println("Out of $nprocs, I am juliarank $juliarank at round $s with $my_first:$my_last receiving from $juliapartner with $partner_first:$partner_last")

                juliapartner = juliarank + delta
                if !random_env && stable
                    _center_inputs!(A, abprepA, abgauge, my_last)
                    _center_inputs!(B, abprepB, abgauge, my_last)
                end

                # println("Juliarank $juliarank at round $s trying to center on site $my_last with X=$(checkorthogonality(X)), X=$([size(X[i]) for i in 1:length(X)])")
                centercanonicalize!(X, my_last)

                X[partner_first:partner_last] = MPI.recv(comm; source = juliapartner-1, tag = 2*juliapartner)
                if abgauge !== :twosided # :twosided: every rank already holds the same AL/AR (see _center_inputs!)
                    MPI.send(A[partner_first:partner_last], comm; dest = juliapartner-1, tag = 2*juliarank)
                    MPI.send(B[partner_first:partner_last], comm; dest = juliapartner-1, tag = 2*juliarank + 1)
                end

                # R = MPI.recv(comm; source = juliapartner-1, tag = 2*juliapartner+1)
                Ri = partner_first

                # TODO riprendi da qui, ragionando su chi sono L e Li
                if s == 1
                    # TODO I should distinguish l2r and r2l
                    if juliarank != 1
                        L = leftenvironment(A, B, X, my_last-1; random_env, L=Ls[my_first-1], Li=my_first) # TODO sicuro?
                        # L = leftenvironment(A, B, X, my_last-1; random_env)
                    else
                        L = leftenvironment(A, B, X, my_last-1; random_env)
                    end
                else
                    L = leftenvironment(A, B, X, my_last-1; L=Ls[my_old_last-1], Li=my_old_last, random_env)
                    # L = leftenvironment(A, B, X, my_last-1; L=Ls[my_old_last-1], Li=my_old_last, random_env)
                    # L = leftenvironment(A, B, X, my_last-1; random_env)
                end
                Li = my_last

                # L_test = leftenvironment(A, B, X, my_last-1; random_env)
                R = MPI.recv(comm; source = juliapartner-1, tag = 2*juliapartner+1)

                # if s == 1
                    # if juliarank != nprocs
                        # R_test = rightenvironment(A, B, X, my_last+2; R, Ri=partner_last, random_env)
                    # else
                        # R_test = rightenvironment(A, B, X, my_last+2; random_env)
                    # end
                # else
                    # R_test = rightenvironment(A, B, X, my_last+2; random_env)
                    # R_test = rightenvironment(A, B, X, my_last+2; R, Ri=partner_last, random_env)
                # end

                Ri = my_last + 1
                # R = rightenvironment(A, B, X, my_last+2; random_env)
                # Rs[my_last+2] = R
                # Ls[my_last-1] = L

                _, Ls[my_last-1], Rs[my_last+2] = updatecore!(
                   A, B, X, my_last;
                   method, tolerance=tolerance/((n-1)), maxbonddim,
                   direction = :forward,
                   random_update, random_env,
                   R, Ri, L, Li
                )
                # Provo con L e R_test:
                # R è sicuramente sbagliato.
            elseif (s == 1) || (((juliarank - 1) % (2*delta)) == delta) # I'm a sender
                # println("Juliarank $juliarank is a sender at round $s and my noderanges last should be noderanges[$(juliarank+delta-1)] since delta=$delta, but max is $(length(noderanges))")
                juliapartner = juliarank - delta
                my_first = noderanges[juliarank][1]
                partner_first = noderanges[juliapartner][1]
                partner_last  = my_first - 1
                my_last = noderanges[juliarank+delta-1][end]

                # println("Juliarank $juliarank is a sender at round $s with $my_first:$my_last sending to $juliapartner with $partner_first:$partner_last")
                # println("Juliarank $juliarank at round $s trying to center on site $my_first with X=$(checkorthogonality(X)), X=$([size(X[i]) for i in 1:length(X)])")
                if !random_env && stable
                    _center_inputs!(A, abprepA, abgauge, my_first)
                    _center_inputs!(B, abprepB, abgauge, my_first)
                end
                # println("Juliarank $juliarank at round $s trying to center on site $my_first with X=$(checkorthogonality(X)), X=$([size(X[i]) for i in 1:length(X)])")
                centercanonicalize!(X, my_first)

                # println("Juliarank $juliarank RIGHT at round $s before communication: my_first=$my_first, my_last=$my_last, partner_first=$partner_first, partner_last=$partner_last")

                 # send X[my_first:my_last] to receiver
                MPI.send(X[my_first:my_last], comm; dest = juliapartner - 1, tag = 2*juliarank)
                if abgauge !== :twosided
                    A[my_first:my_last] = MPI.recv(comm; source = juliapartner-1, tag = 2*juliapartner)
                    B[my_first:my_last] = MPI.recv(comm; source = juliapartner-1, tag = 2*juliapartner+1)
                end


                # TODO this is a bored implementation, I could improve it.
                if s == 1
                    if my_last != n
                        R = rightenvironment(A, B, X, my_first+1; R=Rs[my_last+1], Ri=my_last, random_env)
                    else
                        R = rightenvironment(A, B, X, my_first+1; random_env)
                    end
                else
                    R = rightenvironment(A, B, X, my_first+1; R=Rs[partner_old_first+1], Ri=partner_old_first, random_env)
                end

                MPI.send(R, comm; dest = juliapartner - 1, tag = 2*juliarank+1)
            end

        end

#= Extra experiments
        centercanonicalize!(X, noderanges[juliarank][1])
        if juliarank == 1
            centercanonicalize!(X, noderanges[juliarank][end])
            println("X=$(checkorthogonality(X))")
        end
        if juliarank == 1
            for j in 2:nprocs
                # centercanonicalize!(X, noderanges[j-1][end])
                first = noderanges[j][1]
                last = noderanges[j][end]
                X[first:last] = MPI.recv(comm; source=j-1, tag=1)
                X1X2 = _contract(X[first-1], X[first], (4,), (1,))
                left, sv, right, newbonddim, disc = _factorize(
                    reshape(X1X2, prod(size(X1X2)[1:3]), prod(size(X1X2)[4:6])),
                    method; tolerance=0.0, maxbonddim=999, diamond=true
                    )

                println("X=$(checkorthogonality(X))")

                X[first-1] = reshape(left, size(X[first-1])[1:3]..., newbonddim)
                X[first] = reshape(diagm(sv)*right, newbonddim, size(X[first])[2:4]...)

                println("X=$(checkorthogonality(X))")
                centercanonicalize!(X, last)
                println("X=$(checkorthogonality(X))")
            end
        else
            centercanonicalize!(X, noderanges[juliarank][1])
            MPI.send(X[noderanges[juliarank][1]:noderanges[juliarank][end]], comm; dest=0, tag=1)
        end
=#

    end
    nprocs = MPI.Comm_size(comm) # In case not all processes where used to compute

    # Redistribute the tensor train among the processes (only if asked, as in the inverse-canonical
    # path; the original forced it with `synchedoutput || true`, charging P-1 serial sends of the
    # whole train to the site-canonical timing only).  Rank 0's output is unaffected.
    if synchedoutput
        if juliarank == 1
            for j in 2:nprocs
                MPI.send(X, comm; dest=j-1, tag=1)
            end
        else
            X = MPI.recv(comm; source=0, tag=1)
        end
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
    Psia::Vector{Array{V1,4}},
    Va::Vector{Array{V1,2}},
    Psib::Vector{Array{V2,4}},
    Vb::Vector{Array{V2,2}};
    algorithm::Symbol=:TCI,
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    method::Symbol=:RSVD,
    f::Union{Nothing,Function}=nothing,
    subcomm::Union{Nothing, MPI.Comm}=nothing,
    kwargs...
)::TensorTrain{promote_type(V1,V2), 4} where {V1,V2}
    Vres = promote_type(V1, V2)
    if algorithm != :fit && algorithm != :distrfit
        println("Warning! Inverse canonical form detected, but algorithm $algorithm does not support it. Converting to site canonical form.")
        for i in 1:length(Psia)-1
            Psia[i] = _contract(Psia[i], Va[i], (4,), (1,))
            Psib[i] = _contract(Psib[i], Vb[i], (4,), (1,))
        end
    end
    if algorithm === :TCI
        Psia = TensorTrain{Vres,4}(Psia) # For testing reason, JET prefers it right before call.
        Psib = TensorTrain{Vres,4}(Psib)
        mpo = contract_TCI(Psia, Psib; tolerance=tolerance, maxbonddim=maxbonddim, f=f, kwargs...)
    elseif algorithm === :naive
        error("Naive contraction implementation cannot be used with inverse gauge canonical form. Use algorithm=:fit instead.")
        Psia = TensorTrain{Vres,4}(Psia)
        Psib = TensorTrain{Vres,4}(Psib)
        mpo = contract_naive(Psia, Psib; tolerance=tolerance, maxbonddim=maxbonddim)
    elseif algorithm === :zipup
        error("Zipup contraction implementation cannot be used with inverse gauge canonical form. Use algorithm=:fit instead.")
        Psia = TensorTrain{Vres,4}(Psia)
        Psib = TensorTrain{Vres,4}(Psib)
        mpo = contract_zipup(Psia, Psib; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    elseif algorithm === :distrzipup
        error("Zipup contraction implementation cannot be used with inverse gauge canonical form. Use algorithm=:fit instead.")
        Psia = TensorTrain{Vres,4}(Psia)
        Psib = TensorTrain{Vres,4}(Psib)
        mpo = contract_distr_zipup(Psia, Psib; tolerance=tolerance, maxbonddim=maxbonddim, method=method, subcomm=subcomm, kwargs...)
    elseif algorithm === :fit
        if f !== nothing
            error("Fit contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        mpo = contract_fit(Psia, Va, Psib, Vb; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    elseif algorithm === :distrfit
        if f !== nothing
            error("Fit contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        mpo = contract_distr_fit(Psia, Va, Psib, Vb; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))
    end
    return mpo
end


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
    if algorithm === :TCI
        mpo = contract_TCI(A, B; tolerance=tolerance, maxbonddim=maxbonddim, f=f, kwargs...)
    elseif algorithm === :naive
        if f !== nothing
            error("Naive contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        mpo = contract_naive(A, B; tolerance=tolerance, maxbonddim=maxbonddim)
    elseif algorithm === :zipup
        if f !== nothing
            error("Zipup contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        mpo = contract_zipup(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    elseif algorithm === :distrzipup
        if f !== nothing
            error("Zipup contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        mpo = contract_distr_zipup(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method, subcomm=subcomm, kwargs...)
    elseif algorithm === :fit
        if f !== nothing
            error("Fit contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        mpo = contract_fit(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    elseif algorithm === :distrfit
        if f !== nothing
            error("Fit contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        mpo = contract_distr_fit(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method, kwargs...)
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))
    end
    return mpo
end

# TODO implement this for inverse form
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
