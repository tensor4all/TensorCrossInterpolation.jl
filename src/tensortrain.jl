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
mutable struct TensorTrain{ValueType,N} <: AbstractTensorTrain{ValueType}
    sitetensors::Vector{Array{ValueType,N}}

    function TensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}) where {ValueType,N}
        for i in 1:length(sitetensors)-1
            if (last(size(sitetensors[i])) != size(sitetensors[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        new{ValueType,N}(sitetensors)
    end
end

function Base.show(io::IO, obj::TensorTrain{V,N}) where {V,N}
    print(
        io,
        "$(typeof(obj)) of rank $(maximum(linkdims(obj)))"
    )
end

function TensorTrain{V2,N}(tt::TensorTrain{V})::TensorTrain{V2,N} where {V,V2,N}
    return TensorTrain{V2,N}(Array{V2}.(sitetensors(tt)))
end

"""
    function TensorTrain(sitetensors::Vector{Array{V, 3}}) where {V}

Create a tensor train out of a vector of tensors. Each tensor should have links to the
previous and next tensor as dimension 1 and 3, respectively; the local index ("physical
index" for MPS in physics) is dimension 2.
"""
function TensorTrain(sitetensors::AbstractVector{<:AbstractArray{V,N}}) where {V,N}
    return TensorTrain{V,N}(sitetensors)
end

"""
    function TensorTrain(tci::AbstractTensorTrain{V}) where {V}

Convert a tensor-train-like object into a tensor train. This includes TCI1 and TCI2 objects.

See also: [`TensorCI1`](@ref), [`TensorCI2`](@ref).
"""
function TensorTrain(tci::AbstractTensorTrain{V})::TensorTrain{V,3} where {V}
    return TensorTrain{V,3}(sitetensors(tci))
end

"""
    function TensorTrain{V2,N}(tci::AbstractTensorTrain{V}) where {V,V2,N}

Convert a tensor-train-like object into a tensor train.

Arguments:
- `tt::AbstractTensorTrain{V}`: a tensor-train-like object.
- `localdims`: a vector of local dimensions for each tensor in the tensor train. A each element
  of `localdims` should be an array-like object of `N-2` integers.
"""
function TensorTrain{V2,N}(tt::AbstractTensorTrain{V}, localdims)::TensorTrain{V2,N} where {V,V2,N}
    for d in localdims
        length(d) == N - 2 || error("Each element of localdims be a list of N-2 integers.")
    end
    for n in 1:length(tt)
        prod(size(tt[n])[2:end-1]) == prod(localdims[n]) || error("The local dimensions at n=$n must match the tensor sizes.")
    end
    return TensorTrain{V2,N}(
        [reshape(Array{V2}(t), size(t, 1), localdims[n]..., size(t)[end]) for (n, t) in enumerate(sitetensors(tt))])
end

function TensorTrain{N}(tt::AbstractTensorTrain{V}, localdims)::TensorTrain{V,N} where {V,N}
    return TensorTrain{V,N}(tt, localdims)
end

function tensortrain(tci)
    return TensorTrain(tci)
end

function _factorize(
    A::AbstractMatrix{V}, method::Symbol; tolerance::Float64, maxbonddim::Int, leftorthogonal::Bool=false, diamond::Bool=false, normalizeerror=true, q::Int=0, p::Int=16
)::Union{Tuple{Matrix{V},Matrix{V},Int},Tuple{Matrix{V},Vector{V},Matrix{V},Int}} where {V}
    reltol = 1e-14
    abstol = 0.0
    if normalizeerror
        reltol = tolerance
    else
        abstol = tolerance
    end
    if method === :LU
        factorization = rrlu(A, abstol=abstol, reltol=reltol, maxrank=maxbonddim, leftorthogonal=leftorthogonal)
        return left(factorization), right(factorization), npivots(factorization)
    elseif method === :CI
        factorization = MatrixLUCI(A, abstol=abstol, reltol=reltol, maxrank=maxbonddim, leftorthogonal=leftorthogonal)
        return left(factorization), right(factorization), npivots(factorization)
    elseif method === :SVD
        factorization = LinearAlgebra.svd(A)
        err = [sum(factorization.S[n+1:end] .^ 2) for n in 1:length(factorization.S)]
        normalized_err = err ./ sum(factorization.S .^ 2)

        trunci = min(
            replacenothing(findlast(>(tolerance), Array(factorization.S)), 1),
            maxbonddim
        )
        if diamond
            return (
                    Matrix(factorization.U[:, 1:trunci]),
                    factorization.S[1:trunci],
                    Matrix(factorization.Vt[1:trunci, :]),
                    trunci
                )
        else
            if leftorthogonal
                return (
                    Matrix(factorization.U[:, 1:trunci]),
                    Matrix(Diagonal(factorization.S[1:trunci]) * factorization.Vt[1:trunci, :]),
                    trunci
                )
            else
                return (
                    Matrix(factorization.U[:, 1:trunci] * Diagonal(factorization.S[1:trunci])),
                    Matrix(factorization.Vt[1:trunci, :]),
                    trunci
                )
            end
        end
    elseif method === :RSVD
        invert = false
        if size(A)[1] < size(A)[2]
            A = A'
            invert = true
        end
        if maxbonddim == typemax(Int) || maxbonddim + p > size(A)[1] || maxbonddim + p > size(A)[2]
            factorization = LinearAlgebra.svd(A)
        else
            m, n = size(A)
            G = randn(n, maxbonddim + p)
            Y = A*G 
            Q = Matrix(LinearAlgebra.qr!(Y).Q) # THEORETICAL BOTTLENECK
            for _ in 1:q # q=0 for best performance
                Y = A'*Q
                Q = Matrix(LinearAlgebra.qr!(Y).Q)
                Y = A*Q
                Q = Matrix(LinearAlgebra.qr!(Y).Q)
            end
            B = Q' * A
            factorization = LinearAlgebra.svd(B)
            factorization = SVD((Q*factorization.U)[:,1:maxbonddim], factorization.S[1:maxbonddim], factorization.Vt[1:maxbonddim,:])
        end
        trunci = min(
            replacenothing(findlast(>(tolerance), Array(factorization.S)), 1),
            maxbonddim
        )
        if diamond
            if invert
                return (
                    Matrix(factorization.Vt[1:trunci, :]'),
                    factorization.S[1:trunci],
                    Matrix(factorization.U[:, 1:trunci]'),
                    trunci
                )
            else
                return (
                    Matrix(factorization.U[:, 1:trunci]),
                    factorization.S[1:trunci],
                    Matrix(factorization.Vt[1:trunci, :]),
                    trunci
                )
            end
        else
            if leftorthogonal
                if invert
                    return (
                        Matrix(factorization.Vt[1:trunci, :]' * Diagonal(factorization.S[1:trunci])),
                        Matrix(factorization.U[:, 1:trunci]'),
                        trunci
                    )
                else
                    return (
                        Matrix(factorization.U[:, 1:trunci]),
                        Matrix(Diagonal(factorization.S[1:trunci]) * factorization.Vt[1:trunci, :]),
                        trunci
                    )
                end
            else
                if invert
                    return (
                        Matrix(factorization.Vt[1:trunci, :]' * Diagonal(factorization.S[1:trunci])),
                        Matrix(factorization.U[:, 1:trunci]'),
                        trunci
                    )
                else
                    return (
                        Matrix(factorization.U[:, 1:trunci] * Diagonal(factorization.S[1:trunci])),
                        Matrix(factorization.Vt[1:trunci, :]),
                        trunci
                    )
                end
            end
        end
    else
        error("Not implemented yet.")
    end
end

"""
    function compress!(
        tt::TensorTrain{V, N},
        method::Symbol=:LU;
        tolerance::Float64=1e-12,
        maxbonddim=typemax(Int)
    ) where {V, N}

Compress the tensor train `tt` using `LU`, `CI` or `SVD` decompositions.
"""
function compress!(
    tt::TensorTrain{V,N},
    method::Symbol=:LU;
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    normalizeerror::Bool=true
) where {V,N}
    # From left to right
    for ell in 1:length(tt)-1
        shapel = size(tt.sitetensors[ell])
        left, right, newbonddim = _factorize(
            reshape(tt.sitetensors[ell], prod(shapel[1:end-1]), shapel[end]),
            method; tolerance=0.0, maxbonddim=typemax(Int), leftorthogonal=true # no truncation
        )
        tt.sitetensors[ell] = reshape(left, shapel[1:end-1]..., newbonddim)
        shaper = size(tt.sitetensors[ell+1])
        nexttensor = right * reshape(tt.sitetensors[ell+1], shaper[1], prod(shaper[2:end]))
        tt.sitetensors[ell+1] = reshape(nexttensor, newbonddim, shaper[2:end]...)
    end

    # From right to left
    for ell in length(tt):-1:2
        shaper = size(tt.sitetensors[ell])
        left, right, newbonddim = _factorize(
            reshape(tt.sitetensors[ell], shaper[1], prod(shaper[2:end])),
            method; tolerance, maxbonddim, normalizeerror, leftorthogonal=false
        )
        tt.sitetensors[ell] = reshape(right, newbonddim, shaper[2:end]...)
        shapel = size(tt.sitetensors[ell-1])
        nexttensor = reshape(tt.sitetensors[ell-1], prod(shapel[1:end-1]), shapel[end]) * left
        tt.sitetensors[ell-1] = reshape(nexttensor, shapel[1:end-1]..., newbonddim)
    end

    nothing
end

function multiply!(tt::TensorTrain{V,N}, a) where {V,N}
    tt.sitetensors[end] .= tt.sitetensors[end] .* a
    nothing
end

function multiply!(a, tt::TensorTrain{V,N}) where {V,N}
    tt.sitetensors[end] .= a .* tt.sitetensors[end]
    nothing
end

function multiply(tt::TensorTrain{V,N}, a)::TensorTrain{V,N} where {V,N}
    tt2 = deepcopy(tt)
    multiply!(tt2, a)
    return tt2
end

function multiply(a, tt::TensorTrain{V,N})::TensorTrain{V,N} where {V,N}
    tt2 = deepcopy(tt)
    multiply!(a, tt2)
    return tt2
end

function Base.:*(tt::TensorTrain{V,N}, a)::TensorTrain{V,N} where {V,N}
    return multiply(tt, a)
end

function Base.:*(a, tt::TensorTrain{V,N})::TensorTrain{V,N} where {V,N}
    return multiply(a, tt)
end

function divide!(tt::TensorTrain{V,N}, a) where {V,N}
    tt.sitetensors[end] .= tt.sitetensors[end] ./ a
    nothing
end

function divide(tt::TensorTrain{V,N}, a) where {V,N}
    tt2 = deepcopy(tt)
    divide!(tt2, a)
    return tt2
end

function Base.:/(tt::TensorTrain{V,N}, a) where {V,N}
    return divide(tt, a)
end

function Base.reverse(tt::AbstractTensorTrain{V}) where {V}
    return tensortrain(reverse([
        permutedims(T, (ndims(T), (2:ndims(T)-1)..., 1)) for T in sitetensors(tt)
    ]))
end


"""
Fitting data with a TensorTrain object.
This may be useful when the interpolated function is noisy.
"""
struct TensorTrainFit{ValueType}
    indexsets::Vector{MultiIndex}
    values::Vector{ValueType}
    tt::TensorTrain{ValueType,3}
    offsets::Vector{Int}
end

function TensorTrainFit{ValueType}(indexsets, values, tt) where {ValueType}
    offsets = [0]
    for n in 1:length(tt)
        push!(offsets, offsets[end] + length(tt[n]))
    end
    return TensorTrainFit{ValueType}(indexsets, values, tt, offsets)
end

flatten(obj::TensorTrain{ValueType}) where {ValueType} = vcat(vec.(sitetensors(obj))...)

function to_tensors(obj::TensorTrainFit{ValueType}, x::Vector{ValueType}) where {ValueType}
    return [
        reshape(
            x[obj.offsets[n]+1:obj.offsets[n+1]],
            size(obj.tt[n])
        )
        for n in 1:length(obj.tt)
    ]
end

function _evaluate(tt::Vector{Array{V,3}}, indexset) where {V}
    only(prod(T[:, i, :] for (T, i) in zip(tt, indexset)))
end

function (obj::TensorTrainFit{ValueType})(x::Vector{ValueType}) where {ValueType}
    tensors = to_tensors(obj, x)
    return sum((abs2(_evaluate(tensors, indexset) - obj.values[i]) for (i, indexset) in enumerate(obj.indexsets)))
end


function fulltensor(obj::TensorTrain{T,N})::Array{T} where {T,N}
    sitedims_ = sitedims(obj)
    localdims = collect(prod.(sitedims_))
    result::Matrix{T} = reshape(obj.sitetensors[1], localdims[1], :)
    leftdim = localdims[1]
    for l in 2:length(obj)
        nextmatrix = reshape(
            obj.sitetensors[l], size(obj.sitetensors[l], 1), localdims[l] * size(obj.sitetensors[l])[end])
        leftdim *= localdims[l]
        result = reshape(result * nextmatrix, leftdim, size(obj.sitetensors[l])[end])
    end
    returnsize = collect(Iterators.flatten(sitedims_))
    return reshape(result, returnsize...)
end

function IMPO(R::Int)
    return TensorTrain{Float64, 4}([reshape([1.,0.,0.,1.], 1,2,2,1) for _ in 1:R])
end


@doc raw"""
    function mposwapindex(
        tt::AbstractTensorTrain{V}, theory_dims::Vector{Int}, permute::Vector{Int}, output_dims::Vector{Int}
    )

Swaps the physical indices of a tensor train. Works only on MPOs with 4 indices (left, physical, right, physical).
    

Arguments:
- `tt`: Tensor train.
- `input_dims`: Physical indices of the input tensor train.
- `permute`: Permutation of the indices.
- `output_dims`: Physical indices of the output tensor train.
"""
function mposwapindex(tt::AbstractTensorTrain{V}, input_dims::Vector{Int}, permute::Vector{Int}, output_dims::Vector{Int}) where V
    tensors = Vector{Array{V, 4}}(undef, length(tt))
    for b in 1:length(tt)
        site = reshape(tt.sitetensors[b], (size(tt.sitetensors[b])[1], input_dims..., size(tt.sitetensors[b])[end]))
        site = permutedims(site, (1, (permute.+1)... ,length(permute)+2,))
        sitedim = reshape(site, (size(tt.sitetensors[b])[1], output_dims..., size(tt.sitetensors[b])[end]))
        tensors[b] = sitedim
    end
    wow = TensorTrain{V, 4}(tensors)
end

function diagonalizemps(tt::TensorTrain{V,3})::TensorTrain{V,4} where {V}
    mpo = Vector{Array{V,4}}(undef, length(tt))
    for i in 1:length(tt)
        T = tt.sitetensors[i]
        a, b, c = size(T)
        mpo[i] = zeros(V, a, b, b, c)
        for k in 1:b
            mpo[i][:,k,k,:] .= T[:,k,:]
        end
    end
    return TensorTrain{V,4}(mpo)
end

function extractdiagonal(tt::TensorTrain{V,4})::TensorTrain{V,3} where {V}
    mps = Vector{Array{V,3}}(undef, length(tt))
    for i in 1:length(mps)
        T = mps.sitetensors[i]
        a, b1, b2, c = size(T)
        @assert b1 == b2 "The MPO is not diagonal in the physical indices."
        mps[i] = zeros(V, a, b1, c)
        for k in 1:b1
            mps[i][:,k,:] .= T[:,k,k,:]
        end
    end
    return TensorTrain{V,3}(mps)
end
