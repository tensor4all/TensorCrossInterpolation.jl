import TensorCrossInterpolation as TCI
using Random
using Zygote
using Optim

@testset "tensor train" begin
    g(v) = 1 / (sum(v .^ 2) + 1)
    localdims = (6, 6, 6, 6)
    tolerance = 1e-8
    tci, ranks, errors = TCI.crossinterpolate(Float64, g, localdims; tolerance=tolerance)
    tt = TCI.TensorTrain(tci)
    @test rank(tci) == rank(tt)
    @test TCI.linkdims(tci) == TCI.linkdims(tt)
    gsum = 0.0
    for i in CartesianIndices(localdims)
        @test TCI.evaluate(tci, i) ≈ TCI.evaluate(tt, i)
        @test tt(i) == TCI.evaluate(tt, i)
        functionvalue = g(Tuple(i))
        @test abs(TCI.evaluate(tt, i) - functionvalue) < tolerance
        gsum += functionvalue
    end
    @test gsum ≈ TCI.sum(tt)
end

@testset "batchevaluate" begin
    N = 4
    #bonddims = fill(3, N + 1)
    bonddims = [1, 2, 3, 2, 1]
    @assert length(bonddims) == N + 1
    #bonddims[1] = 1
    #bonddims[end] = 1
    A = TCI.TTCache([rand(bonddims[n], 2, bonddims[n+1]) for n in 1:N])

    leftindexset = [[1], [2]]
    rightindexset = [[1], [2]]

    result = A(leftindexset, rightindexset, Val(2))
    for cindex in [[1, 1], [1, 2]]
        for (il, lindex) in enumerate(leftindexset)
            for (ir, rindex) in enumerate(rightindexset)
                @test result[il, cindex..., ir] ≈ TCI.evaluate(A, vcat(lindex, cindex, rindex))
            end
        end
    end
end

function genindices(localdims::Vector{Int})::Vector{Vector{Int}}
    iter = vec(collect(Iterators.product([1:d for d in localdims]...)))
    return [collect(l) for l in iter]
end

@testset "batchevaluate2" begin
    N = 4
    bonddims = [1, 2, 3, 2, 1]
    @assert length(bonddims) == N + 1
    localdims = [2, 3, 3, 2]

    A = TCI.TTCache([rand(bonddims[n], localdims[n], bonddims[n+1]) for n in 1:N])

    for nleft in 0:N, nright in 0:N
        leftindexset = genindices(localdims[1:nleft])
        rightindexset = genindices(localdims[N-nright+1:N])

        ncent = N - nleft - nright
        if ncent < 0
            continue
        end

        result = A(leftindexset, rightindexset, Val(ncent))
        for cindex in genindices(localdims[nleft+1:nleft+ncent])
            for (il, lindex) in enumerate(leftindexset)
                for (ir, rindex) in enumerate(rightindexset)
                    @test result[il, cindex..., ir] ≈ TCI.evaluate(A, [lindex..., cindex..., rindex...], usecache=true)
                    @test result[il, cindex..., ir] ≈ TCI.evaluate(A, [lindex..., cindex..., rindex...], usecache=false)
                end
            end
        end
    end
end

@testset "ttfit" for T in [Float64, ComplexF64]
    Random.seed!(10)
    localdims = [2, 2, 2]
    linkdims = [1, 2, 3, 1]
    L = length(localdims)
    @assert L == length(linkdims) - 1
    tt0 = TCI.TensorTrain{T,3}([randn(T, linkdims[n], localdims[n], linkdims[n+1]) for n in 1:L])

    indexsets = [[1, 1, 1], [2, 2, 2]]
    values = randn(T, length(indexsets))

    # Loss function for fitting
    ttfit = TCI.TensorTrainFit{T}(indexsets, values, tt0)

    # Convert an intial tensor train to a flatten vector
    x0::Vector{T} = TCI.flatten(tt0)

    loss(x) = ttfit(x)

    function dloss!(g, x)
        g .= Zygote.gradient(ttfit, x)[1]
    end

    xopt = Optim.minimizer(optimize(loss, dloss!, x0, LBFGS()))
    ttopt = TCI.TensorTrain{T,3}(TCI.to_tensors(ttfit, xopt))
    @test [TCI.evaluate(ttopt, idx) for (idx, v) in zip(indexsets, values)] ≈ values
end

@testset "tensor train addition" for T in [Float64, ComplexF64]
    Random.seed!(10)
    localdims = [2, 2, 2]
    linkdims = [1, 2, 3, 1]
    L = length(localdims)
    @assert L == length(linkdims) - 1
    tt1 = TCI.TensorTrain{T,3}([randn(T, linkdims[n], localdims[n], linkdims[n+1]) for n in 1:L])
    tt2 = TCI.TensorTrain{T,3}([randn(T, linkdims[n], localdims[n], linkdims[n+1]) for n in 1:L])

    ttadd = TCI.add(tt1, tt2)
    indices = [[i, j, k] for i in 1:2, j in 1:2, k in 1:2]
    @test ttadd.(indices) ≈ [tt1(v) + tt2(v) for v in indices]

    ttshort = TCI.TensorTrain{T,3}([randn(T, linkdims[n], localdims[n], linkdims[n+1]) for n in 1:L-1])
    @test_throws DimensionMismatch TCI.add(tt1, ttshort)

    ttmultileg = TCI.TensorTrain{T,4}([randn(T, linkdims[n], localdims[n], localdims[n], linkdims[n+1]) for n in 1:L-1])
    @test_throws DimensionMismatch TCI.add(tt1, ttmultileg)
end
