import TensorCrossInterpolation as TCI
using Random

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

@testset "tensor train (TCI2)" begin
    g(v) = 1 / (sum(v .^ 2) + 1)
    localdims = (6, 6, 6, 6)
    L = length(localdims)
    tolerance = 1e-8
    tci, ranks, errors = TCI.crossinterpolate2(Float64, g, localdims; tolerance=tolerance)

    tt = TCI.tensortrain(tci, g)
    tt2 = TCI.tensortrain(tci)

    reconst = [TCI.evaluate(tt, collect(i)) for i in Iterators.product((1:localdims[n] for n in 1:L)...)]
reconst2 = [TCI.evaluate(tt2, collect(i)) for i in Iterators.product((1:localdims[n] for n in 1:L)...)]

    @test reconst ≈ reconst2
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

    result = TCI.batchevaluate(A, leftindexset, rightindexset, Val(2))
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

        result = TCI.batchevaluate(A, leftindexset, rightindexset, Val(ncent))
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