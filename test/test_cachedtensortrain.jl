using Test
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: rank, linkdims, TensorCI2, updatepivots!, addglobalpivots1sitesweep!, MultiIndex, evaluate, crossinterpolate2, pivoterror, tensortrain
import Random
import QuanticsGrids as QD

@testset "CachedTensorTrain" begin
    @testset "TTCache" begin
        ValueType = Float64

        N = 4
        bonddims = [1, 2, 3, 2, 1]
        @assert length(bonddims) == N + 1
        localdims = [2, 3, 3, 2]

        tt = TCI.TensorTrain{ValueType,3}([rand(bonddims[n], localdims[n], bonddims[n+1]) for n in 1:N])
        ttc = TCI.TTCache(tt)

        @test [tt(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))] ≈ [ttc(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))]

        leftindexset = [[1]]
        rightindexset = [[1]]

        # Without projection
        ttc_batch = ttc(leftindexset, rightindexset, Val(2))
        @test size(ttc_batch) == (1, 3, 3, 1)
        @test [tt([1, Tuple(i)..., 1]) for i in CartesianIndices(Tuple(localdims[2:end-1]))] ≈
              [ttc_batch[1, collect(Tuple(i))..., 1] for i in CartesianIndices(Tuple(localdims[2:end-1]))]

        # With projoection
        ttc_batchproj = TCI.batchevaluate(ttc, leftindexset, rightindexset, Val(2), [[1], [0]])
        @test vec([tt([1, 1, i, 1]) for i in 1:localdims[3]]) ≈ vec(ttc_batchproj)
    end

    @testset "TTCache (multi site indices)" begin
        ValueType = Float64

        N = 4
        bonddims = [1, 2, 3, 2, 1]
        @assert length(bonddims) == N + 1
        localdims = [4, 4, 4, 4]
        sitedims = [[2, 2] for _ in 1:N]

        tt = TCI.TensorTrain{ValueType,3}([rand(bonddims[n], localdims[n], bonddims[n+1]) for n in 1:N])
        ttc = TCI.TTCache(tt, sitedims)

        ref = [tt(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))]
        rest = [ttc([[i1, j1], [i2, j2], [i3, j3], [i4, j4]])
                for i1 in 1:2, j1 in 1:2, i2 in 1:2, j2 in 1:2, i3 in 1:2, j3 in 1:2, i4 in 1:2, j4 in 1:2]
        @test vec(ref) ≈ vec(rest)

        leftindexset = [[1]]
        rightindexset = [[1]]

        # Without projection
        ttc_batch = ttc(leftindexset, rightindexset, Val(2))
        @test size(ttc_batch) == (1, 4, 4, 1)

        @test [tt([1, Tuple(i)..., 1]) for i in CartesianIndices(Tuple(localdims[2:end-1]))] ≈
              [ttc_batch[1, collect(Tuple(i))..., 1] for i in CartesianIndices(Tuple(localdims[2:end-1]))]

        # With projoection
        ttc_batchproj = TCI.batchevaluate(ttc, leftindexset, rightindexset, Val(2), [[1, 1], [0, 0]])
        @test vec([tt([1, 1, i, 1]) for i in 1:localdims[3]]) ≈ vec(ttc_batchproj)
    end
end
