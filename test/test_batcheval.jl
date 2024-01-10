using Test
import TensorCrossInterpolation as TCI

@testset "batcheval" begin
    @testset "M=1" begin
        localdims = [2, 2, 2, 2, 2]
        leftindexset = [[1, 1] for _ in 1:100]
        rightindexset = [[1, 1] for _ in 1:100]

        f = x -> sum(x)
        result = TCI._batchevaluate_dispatch(Float64, f, localdims, leftindexset, rightindexset, Val(1))
        ref = [sum(vcat(l, c, r)) for l in leftindexset, c in 1:localdims[3], r in rightindexset]

        @test result ≈ ref
    end

    @testset "M=2" begin
        localdims = [2, 2, 2, 2, 2]
        leftindexset = [[1] for _ in 1:100]
        rightindexset = [[1, 1] for _ in 1:100]

        f = x -> sum(x)
        result = TCI._batchevaluate_dispatch(Float64, f, localdims, leftindexset, rightindexset, Val(2))
        ref = [sum(vcat(l, c, cp, r)) for l in leftindexset, c in 1:localdims[2], cp in 1:localdims[3], r in rightindexset]

        @test result ≈ ref
    end

    @testset "ThreadedBatchEvaluator" begin
        L = 20
        localdims = fill(2, L)
        f = x -> sum(x)

        f = TCI.ThreadedBatchEvaluator{Float64}(f, localdims)

        # Compute Pi tensor
        nl = 10
        nr = L - nl - 2

        # 20 left index sets, 20 right index sets
        leftindexset = [[rand(1:d) for d in localdims[1:nl]] for _ in 1:20]
        rightindexset = [[rand(1:d) for d in localdims[nl+3:end]] for _ in 1:20]

        result = f(leftindexset, rightindexset, Val(2))
        ref = [sum(vcat(l, c, cp, r)) for l in leftindexset, c in 1:localdims[nl+1], cp in 1:localdims[nl+2], r in rightindexset]

        @test result ≈ ref
    end
end