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
end