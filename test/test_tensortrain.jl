import TensorCrossInterpolation as TCI

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
