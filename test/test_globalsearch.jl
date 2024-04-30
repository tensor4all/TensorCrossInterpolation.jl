using Test
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: crossinterpolate2, MultiIndex
import Random
import QuanticsGrids as QD

@testset "globalsearch" begin
    Random.seed!(1240)
    R = 20
    abstol = 1e-10

    grid = QD.DiscretizedGrid{1}(R, (0.0,), (1.0,))

    fx(x) = exp(-x) + 1e-3 * sin(1000 * x)
    f(bitlist::MultiIndex) = fx(QD.quantics_to_origcoord(grid, bitlist)[1])

    abstol = 1e-4
    localdims = fill(2, R)
    firstpivots = [ones(Int, R), vcat(1, fill(2, R - 1))]
    tci, ranks, errors = crossinterpolate2(
        Float64,
        f,
        localdims,
        firstpivots;
        tolerance=abstol,
        maxbonddim=1,
        verbosity=1,
        normalizeerror=false,
    )


    pivoterrors = TCI.estimatetrueerror(TCI.TensorTrain(tci), f)

    errors = [e for (_, e) in pivoterrors] 
    @test all([abs(f(p) - tci(p)) for (p, _) in pivoterrors] .== errors)
    @test all(errors[1:end-1] .>= errors[2:end]) # check if errors are sorted in descending order
end