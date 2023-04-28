using Test
import TensorCrossInterpolation: maxabs, optfirstpivot, pushunique!

@testset "Test updatemax" begin
    s = 1.0

    @test maxabs(s, []) == 1.0

    u = [
        0.11892436782208138,
        -0.5312119179782191,
        0.15328557552100353,
        0.9343319135479445,
        -0.04286173791053016
    ]
    @test maxabs(s, u) == 1.0

    v = [
        -7.512961239635482,
        -0.644254782278785,
        1.1242493861712504,
        6.5875869748554186,
        -5.400768247401216
    ]
    @test maxabs(s, v) == 7.512961239635482
end

@testset "optfirstpivot" begin
    f(v) = 2^2 * (v[3] - 1) + 2^1 * (v[2] - 1) + 2^0 * (v[1] - 1)
    localdims = [2, 2, 2]
    firstpivot = [1, 1, 1]

    pivot = optfirstpivot(f, localdims, firstpivot)

    @test pivot == [2, 2, 2]
end

@testset "pushunique" begin
    v = [9, 29, 4, 5]
    pushunique!(v, 10)
    @test v == [9, 29, 4, 5, 10]
    pushunique!(v, 10)
    @test v == [9, 29, 4, 5, 10]
    pushunique!(v, 2, 3)
    @test v == [9, 29, 4, 5, 10, 2, 3]
    pushunique!(v, 29, 8, 4, 5)
    @test v == [9, 29, 4, 5, 10, 2, 3, 8]
end
