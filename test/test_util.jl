using Test
import TensorCrossInterpolation: maxabs

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
