using Test
import TensorCrossInterpolation as TCI

@testset "Cached Function" begin
    cf = TCI.CachedFunction{Float64}(sqrt)
    @test cf.f == sqrt
    @test cf.d == Dict{UInt,Float64}()

    @test cf(5) == sqrt(5)
    @test cf.d == Dict(hash(5) => sqrt(5))

    for i in 1:4
        @test cf(i) == sqrt(i)
    end

    @test cf.d == Dict(
        hash(1) => sqrt(1),
        hash(2) => sqrt(2),
        hash(3) => sqrt(3),
        hash(4) => sqrt(4),
        hash(5) => sqrt(5))
end