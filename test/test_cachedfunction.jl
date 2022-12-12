@testset "Cached Function" begin
    cf = TensorCrossInterpolation.CachedFunction{Int64,Float64}(sqrt)
    @test cf.f == sqrt
    @test cf.d == Dict{Int64,Float64}()

    @test cf(5) == sqrt(5)
    @test cf.d == Dict(5 => sqrt(5))

    for i in 1:4
        @test cf(i) == sqrt(i)
    end

    @test cf.d == Dict(
        1 => sqrt(1),
        2 => sqrt(2),
        3 => sqrt(3),
        4 => sqrt(4),
        5 => sqrt(5))
end