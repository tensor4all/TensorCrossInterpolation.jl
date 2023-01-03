using Test
import TensorCrossInterpolation as TCI

@testset "Cached Function" begin
    f(x) = 2*(x[1]-1) + (x[2]-1)
    cf = TCI.CachedFunction{Float64}(f, [2, 2])

    @test cf.f == f
    for i in 1:2, j in 1:2
        x =  [i, j]
        @test cf(x) == f(x)
        @test TCI._key(cf, x) ∈ keys(cf.d)
        @test cf(x) == f(x) # Second access
    end
end

@testset "Cached Function (many bits)" begin
    f(x) = 1.0
    nint = 4
    N = 64*nint
    cf = TCI.CachedFunction{Float64}(f, fill(2, N))
    x = ones(Int, N)
    @test cf(x) == 1.0
    @test TCI._key(cf, x).size == nint
end