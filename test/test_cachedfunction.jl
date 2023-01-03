using Test
import TensorCrossInterpolation as TCI

@testset "Cached Function" begin
    f(x) = 2*(x[1]-1) + (x[2]-1)
    cf = TCI.CachedFunction{Float64}(f, [4, 2])

    @test cf.f == f
    for i in 1:4, j in 1:2
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

function tobins(i, nbit)
    @assert 1 ≤ i ≤ 2^nbit
    mask = 1 << (nbit-1)
    bin = ones(Int, nbit)
    for n in 1:nbit
        bin[n] = (mask & (i-1)) >> (nbit-n) + 1
        mask = mask >> 1
    end
    return bin
end

@testset "Cached Function (key collision)" begin
    T = ComplexF64
    nbit = 16
    f(x)::T = 1.0
    cf = TCI.CachedFunction{T}(f, fill(2, nbit))
    for i in 1:2^nbit
        cf(tobins(i, nbit))
    end
    @test length(cf.d) == 2^nbit
    databytes = sizeof(T) * 2^nbit

    # Overhead must be small enough
    @test Base.summarysize(cf.d)/databytes < 8
end
