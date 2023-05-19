using Test
import TensorCrossInterpolation as TCI

@testset "Cached Function" begin
    @testset "cache" begin
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

    @testset "many bits" begin
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

    @testset "key collision and memory overhead" begin
        T = ComplexF64
        nbit = 36
        nsample = Int(1e+5)
        f(x)::T = 1.0
        cf = TCI.CachedFunction{T}(f, fill(2, nbit))
        d = Dict{Vector{Int},T}()
        for i in 1:nsample
            x = tobins(i, nbit)
            d[x] = cf(x)
        end
        @test length(cf.d) == nsample
        databytes = sizeof(T) * nsample
    
        # Overhead must be small enough
        @test 24 < Base.summarysize(d)/databytes < 25
        @test 5 < Base.summarysize(cf.d)/databytes < 7
    end
end
