using Test
import TensorCrossInterpolation as TCI
import BitIntegers


@testset "Cached Function" begin
    @testset "cache" for T in [Float64, ComplexF64]
        f(x) = 2 * (x[1] - 1) + (x[2] - 1)
        cf = TCI.CachedFunction{T}(f, [4, 2])

        @test cf.f == f
        for i in 1:4, j in 1:2
            x = [i, j]
            @test cf(x) == f(x)
            @test TCI._key(cf, x) ∈ keys(cf.cache)
            @test cf(x) == f(x) # Second access
        end
    end

    @testset "many bits" begin
        f(x) = 1.0
        nint = 4
        N = 64 * nint
        cf = TCI.CachedFunction{Float64,BitIntegers.UInt512}(f, fill(2, N))
        x = ones(Int, N)
        @test cf(x) == 1.0
        @test TCI._key(cf, x) == 0
    end

    function tobins(i, nbit)
        @assert 1 ≤ i ≤ 2^nbit
        mask = 1 << (nbit - 1)
        bin = ones(Int, nbit)
        for n in 1:nbit
            bin[n] = (mask & (i - 1)) >> (nbit - n) + 1
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
        @test length(cf.cache) == nsample
        databytes = sizeof(T) * nsample

        # Overhead must be small enough
        @test 24 < Base.summarysize(d) / databytes < 25
        @test 5 < Base.summarysize(cf.cache) / databytes < 7
    end

    @testset "computekey boundary check" begin
        L = 40
        localdims = fill(2, L)
        indexsets = [rand(1:d) for d in localdims]
        cf = TCI.CachedFunction{ComplexF64}(x -> 1.0, localdims)
        wrongindexset = fill(1, 2 * L)
        @test_throws ErrorException TCI._key(cf, wrongindexset)
    end

    @testset "encode and decode cachekey" begin
        localdims = [2, 3, 4]
        cf = TCI.CachedFunction{ComplexF64}(x -> Float64(sum(x)), localdims)
        for i1 in 1:localdims[1], i2 in 1:localdims[2], i3 in 1:localdims[3]
            x = [i1, i2, i3]
            cf(x) # fill cache
            key = TCI.encodecachekey(cf, x)
            @test TCI.decodecachekey(cf, key) == x
        end

        cachedata = TCI.cachedata(cf)
        for (x, v) in cachedata
            @test cf(x) == v
        end
    end
end
