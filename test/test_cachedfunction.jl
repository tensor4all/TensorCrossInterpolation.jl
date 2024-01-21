using Test
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: BatchEvaluator, MultiIndex

struct TestF <: TCI.BatchEvaluator{Float64}
end

function (f::TestF)(x::T) where {T}
    return one(T)
end

function (obj::TestF)(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{Float64,M + 2} where {M}
    if length(leftindexset) * length(rightindexset) == 0
        return zeros{Float64}(0)
    end

    nl = length(first(leftindexset))
    nr = length(first(rightindexset))
    L = nl + nr + M

    return ones(Float64, length(leftindexset), fill(2, M)..., length(rightindexset))
end


struct TestFunction{T} <: TCI.BatchEvaluator{T}
    localdims::Vector{Int}
    function TestFunction{T}(localdims) where {T}
        new{T}(localdims)
    end
end

function (obj::TestFunction{T})(indexset)::T where {T}
    return sum(indexset)
end

function (obj::TestFunction{T})(leftindexset, rightindexset, ::Val{M})::Array{T,M + 2} where {T,M}
    nl = length(first(leftindexset))
    result = [sum(vcat(l, collect(c), r)) for l in leftindexset, c in Iterators.product((1:d for d in obj.localdims[nl+1:nl+M])...), r in rightindexset]
    return reshape(result, length(leftindexset), obj.localdims[nl+1:nl+M]..., length(rightindexset))
end


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

    #==
    @testset "cache(batcheval)" begin
        localdims = [2, 2, 2, 2, 2]
        cf = TCI.CachedFunction{Float64}(TestF(), localdims)
        x = [1, 1, 1, 1, 1]
        @test cf(x) == 1.0
    end

    @testset "cache(non-batcheval)" begin
        localdims = [2, 2, 2, 2, 2]
        cf = TCI.CachedFunction{Float64}(x->1.0, localdims)
        x = [1, 1, 1, 1, 1]
        @show cf(x)
        @test cf(x) == 1.0
        @time cf([[1,1]], [[1,1]], Val(1))
        @time cf([[1,1]], [[1,1]], Val(1))
    end

    @testset "cache(non-batcheval)" begin
    end
    ==#
    @testset "cache(batcheval)" for T in [Float64, ComplexF64]
        localdims = [2, 2, 2, 2, 2]
        leftindexset = [[1, 1] for _ in 1:100]
        rightindexset = [[1, 1] for _ in 1:100]

        f = TCI.CachedFunction{T}(TestFunction{T}(localdims), localdims)
        @assert TCI.isbatchevaluable(f)
        result = TCI._batchevaluate_dispatch(T, f, localdims, leftindexset, rightindexset, Val(1))
        ref = [sum(vcat(l, c, r)) for l in leftindexset, c in 1:localdims[3], r in rightindexset]

        @test result ≈ ref
    end

    @testset "many bits" begin
        f(x) = 1.0
        nint = 4
        N = 64 * nint
        cf = TCI.CachedFunction{Float64,BigInt}(f, fill(2, N))
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
        cf = TCI.CachedFunction{ComplexF64}(x->1.0, localdims)
        wrongindexset = fill(1, 2*L)
        @test_throws ErrorException TCI._key(cf, wrongindexset)
    end
end
