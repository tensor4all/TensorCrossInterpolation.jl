function binary_representation(index::Int, numdigits::Int)
    return Int[(index & (1 << (numdigits - i))) != 0 for i in 1:numdigits]
end

@testset "mps util" begin
    @testset "one mps evaluation" begin
        m = 5
        mps = ITensors.MPS(m)
        linkindices = [ITensors.Index(2, "link") for i in 1:(m-1)]
        mps[1] = ITensors.ITensor(
            ones(2, 2),
            ITensors.Index(2, "site"),
            linkindices[1]
        )
        for i in 2:(m-1)
            mps[i] = ITensors.ITensor(
                ones(2, 2, 2),
                linkindices[i-1],
                ITensors.Index(2, "site"),
                linkindices[i]
            )
        end
        mps[m] = ITensors.ITensor(
            ones(2, 2),
            linkindices[m-1],
            ITensors.Index(2, "site")
        )

        for i in 1:(2^m)
            q = binary_representation(i, m) .+ 1
            @test TCI.evaluate(mps, ITensors.siteinds(mps), q) == 2^(m - 1)
            @test TCI.evaluate(mps, collect(zip(ITensors.siteinds(mps), q))) == 2^(m - 1)
        end
    end

    @testset "delta mps evaluation" begin
        m = 5
        mps =ITensors.MPS(m)
        linkindices = [ITensors.Index(2, "link") for i in 1:(m-1)]
        mps[1] = ITensors.ITensor(
            [i == j for i in 0:1, j in 0:1],
            ITensors.Index(2, "site"),
            linkindices[1]
        )
        for i in 2:(m-1)
            mps[i] = ITensors.ITensor(
                [i == j && j == k for i in 0:1, j in 0:1, k in 0:1],
                linkindices[i-1],
                ITensors.Index(2, "site"),
                linkindices[i]
            )
        end
        mps[m] = ITensors.ITensor(
            [i == j for i in 0:1, j in 0:1],
            linkindices[m-1],
            ITensors.Index(2, "site")
        )

        for i in 1:(2^m)
            q = binary_representation(i, m) .+ 1
            @test TCI.evaluate(mps, ITensors.siteinds(mps), q) == all(q .== first(q))
            @test (
                TCI.evaluate(mps, collect(zip(ITensors.siteinds(mps), q))) ==
                all(q .== first(q))
            )
        end
    end
end
