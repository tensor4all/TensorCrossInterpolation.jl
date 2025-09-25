using Test

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: evaluate
using ITensors
using ITensorMPS

@testset "TCIITensorConversion.jl" begin
    @testset "TT to MPS conversion" begin
        tt = TCI.TensorTrain([rand(1, 4, 4), rand(4, 4, 2), rand(2, 4, 7), rand(7, 4, 1)])
        mps = ITensorMPS.MPS(tt)
        @test ITensorMPS.linkdims(mps) == [4, 2, 7]
    end

    @testset "TT to MPO conversion and back" begin
        tt = TCI.TensorTrain([rand(1, 4, 3, 4), rand(4, 2, 4, 2), rand(2, 5, 1, 7), rand(7, 9, 4, 1)])
        mpo = ITensorMPS.MPO(tt)
        @test ITensorMPS.linkdims(mpo) == [4, 2, 7]
        @test dim.(siteinds(mpo)[1]) == [4, 3]
        @test dim.(siteinds(mpo)[2]) == [2, 4]
        @test dim.(siteinds(mpo)[3]) == [5, 1]
        @test dim.(siteinds(mpo)[4]) == [9, 4]

        tt2 = TCI.TensorTrain{Float64,4}(mpo)
        @test all(tt2[i] == tt[i] for i in 1:length(tt))

        sites = reverse.(siteinds(mpo))
        ttreverse = TCI.TensorTrain{Float64,4}(mpo, sites=sites)
        @test all(permutedims(ttreverse[i], [1, 3, 2, 4]) == tt[i] for i in 1:length(tt))
    end

    function binary_representation(index::Int, numdigits::Int)
        return Int[(index & (1 << (numdigits - i))) != 0 for i in 1:numdigits]
    end

    @testset "mps util" begin
        @testset "one mps evaluation" begin
            m = 5
            mps = MPS(m)
            linkindices = [Index(2, "link") for i in 1:(m-1)]
            mps[1] = ITensor(
                ones(2, 2),
                Index(2, "site"),
                linkindices[1]
            )
            for i in 2:(m-1)
                mps[i] = ITensor(
                    ones(2, 2, 2),
                    linkindices[i-1],
                    Index(2, "site"),
                    linkindices[i]
                )
            end
            mps[m] = ITensor(
                ones(2, 2),
                linkindices[m-1],
                Index(2, "site")
            )

            for i in 1:(2^m)
                q = binary_representation(i, m) .+ 1
                @test evaluate(mps, siteinds(mps), q) == 2^(m - 1)
                @test evaluate(mps, collect(zip(siteinds(mps), q))) == 2^(m - 1)
            end
        end

        @testset "delta mps evaluation" begin
            m = 5
            mps = MPS(m)
            linkindices = [Index(2, "link") for i in 1:(m-1)]
            mps[1] = ITensor(
                [i == j for i in 0:1, j in 0:1],
                Index(2, "site"),
                linkindices[1]
            )
            for i in 2:(m-1)
                mps[i] = ITensor(
                    [i == j && j == k for i in 0:1, j in 0:1, k in 0:1],
                    linkindices[i-1],
                    Index(2, "site"),
                    linkindices[i]
                )
            end
            mps[m] = ITensor(
                [i == j for i in 0:1, j in 0:1],
                linkindices[m-1],
                Index(2, "site")
            )

            for i in 1:(2^m)
                q = binary_representation(i, m) .+ 1
                @test evaluate(mps, siteinds(mps), q) == all(q .== first(q))
                @test (
                    evaluate(mps, collect(zip(siteinds(mps), q))) ==
                    all(q .== first(q))
                )
            end
        end
    end
end
