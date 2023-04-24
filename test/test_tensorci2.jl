using Test
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: rank, linkdims

@testset "TensorCrossInterpolation2" begin
    @testset "kronecker util function" begin
        multiset = [collect(1:5) for _ in 1:5]
        localset = collect(10:14)

        c = TCI.kronecker(multiset, localset)
        for (i, ci) in enumerate(c)
            @test ci[1:5] == collect(1:5)
            @test ci[6] in localset
        end

        d = TCI.kronecker(localset, multiset)
        for (i, di) in enumerate(d)
            @test di[1] in localset
            @test di[2:6] == collect(1:5)
        end
    end

    @testset "trivial MPS" begin
        n = 5
        f(v) = sum(v) * 0.5

        tci = TCI.TensorCI2{Float64}(fill(2, n))
        @test length(tci) == n
        @test rank(tci) == 0
        @test linkdims(tci) == fill(0, n-1)
        @test tci.localset == fill([1, 2], n)
        for i in 1:n
            @test isempty(tci.Iset[i])
            @test isempty(tci.Jset[i])
        end

        tci = TCI.TensorCI2{Float64}(f, fill(2, n))
        @test length(tci) == n
        @test rank(tci) == 0
        @test linkdims(tci) == fill(0, n)
        @test tci.localset == fill([1, 2], n)
        for i in 1:n
            @test tci.Iset[i].fromint == [fill(1, i - 1)]
            @test tci.Jset[i].fromint == [fill(1, n - i)]
        end
    end
end
