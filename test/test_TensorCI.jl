import TensorCrossInterpolation: IndexSet, MultiIndex

@testset "TensorCI" begin
    @testset "trivial MPS" begin
        n = 3
        f(v) = 1

        cf = TensorCrossInterpolation.CachedFunction(
            f,
            Dict(fill(1, n) => 1.0))

        tci = TensorCI(cf, fill(2, n))

        @test tci.f.f == f

        for i in 1:n
            @test isempty(tci.Iset[i])
            @test isempty(tci.Jset[i])
            @test tci.localSet[i] == [1, 2]
            @test tci.T[i] == zeros(0, 2, 0)
            @test tci.P[i] == zeros(0, 0)
            @test isempty(tci.Pi_Iset[i])
            @test isempty(tci.Pi_Jset[i])
        end

        for i in 1:n-1
            @test tci.Pi[i] == zeros(0, 0)
            @test tci.pivot_errors[i] == Inf
        end

        tci = TensorCI(cf, fill(2, n), fill(1, n))

        for i in 1:n
            @test tci.Iset[i].from_int == [fill(1, i - 1)]
            @test tci.Jset[i].from_int == [fill(1, n - i)]
            @test tci.localSet[i] == [1, 2]
            @test tci.T[i] == ones(1, 2, 1)
            @test tci.P[i] == ones(1, 1)
            @test tci.Pi_Iset[i].from_int == [[fill(1, i - 1)..., k] for k in 1:2]
            @test tci.Pi_Jset[i].from_int == [[k, fill(1, n - i)...] for k in 1:2]
        end

        for i in 1:n-1
            @test tci.Pi[i] == ones(2, 2)
        end

        # Because the MPS is trivial, no new pivot should be added.
        for i in 1:n-1
            TensorCrossInterpolation.addPivotAt!(tci, i, 1e-8)
        end

        for i in 1:n
            @test length(tci.Iset[i]) == 1
            @test length(tci.Jset[i]) == 1
            @test tci.localSet[i] == [1, 2]
            @test tci.T[i] == ones(1, 2, 1)
            @test tci.P[i] == ones(1, 1)
            @test length(tci.Pi_Iset[i]) == 2
            @test length(tci.Pi_Jset[i]) == 2
        end

        for i in 1:n-1
            @test tci.Pi[i] == ones(2, 2)
        end
    end
end
