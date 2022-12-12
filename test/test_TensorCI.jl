using TensorCrossInterpolation
using Test
using LinearAlgebra
import TensorCrossInterpolation: IndexSet, MultiIndex, CachedFunction,
    addPivotAt!

@testset "TensorCI" begin
    @testset "trivial MPS" begin
        n = 5
        f(v) = 1

        cf = CachedFunction(
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
            addPivotAt!(tci, i, 1e-8)
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

    @testset "Lorentz MPS" for coeff in [1.0, 1.0im]
        n = 5
        f(v) = coeff ./ (sum(v .^ 2) + 1)

        ValueType = typeof(coeff)

        tci = TensorCI(
            CachedFunction{Vector{Int},ValueType}(f),
            fill(10, n),
            ones(Int, n)
        )

        @test rank(tci) == ones(n - 1)

        for p in 1:n-1
            addPivotAt!(tci, p, 1e-8)
        end
        @test rank(tci) == fill(2, n - 1)

        for iter in 3:8
            for p in 1:n-1
                addPivotAt!(tci, p, 1e-8)
            end
            @test rank(tci) == fill(iter, n - 1)
        end

        tci2, ranks, errors = cross_interpolate(
            f,
            fill(10, n),
            ones(Int, n);
            tolerance=1e-8,
            maxiter=8,
            sweepstrategy=SweepStrategies.forward
        )

        @test rank(tci) == rank(tci2)

        tci3, ranks, errors = cross_interpolate(
            f,
            fill(3, n),
            ones(Int, n);
            tolerance=1e-12,
            maxiter=200
        )

        @test all(tci3.pivot_errors .<= 1e-12)
        @test all(rank(tci3) .<= 200)

        tt3 = tensortrain(tci3)

        for v in Iterators.product([1:3 for p in 1:n]...)
            value = evaluateTCI(tci3, [i for i in v])
            @test value ≈ prod([tt3[p][:, v[p], :] for p in eachindex(v)])[1]
            @test value ≈ f(v)
        end
    end
end
