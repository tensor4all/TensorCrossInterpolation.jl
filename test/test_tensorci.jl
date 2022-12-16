using TensorCrossInterpolation
using Test
using LinearAlgebra
import TensorCrossInterpolation: IndexSet, MultiIndex, CachedFunction, TensorCI, linkdims

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
            @test tci.localset[i] == [1, 2]
            @test tci.T[i] == zeros(0, 2, 0)
            @test tci.P[i] == zeros(0, 0)
            @test isempty(tci.PiIset[i])
            @test isempty(tci.PiJset[i])
        end

        for i in 1:n-1
            @test tci.Pi[i] == zeros(0, 0)
            @test tci.pivoterrors[i] == Inf
        end

        tci = TensorCI(cf, fill(2, n), fill(1, n))

        for i in 1:n
            @test tci.Iset[i].fromint == [fill(1, i - 1)]
            @test tci.Jset[i].fromint == [fill(1, n - i)]
            @test tci.localset[i] == [1, 2]
            @test tci.T[i] == ones(1, 2, 1)
            @test tci.P[i] == ones(1, 1)
            @test tci.PiIset[i].fromint == [[fill(1, i - 1)..., k] for k in 1:2]
            @test tci.PiJset[i].fromint == [[k, fill(1, n - i)...] for k in 1:2]
        end

        for i in 1:n-1
            @test tci.Pi[i] == ones(2, 2)
        end

        # Because the MPS is trivial, no new pivot should be added.
        for i in 1:n-1
            addpivot!(tci, i, 1e-8)
        end

        for i in 1:n
            @test length(tci.Iset[i]) == 1
            @test length(tci.Jset[i]) == 1
            @test tci.localset[i] == [1, 2]
            @test tci.T[i] == ones(1, 2, 1)
            @test tci.P[i] == ones(1, 1)
            @test length(tci.PiIset[i]) == 2
            @test length(tci.PiJset[i]) == 2
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

        @test linkdims(tci) == ones(n - 1)
        @test rank(tci) == 1

        for p in 1:n-1
            addpivot!(tci, p, 1e-8)
        end
        @test linkdims(tci) == fill(2, n - 1)
        @test rank(tci) == 2

        for iter in 3:8
            for p in 1:n-1
                addpivot!(tci, p, 1e-8)
            end
            @test linkdims(tci) == fill(iter, n - 1)
            @test rank(tci) == iter
        end

        tci2, ranks, errors = crossinterpolate(
            f,
            fill(10, n),
            ones(Int, n);
            tolerance=1e-8,
            maxiter=8,
            sweepstrategy=SweepStrategies.forward
        )

        @test linkdims(tci) == linkdims(tci2)
        @test rank(tci) == rank(tci2)

        tci3, ranks, errors = crossinterpolate(
            f,
            fill(3, n),
            ones(Int, n);
            tolerance=1e-12,
            maxiter=200
        )

        @test all(tci3.pivoterrors .<= 1e-12)
        @test all(linkdims(tci3) .<= 200)
        @test rank(tci3) <= 200

        tt3 = tensortrain(tci3)

        for v in Iterators.product([1:3 for p in 1:n]...)
            value = evaluate(tci3, [i for i in v])
            @test value ≈ prod([tt3[p][:, v[p], :] for p in eachindex(v)])[1]
            @test value ≈ f(v)
        end
    end

    @testset "optfirstpivot" begin
        f(v) = 2^2 * (v[3]-1) + 2^1 * (v[2]-1) + 2^0 * (v[1]-1)
        localdims = [2, 2, 2]
        firstpivot = [1, 1, 1]
        firstpivotval = f(firstpivot)
        cf = CachedFunction(f, Dict(firstpivot => firstpivotval))

        pivot = optfirstpivot(cf, localdims, firstpivot)

        @test pivot == [2, 2, 2]
    end
end
