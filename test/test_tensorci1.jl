using TensorCrossInterpolation
using Test
using LinearAlgebra
import TensorCrossInterpolation: IndexSet, MultiIndex, CachedFunction, TensorCI1, linkdims, addpivot!, addglobalpivot!, evaluate

@testset "TensorCI1" begin
    @testset "trivial MPS" begin
        n = 5
        f(v) = 1

        tci = TensorCI1{Int}(fill(2, n))

        for i in 1:n
            @test isempty(tci.Iset[i])
            @test isempty(tci.Jset[i])
            @test tci.T[i] == zeros(0, 2, 0)
            @test tci.P[i] == zeros(0, 0)
            @test isempty(tci.PiIset[i])
            @test isempty(tci.PiJset[i])
        end

        for i in 1:n-1
            @test tci.Pi[i] == zeros(0, 0)
            @test tci.pivoterrors[i] == Inf
        end

        tci = TensorCI1{Int}(f, fill(2, n), fill(1, n))

        for i in 1:n
            @test tci.Iset[i].fromint == [fill(1, i - 1)]
            @test tci.Jset[i].fromint == [fill(1, n - i)]
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
            addpivot!(tci, i, f, 1e-8)
        end

        for i in 1:n
            @test length(tci.Iset[i]) == 1
            @test length(tci.Jset[i]) == 1
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

        tci = TensorCI1{ValueType}(
            f,
            fill(10, n),
            ones(Int, n)
        )

        @test linkdims(tci) == ones(n - 1)
        @test rank(tci) == 1

        for p in 1:n-1
            addpivot!(tci, p, f, 1e-8)
        end
        @test linkdims(tci) == fill(2, n - 1)
        @test rank(tci) == 2

        globalpivot = [2, 9, 10, 5, 7]
        addglobalpivot!(tci, f, globalpivot, 1e-12)
        @test linkdims(tci) == fill(3, n - 1)
        @test rank(tci) == 3
        @test evaluate(tci, globalpivot) ≈ f(globalpivot)

        addglobalpivot!(tci, f, globalpivot, 1e-12)
        @test linkdims(tci) == fill(3, n - 1)
        @test rank(tci) == 3
        @test evaluate(tci, globalpivot) ≈ f(globalpivot)

        for iter in 4:8
            for p in 1:n-1
                addpivot!(tci, p, f, 1e-8)
            end
            @test linkdims(tci) == fill(iter, n - 1)
            @test rank(tci) == iter
        end

        tci2, ranks, errors = crossinterpolate1(
            ValueType,
            f,
            fill(10, n),
            ones(Int, n);
            tolerance=1e-8,
            maxiter=8,
            sweepstrategy=:forward
        )

        @test linkdims(tci) == linkdims(tci2)
        @test rank(tci) == rank(tci2)

        tci3, ranks, errors = crossinterpolate1(
            ValueType,
            f,
            fill(10, n),
            ones(Int, n);
            tolerance=1e-12,
            maxiter=200
        )

        @test all(tci3.pivoterrors .<= 1e-12)
        @test all(linkdims(tci3) .<= 200)
        @test rank(tci3) <= 200

        tci4, ranks, errors = crossinterpolate1(
            ValueType,
            f,
            fill(10, n),
            ones(Int, n);
            tolerance=1e-12,
            maxiter=200,
            additionalpivots=[
                [10, 8, 10, 4, 4],
                [5, 4, 8, 9, 3],
                [7, 7, 10, 5, 9],
                [7, 7, 10, 5, 9]
            ]
        )

        @test all(tci4.pivoterrors .<= 1e-12)
        @test all(linkdims(tci4) .<= 200)
        @test rank(tci4) <= 200

        tt3 = tensortrain(tci3)

        for v in Iterators.product([1:3 for p in 1:n]...)
            value = evaluate(tci3, [i for i in v])
            @test value ≈ prod([tt3[p][:, v[p], :] for p in eachindex(v)])[1]
            @test value ≈ f(v)
        end
    end
end
