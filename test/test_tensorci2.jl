using Test
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: rank, linkdims, TensorCI2, updatepivots!, addglobalpivots!,
    evaluate, SweepStrategies, crossinterpolate2, pivoterror, tensortrain
import Random

@testset "TensorCI2" begin
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

        tci = TensorCI2{Float64}(fill(2, n))
        @test length(tci) == n
        @test rank(tci) == 0
        @test linkdims(tci) == fill(0, n - 1)
        @test tci.localset == fill([1, 2], n)
        for i in 1:n
            @test isempty(tci.Iset[i])
            @test isempty(tci.Jset[i])
        end

        tci = TCI.TensorCI2{Float64}(f, fill(2, n), [fill(1, n)])
        @test length(tci) == n
        @test rank(tci) == 1
        @test linkdims(tci) == fill(1, n - 1)
        @test tci.localset == fill([1, 2], n)
    end

    @testset "Lorentz MPS" for coeff in [1.0, 1.0im]
        n = 5
        f(v) = coeff ./ (sum(v .^ 2) + 1)

        ValueType = typeof(coeff)

        tci = TensorCI2{ValueType}(f, fill(10, n))

        @test linkdims(tci) == ones(n - 1)
        @test rank(tci) == 1
        @test length(tci.Iset[1]) == 1
        @test length(tci.Jset[end]) == 1

        for p in 1:n-1
            updatepivots!(tci, p, f, true; reltol=1e-8, maxbonddim=2)
        end
        @test linkdims(tci) == fill(2, n - 1)
        @test rank(tci) == 2
        @test length(tci.Iset[1]) == 1
        @test length(tci.Jset[end]) == 1

        globalpivot = [2, 9, 10, 5, 7]
        addglobalpivots!(tci, f, [globalpivot], reltol=1e-12)
        @test linkdims(tci) == fill(3, n - 1)
        @test rank(tci) == 3
        @test length(tci.Iset[1]) == 1
        @test length(tci.Jset[end]) == 1

        for iter in 4:8
            for p in 1:n-1
                updatepivots!(tci, p, f, true; reltol=1e-8)
            end
        end

        tci2, ranks, errors = crossinterpolate2(
            ValueType,
            f,
            fill(10, n),
            [ones(Int, n)];
            tolerance=1e-8,
            pivottolerance=1e-8,
            maxiter=8,
            sweepstrategy=SweepStrategies.forward
        )

        @test linkdims(tci) == linkdims(tci2)
        @test rank(tci) == rank(tci2)

        tci3, ranks, errors = crossinterpolate2(
            ValueType,
            f,
            fill(10, n),
            [ones(Int, n)];
            tolerance=1e-12,
            maxiter=200
        )

        @test pivoterror(tci3) <= 1e-12
        @test all(linkdims(tci3) .<= 200)
        @test rank(tci3) <= 200

        initialpivots = [
            [1, 1, 1, 1, 1],
            [10, 8, 10, 4, 4],
            [5, 4, 8, 9, 3],
            [7, 7, 10, 5, 9],
            [7, 7, 10, 5, 9]
        ]

        tci4, ranks, errors = crossinterpolate2(
            ValueType,
            f,
            fill(10, n),
            initialpivots;
            tolerance=1e-12,
            maxiter=200
        )

        @test pivoterror(tci4) <= 1e-12
        @test all(linkdims(tci4) .<= 200)
        @test rank(tci4) <= 200

        tt3 = tensortrain(tci3)

        for v in Iterators.product([1:3 for p in 1:n]...)
            value = evaluate(tci3, [i for i in v])
            @test value ≈ prod([tt3[p][:, v[p], :] for p in eachindex(v)])[1]
            @test value ≈ f(v)
        end
    end

    @testset "insert_global_pivots" begin
        Random.seed!(1234)

        n = 10
        fx(x) = exp(-10*x) * sin(2 * pi * 100 * x^1.1) # Nasty function
        index_to_x(i) = (i-1) / 2^n # x ∈ [0, 1)
        f(bitlist) = bitlist |> quantics_to_index |> index_to_x |> fx

        localdims = fill(2, n)
        tci, ranks, errors = crossinterpolate2(
            Float64,
            f,
            localdims,
            tolerance=1e-12,
            maxiter=200
        )

        nglobalpivot = TCI.insertglobalpivots!(tci, f, verbosity=0)
        @test nglobalpivot > 0

        nglobalpivot = TCI.insertglobalpivots!(tci, f, verbosity=0)
        @test nglobalpivot == 0

    end
end
