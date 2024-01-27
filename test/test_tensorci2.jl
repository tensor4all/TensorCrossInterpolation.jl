using Test
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: rank, linkdims, TensorCI2, updatepivots!, addglobalpivots1sitesweep!, MultiIndex, evaluate, SweepStrategies, crossinterpolate2, pivoterror, tensortrain
import Random
import QuanticsGrids as QD

@testset "TensorCI2" begin
    @testset "kronecker util function" begin
        multiset = [collect(1:5) for _ in 1:5]
        localdim = 4
        localset = collect(1:localdim)

        c = TCI.kronecker(multiset, localdim)
        for (i, ci) in enumerate(c)
            @test ci[1:5] == collect(1:5)
            @test ci[6] in localset
        end

        d = TCI.kronecker(localdim, multiset)
        for (i, di) in enumerate(d)
            @test di[1] in localset
            @test di[2:6] == collect(1:5)
        end
    end

    @testset "trivial MPS(exp): pivotsearch=$pivotsearch" for pivotsearch in [:full, :rook], partialnesting in [false, true], nsearchglobalpivot in [0, 10]
        # f(x) = exp(-x)
        Random.seed!(1240)
        R = 8
        abstol = 1e-4

        grid = QD.DiscretizedGrid{1}(R, (0.0,), (1.0,))

        #index_to_x(i) = (i - 1) / 2^R # x ∈ [0, 1)
        fx(x) = exp(-x)
        f(bitlist::MultiIndex) = fx(QD.quantics_to_origcoord(grid, bitlist)[1])

        localdims = fill(2, R)
        firstpivots = [ones(Int, R), vcat(1, fill(2, R - 1))]
        tci, ranks, errors = crossinterpolate2(
            Float64,
            f,
            localdims,
            firstpivots;
            tolerance=abstol,
            maxbonddim=1,
            maxiter=2,
            loginterval=1,
            verbosity=0,
            normalizeerror=false,
            nsearchglobalpivot=nsearchglobalpivot,
            pivotsearch=pivotsearch
        )

        @test all(TCI.linkdims(tci) .== 1)

        for x in [0.1, 0.3, 0.6, 0.9]
            indexset = QD.origcoord_to_quantics(
                grid, (x,)
            )
            @test abs(TCI.evaluate(tci, indexset) - f(indexset)) < abstol
        end

    end

    @testset "trivial MPS" begin
        n = 5
        f(v) = sum(v) * 0.5

        tci = TensorCI2{Float64}(fill(2, n))
        @test length(tci) == n
        @test rank(tci) == 0
        @test linkdims(tci) == fill(0, n - 1)
        for i in 1:n
            @test isempty(tci.Iset[i])
            @test isempty(tci.Jset[i])
        end

        tci = TCI.TensorCI2{Float64}(f, fill(2, n), [fill(1, n)])
        @test length(tci) == n
        @test rank(tci) == 1
        @test linkdims(tci) == fill(1, n - 1)
    end

    @testset "Lorentz MPS with ValueType=$(typeof(coeff)), pivotsearch=$pivotsearch" for coeff in [1.0, 0.5 - 1.0im], pivotsearch in [:full, :rook]
        n = 5
        f(v) = coeff ./ (sum(v .^ 2) + 1)

        ValueType = typeof(coeff)

        tci = TensorCI2{ValueType}(f, fill(10, n))

        @test linkdims(tci) == ones(n - 1)
        @test rank(tci) == 1
        @test length(tci.Iset[1]) == 1
        @test length(tci.Jset[end]) == 1

        for p in 1:n-1
            updatepivots!(tci, p, f, true; reltol=1e-8, maxbonddim=2, pivotsearch)
        end
        @test linkdims(tci) == fill(2, n - 1)
        @test rank(tci) == 2
        @test length(tci.Iset[1]) == 1
        @test length(tci.Jset[end]) == 1

        globalpivot = [2, 9, 10, 5, 7]
        addglobalpivots1sitesweep!(tci, f, [globalpivot], reltol=1e-12)
        @test linkdims(tci) == fill(3, n - 1)
        @test rank(tci) == 3
        @test length(tci.Iset[1]) == 1
        @test length(tci.Jset[end]) == 1

        for iter in 4:20
            for p in 1:n-1
                updatepivots!(tci, p, f, true; reltol=1e-8, pivotsearch)
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
            sweepstrategy=SweepStrategies.forward,
            pivotsearch=pivotsearch
        )

        #@test linkdims(tci) == linkdims(tci2) Too strict
        if pivotsearch == :full
            @test rank(tci) == rank(tci2)
        end

        tci3, ranks, errors = crossinterpolate2(
            ValueType,
            f,
            fill(10, n),
            [ones(Int, n)];
            tolerance=1e-12,
            maxiter=200,
            pivotsearch
        )

        @test pivoterror(tci3) <= 2e-12
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
            maxiter=200,
            pivotsearch
        )

        @test pivoterror(tci4) <= 2e-12
        @test all(linkdims(tci4) .<= 200)
        @test rank(tci4) <= 200

        tt3 = tensortrain(tci3)

        for v in Iterators.product([1:3 for p in 1:n]...)
            value = evaluate(tci3, [i for i in v])
            @test value ≈ prod([tt3[p][:, v[p], :] for p in eachindex(v)])[1]
            @test value ≈ f(v)
        end
    end


    @testset "insert_global_pivots: pivotsearch=$pivotsearch, partialnesting=$partialnesting, seed=$seed" for seed in collect(1:20), pivotsearch in [:full, :rook], partialnesting in [false]
        Random.seed!(seed)

        R = 20
        abstol = 1e-4
        δ = 10.0/2^R # Peaks are wider than 1/2^R.
        grid = QD.DiscretizedGrid{1}(R, (0.0,), (1.0,))

        rindex = [rand(1:2, R) for _ in 1:100]

        f(bitlist) = fx(QD.quantics_to_origcoord(grid, bitlist)[1])
        rpoint = Float64[QD.quantics_to_origcoord(grid, r)[1] for r in rindex]

        function fx(x)
            res = exp(-10 * x)
            for r in rpoint
                res += abs(x - r) < δ ? 2 * abstol : 0.0
            end
            res
        end

        localdims = fill(2, R)
        firstpivot = ones(Int, R)
        tci, ranks, errors = crossinterpolate2(
            Float64,
            f,
            localdims,
            [firstpivot];
            tolerance=abstol,
            maxbonddim=1000,
            maxiter=20,
            loginterval=1,
            verbosity=0,
            normalizeerror=false,
            pivotsearch=pivotsearch,
            partialnesting=partialnesting
        )

        TCI.addglobalpivots2sitesweep!(
            tci, f, rindex,
            tolerance=abstol,
            normalizeerror=false,
            maxbonddim=1000,
            pivotsearch=pivotsearch,
            verbosity=0,
            partialnesting=partialnesting,
            ntry = pivotsearch == :full ? 1 : 10
        )

        @test sum(abs.([TCI.evaluate(tci, r) - f(r) for r in rindex]) .> abstol) == 0
    end

    @testset "globalsearch" begin
        Random.seed!(1234)

        n = 10
        fx(x) = exp(-10 * x) * sin(2 * pi * 100 * x^1.1) # Nasty function
        f(bitlist) = fx(QD.quantics_to_origcoord(grid, bitlist)[1])
        grid = QD.DiscretizedGrid{1}(n, (0.0,), (1.0,))

        localdims = fill(2, n)

        # This checks only that the function runs without error
        tci, ranks, errors = crossinterpolate2(
            Float64,
            f,
            localdims,
            tolerance=1e-12,
            maxbonddim=100,
            maxiter=100,
            nsearchglobalpivot=10
        )

        @test errors[end] < 1e-10
    end


    @testset "crossinterpolate2_ttcache" begin
        ValueType = Float64

        N = 4
        bonddims = [1, 2, 3, 2, 1]
        @assert length(bonddims) == N + 1
        localdims = [2, 3, 3, 2]

        tt = TCI.TensorTrain{ValueType,3}([rand(bonddims[n], localdims[n], bonddims[n+1]) for n in 1:N])
        ttc = TCI.TTCache(tt.T)

        tci2, ranks, errors = TCI.crossinterpolate2(
            ValueType,
            ttc,
            localdims;
            tolerance=1e-10,
            maxbonddim = 10
        )

        tt_reconst = TCI.TensorTrain(tci2)

        vals_reconst = [tt_reconst(collect(indices)) for indices in Iterators.product((1:d for d in localdims)...)]
        vals_ref = [tt(collect(indices)) for indices in Iterators.product((1:d for d in localdims)...)]

        @test vals_reconst ≈ vals_ref
    end
end
