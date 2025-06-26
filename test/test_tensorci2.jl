using Test
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: rank, linkdims, TensorCI2, updatepivots!, addglobalpivots1sitesweep!, MultiIndex, evaluate, crossinterpolate2, pivoterror, tensortrain, optimize!
import Random
import QuanticsGrids as QD

TCI.initializempi(false)

# TODO write test for parallel subfunctions

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

    @testset "pivoterrors" for sweepstrategy in [:backandforth, :parallel]
        diags = [1.0, 1e-5, 0.0]
        f(x) = (x[1] == x[2] ? diags[x[1]] : 0.0)
        localdims = [3, 3]
        tci, ranks, errors = crossinterpolate2(
            Float64,
            f,
            localdims,
            [[1, 1]];
            tolerance=1e-8,
            sweepstrategy=:sweepstrategy
        )
        @test tci.pivoterrors == diags
    end

    @testset "checkbatchevaluatable" for sweepstrategy in [:backandforth, :parallel]
        f(x) = 1.0 # Constant function without batch evaluation
        L = 10
        localdims = fill(2, L)
        firstpivots = [fill(1, L)]
        @test_throws ErrorException crossinterpolate2(
            Float64,
            f,
            localdims,
            firstpivots;
            checkbatchevaluatable=true,
            sweepstrategy=:sweepstrategy
        )
    end

    @testset "trivial MPS(exp): pivotsearch=$pivotsearch" for pivotsearch in [:full, :rook], strictlynested in [false, true], nsearchglobalpivot in [0, 10], sweepstrategy in [:backandforth, :parallel]
        if nsearchglobalpivot > 0 && strictlynested
            continue
        end
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
            pivotsearch=pivotsearch,
            strictlynested=strictlynested,
            sweepstrategy=:sweepstrategy
        )

        @test all(TCI.linkdims(tci) .== 1)

        # Conversion to TT
        tt = TCI.TensorTrain(tci)

        for x in [0.1, 0.3, 0.6, 0.9]
            indexset = QD.origcoord_to_quantics(
                grid, (x,)
            )
            @test abs(TCI.evaluate(tci, indexset) - f(indexset)) < abstol
            @test abs(TCI.evaluate(tt, indexset) - f(indexset)) < abstol
        end


    end

    @testset "trivial MPS(exp), small maxbonddim" for sweepstrategy in [:backandforth, :parallel]
        pivotsearch = :full
        strictlynested = false
        nsearchglobalpivot = 10

        # f(x) = exp(-x)
        Random.seed!(1240)
        R = 8
        abstol = 1e-10

        grid = QD.DiscretizedGrid{1}(R, (0.0,), (1.0,))

        fx(x) = exp(-x) + 1e-4 * exp(-2x)
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
            maxiter=10,
            loginterval=1,
            verbosity=0,
            normalizeerror=false,
            nsearchglobalpivot=nsearchglobalpivot,
            pivotsearch=pivotsearch,
            strictlynested=strictlynested,
            sweepstrategy=:sweepstrategy
        )

        @test all(TCI.linkdims(tci) .== 1)

        # Conversion to TT
        tt = TCI.TensorTrain(tci)

        for x in [0.1, 0.3, 0.6, 0.9]
            indexset = QD.origcoord_to_quantics(
                grid, (x,)
            )
            @test abs(TCI.evaluate(tci, indexset) - f(indexset)) < 1e-4
            @test abs(TCI.evaluate(tt, indexset) - f(indexset)) < 1e-4
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

    @testset "TCI2 errors and warnings" begin
        n = 5
        f(v) = 1.0 ./ (sum(v .^ 2) + 1)

        @test_throws ArgumentError crossinterpolate2(Float64, f, fill(2, n); tolerance=1e-9, pivottolerance=1e-2)
        @test_throws ArgumentError crossinterpolate2(Float64, f, fill(2, n); tolerance=0.0)

        tci, = crossinterpolate2(Float64, f, fill(2, n); tolerance=0.1)
        @test_throws ArgumentError optimize!(tci, f; pivottolerance = 0.1, tolerance = 0.01)
        @test_throws ArgumentError optimize!(tci, f; tolerance = 0.0)
        @test_logs (:warn, "The option `pivottolerance` of `optimize!(tci::TensorCI2, f)` is deprecated. Please update your code to use `tolerance`, as `pivottolerance` will be removed in the future.") optimize!(tci, f; pivottolerance = 0.1)
    end

    # This is a stochastic test
    @testset "Lorentz MPS with ValueType=$(typeof(coeff)), pivotsearch=$pivotsearch" for seed in collect(1:20), coeff in [1.0, 0.5 - 1.0im], pivotsearch in [:full, :rook], sweepstrategy in [:backandforth, :parallel]
        Random.seed!(seed)
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
            maxiter=8,
            sweepstrategy=:sweepstrategy,
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
            pivotsearch,
            sweepstrategy=:sweepstrategy
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
            pivotsearch,
            sweepstrategy=:sweepstrategy
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


    @testset "insert_global_pivots: pivotsearch=$pivotsearch, strictlynested=$strictlynested, seed=$seed" for seed in collect(1:20), pivotsearch in [:full, :rook], strictlynested in [false], sweepstrategy in [:backandforth, :parallel]
        Random.seed!(seed)

        R = 20
        abstol = 1e-4
        δ = 10.0 / 2^R # Peaks are wider than 1/2^R.
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
            strictlynested=strictlynested,
            sweepstrategy=:sweepstrategy
        )

        TCI.addglobalpivots2sitesweep!(
            tci, f, rindex,
            tolerance=abstol,
            normalizeerror=false,
            maxbonddim=1000,
            pivotsearch=pivotsearch,
            verbosity=0,
            strictlynested=strictlynested,
            ntry=pivotsearch == :full ? 1 : 10
        )

        @test sum(abs.([TCI.evaluate(tci, r) - f(r) for r in rindex]) .> abstol) == 0
    end

    @testset "insert_global_pivots" for sweepstrategy in [:backandforth, :parallel]
        Random.seed!(1234)

        R = 20
        abstol = 1e-4

        f(q) = (all(q .== 1) || all(q .== 2)) ? 1.0 : 0.0

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
            strictlynested=false,
            sweepstrategy=:sweepstrategy
        )

        r = fill(2, R)

        TCI.addglobalpivots2sitesweep!(
            tci, f, [r],
            tolerance=abstol,
            normalizeerror=false,
            maxbonddim=1000,
            verbosity=0,
            strictlynested=false
        )

        @test TCI.evaluate(tci, r) ≈ f(r)
    end

    @testset "globalsearch" for sweepstrategy in [:backandforth, :parallel]
        Random.seed!(1234)

        n = 10
        fx(x) = exp(-10 * x) * sin(2 * pi * 100 * x^1.1) # Nasty function
        f(bitlist) = fx(QD.quantics_to_origcoord(grid, bitlist)[1])
        grid = QD.DiscretizedGrid{1}(n, (0.0,), (1.0,))

        localdims = fill(2, n)

        # This checks only that the function runs without error
        firstpivot = TCI.optfirstpivot(f, localdims, [rand(1:d) for d in localdims])
        tci, ranks, errors = crossinterpolate2(
            Float64,
            f,
            localdims,
            [firstpivot];
            tolerance=1e-12,
            maxbonddim=100,
            maxiter=100,
            nsearchglobalpivot=10,
            strictlynested=false,
            sweepstrategy=:parallel
        )

        @test errors[end] < 1e-10
    end


    @testset "crossinterpolate2_ttcache" for sweepstrategy in [:backandforth, :parallel]
        ValueType = Float64

        N = 4
        bonddims = [1, 2, 3, 2, 1]
        @assert length(bonddims) == N + 1
        localdims = [2, 3, 3, 2]

        tt = TCI.TensorTrain{ValueType,3}([rand(bonddims[n], localdims[n], bonddims[n+1]) for n in 1:N])
        ttc = TCI.TTCache(tt)

        tci2, ranks, errors = TCI.crossinterpolate2(
            ValueType,
            ttc,
            localdims;
            tolerance=1e-10,
            maxbonddim=10,
            sweepstrategy=:sweepstrategy
        )

        tt_reconst = TCI.TensorTrain(tci2)

        vals_reconst = [tt_reconst(collect(indices)) for indices in Iterators.product((1:d for d in localdims)...)]
        vals_ref = [tt(collect(indices)) for indices in Iterators.product((1:d for d in localdims)...)]

        @test vals_reconst ≈ vals_ref
    end

    @testset "multithreadPi" begin
        f = (x -> sum(x))

        localdims = [2,2,2,2,2,2]

        Icombined = [[1,1,1,1], [1,2,1,1], [1,1,2,1]]
        Jcombined = [[2,2], [1,2], [2,1], [1,1]]

        mPi = TCI.multithreadPi(Float64, f, localdims, Icombined, Jcombined)

        Pi = reshape(
            TCI.filltensor(Float64, f, localdims,
            Icombined, Jcombined, Val(0)),
            length(Icombined), length(Jcombined)
        )

        @test size(Pi) == (3, 4)

        @test mPi == Pi
    end

    @testset "noderanges" begin
        @test TCI._noderanges(8, 24, 2, 100) == [1:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:24]
        @test TCI._noderanges(8, 24, 2, 500) == [1:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:24]
        @test TCI._noderanges(8, 24, 2, 100; interbond=false) == [1:24 for _ in 1:8]
        @test TCI._noderanges(8, 24, 2, 500; interbond=false) == [1:24 for _ in 1:8]
        @test TCI._noderanges(8, 24, 2, 100) == TCI._noderanges(8, 24, 2, 100; estimatedbonds=[2,4,8,16,32,64,100,100,100,100,100,100,100,100,100,100,100,64,32,16,8,4,2])

        ranges, eff = TCI._noderanges(4, 8, 8, 64, efficiencycheck=true)
        @test eff > 3.9
    end

    @testset "convergencecriterion" begin
        let
            ranks = [1, 2]
            errors = [1e-2, 1e-5]
            nglobalpivot = [0, 0]
            tol = 1e-4
            maxbonddim = 4
            ncheckhistory = 3
            @test TCI.convergencecriterion(ranks, errors, nglobalpivot, tol, maxbonddim, ncheckhistory) == false
        end

        let
            ranks = [1, 2, 2, 2]
            errors = [1e-2, 1e-5, 1e-5, 1e-5]
            nglobalpivot = [0, 0, 0, 0]
            tol = 1e-4
            maxbonddim = 4
            ncheckhistory = 3
            @test TCI.convergencecriterion(ranks, errors, nglobalpivot, tol, maxbonddim, ncheckhistory) == true
        end

        let
            ranks = [1, 2, 2, 2]
            errors = [1e-2, 1e-2, 1e-5, 1e-5]
            nglobalpivot = [0, 0, 0, 0]
            tol = 1e-4
            maxbonddim = 4
            ncheckhistory = 3
            @test TCI.convergencecriterion(ranks, errors, nglobalpivot, tol, maxbonddim, ncheckhistory) == false
        end

        let
            ranks = [1, 2, 2, 2]
            errors = [1e-2, 1e-2, 1e-2, 1e-2]
            nglobalpivot = [0, 0, 0, 0]
            tol = 1e-4
            maxbonddim = 2
            ncheckhistory = 3
            @test TCI.convergencecriterion(ranks, errors, nglobalpivot, tol, maxbonddim, ncheckhistory) == true
        end

        let
            ranks = [1, 2, 2, 2]
            errors = [1e-2, 1e-2, 1e-2, 1e-2]
            nglobalpivot = [0, 1, 1, 1]
            tol = 1e-4
            maxbonddim = 2
            ncheckhistory = 3
            @test TCI.convergencecriterion(ranks, errors, nglobalpivot, tol, maxbonddim, ncheckhistory) == true
        end
    end
end

