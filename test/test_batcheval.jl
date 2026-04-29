using Test
import TensorCrossInterpolation as TCI


struct NonBatchEvaluator{T} <: Function end

function (f::NonBatchEvaluator{T})(x::Vector{Int})::T where {T}
    return sum(x)
end


@testset "batcheval" begin
    @testset "M=1" begin
        localdims = [2, 2, 2, 2, 2]
        leftindexset = [[1, 1] for _ in 1:100]
        rightindexset = [[1, 1] for _ in 1:100]

        f = x -> sum(x)
        result = TCI._batchevaluate_dispatch(Float64, f, localdims, leftindexset, rightindexset, Val(1))
        ref = [sum(vcat(l, c, r)) for l in leftindexset, c in 1:localdims[3], r in rightindexset]

        @test result ≈ ref
    end

    @testset "M=2" begin
        localdims = [2, 2, 2, 2, 2]
        leftindexset = [[1] for _ in 1:100]
        rightindexset = [[1, 1] for _ in 1:100]

        f = x -> sum(x)
        result = TCI._batchevaluate_dispatch(Float64, f, localdims, leftindexset, rightindexset, Val(2))
        ref = [sum(vcat(l, c, cp, r)) for l in leftindexset, c in 1:localdims[2], cp in 1:localdims[3], r in rightindexset]

        @test result ≈ ref
    end

    @testset "batchedf matrix API" begin
        localdims = [2, 3, 2, 4]
        leftindexset = [[1], [2]]
        rightindexset = [[1], [3], [4]]
        seen_shape = Ref{Tuple{Int,Int}}()
        seen_indices = Ref{Matrix{Int}}()

        f = x -> sum(x)
        batchedf = indices -> begin
            @test indices isa Matrix{Int}
            seen_shape[] = size(indices)
            seen_indices[] = copy(indices)
            [sum(view(indices, :, p)) for p in axes(indices, 2)]
        end

        result = TCI._batchevaluate_dispatch(
            Float64,
            f,
            batchedf,
            localdims,
            leftindexset,
            rightindexset,
            Val(2),
        )

        ref = [
            sum(vcat(l, c, cp, r))
            for l in leftindexset,
                c in 1:localdims[2],
                cp in 1:localdims[3],
                r in rightindexset
        ]
        expected_indices = Matrix{Int}(undef, length(localdims), length(result))
        p = 1
        for r in rightindexset
            for cp in 1:localdims[3]
                for c in 1:localdims[2]
                    for l in leftindexset
                        expected_indices[:, p] .= vcat(l, c, cp, r)
                        p += 1
                    end
                end
            end
        end

        @test seen_shape[] == (length(localdims), length(result))
        @test result ≈ ref
        @test seen_indices[] == expected_indices
    end

    @testset "batchedf validates output length" begin
        localdims = [2, 2, 2]
        f = x -> sum(x)
        bad_batchedf = indices -> ones(Float64, size(indices, 2) - 1)

        @test_throws DimensionMismatch TCI._batchevaluate_dispatch(
            Float64,
            f,
            bad_batchedf,
            localdims,
            [[1]],
            [[1]],
            Val(1),
        )
    end

    @testset "BatchEvaluator" begin
        tbf = NonBatchEvaluator{Float64}()

        leftindexset = [[1], [2]]
        rightindexset = [[1], [2]]
        localdims = [3, 3, 3, 3]

        bf = TCI.makebatchevaluatable(Float64, tbf, localdims)
        @test size(bf(leftindexset, rightindexset, Val(1))) == (2, 3, 2)
    end

    @testset "BatchEvaluator direct batchedf wrapper" begin
        localdims = [1, 2, 1]
        f = x -> sum(x)
        batchedf_calls = Ref(0)
        batchedf_indices = Ref{Matrix{Int}}()
        batchedf = indices -> begin
            batchedf_calls[] += 1
            batchedf_indices[] = copy(indices)
            [100 + p for p in axes(indices, 2)]
        end

        wrapped = TCI.makebatchevaluatable(Float64, f, batchedf, localdims)

        @test TCI.isbatchevaluable(wrapped)
        @test wrapped([1, 1, 1]) == f([1, 1, 1])
        @test wrapped([[1]], [[1]], Val(1)) ≈ reshape([101, 102], 1, 2, 1)
        @test batchedf_calls[] == 1
        @test batchedf_indices[] == [1 1; 1 2; 1 1]
    end

    @testset "ThreadedBatchEvaluator" begin
        L = 20
        localdims = fill(2, L)
        f = x -> sum(x)

        f = TCI.ThreadedBatchEvaluator{Float64}(f, localdims)

        # Compute Pi tensor
        nl = 10
        nr = L - nl - 2

        # 20 left index sets, 20 right index sets
        leftindexset = [[rand(1:d) for d in localdims[1:nl]] for _ in 1:20]
        rightindexset = [[rand(1:d) for d in localdims[nl+3:end]] for _ in 1:20]

        result = f(leftindexset, rightindexset, Val(2))
        ref = [sum(vcat(l, c, cp, r)) for l in leftindexset, c in 1:localdims[nl+1], cp in 1:localdims[nl+2], r in rightindexset]

        @test result ≈ ref
    end

    @testset "ThreadedBatchEvaluator (from Matsuura)" begin
        function f(x)
            sleep(1e-3)
            return sum(x)
        end

        L = 20
        localdims = fill(2, L)
        parf = TCI.ThreadedBatchEvaluator{Float64}(f, localdims)

        tci, ranks, errors = TCI.crossinterpolate2(Float64, parf, localdims)

        tci_ref, ranks_ref, errors_ref = TCI.crossinterpolate2(Float64, f, localdims)

        @test TCI.fulltensor(TCI.TensorTrain(tci)) ≈ TCI.fulltensor(TCI.TensorTrain(tci_ref))
    end
end
