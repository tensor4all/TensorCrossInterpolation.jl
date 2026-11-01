using Test
import TensorCrossInterpolation as TCI


@testset "batcheval" begin
    @testset "removed inherited API" begin
        @test !isdefined(TCI, :BatchEvaluator)
        @test !isdefined(TCI, :ThreadedBatchEvaluator)
        @test !isdefined(TCI, :makebatchevaluatable)
    end

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

    @testset "batchedf! matrix API" begin
        localdims = [2, 3, 2, 4]
        leftindexset = [[1], [2]]
        rightindexset = [[1], [3], [4]]
        seen_shape = Ref{Tuple{Int,Int}}()
        seen_indices = Ref{Matrix{Int}}()
        returned = Ref(false)

        f = x -> sum(x)
        function batchedf!(values, indices)
            @test indices isa Matrix{Int}
            @test values isa Vector{Float64}
            seen_shape[] = size(indices)
            seen_indices[] = copy(indices)
            for p in axes(indices, 2)
                values[p] = sum(view(indices, :, p))
            end
            returned[] = true
            return nothing
        end

        result = TCI._batchevaluate_dispatch(
            Float64,
            f,
            batchedf!,
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
        @test returned[]
    end

    @testset "batchedf! ignores return value" begin
        localdims = [2, 3, 2]
        f = x -> sum(x)
        function batchedf!(values, indices)
            values .= [100 + p for p in eachindex(values)]
            return fill(-1.0, length(values))
        end
        result = TCI._batchevaluate_dispatch(
            Float64,
            f,
            batchedf!,
            localdims,
            [[1]],
            [[1]],
            Val(1),
        )
        @test vec(result) == [101.0, 102.0, 103.0]
    end

    @testset "threaded batchedf!" begin
        L = 20
        localdims = fill(2, L)
        f = x -> sum(x)
        function batchedf!(values, indices)
            Threads.@threads for p in axes(indices, 2)
                values[p] = sum(view(indices, :, p))
            end
            return values
        end

        # Compute Pi tensor
        nl = 10
        nr = L - nl - 2

        # 20 left index sets, 20 right index sets
        leftindexset = [[rand(1:d) for d in localdims[1:nl]] for _ in 1:20]
        rightindexset = [[rand(1:d) for d in localdims[nl+3:end]] for _ in 1:20]

        result = TCI._batchevaluate_dispatch(Float64, f, batchedf!, localdims, leftindexset, rightindexset, Val(2))
        ref = [sum(vcat(l, c, cp, r)) for l in leftindexset, c in 1:localdims[nl+1], cp in 1:localdims[nl+2], r in rightindexset]

        @test result ≈ ref
    end

    @testset "crossinterpolate2 threaded batchedf!" begin
        function f(x)
            sleep(1e-3)
            return Float64(sum(x))
        end

        L = 20
        localdims = fill(2, L)
        function batchedf!(values, indices)
            Threads.@threads for p in axes(indices, 2)
                values[p] = f(collect(view(indices, :, p)))
            end
            values
        end

        tci, ranks, errors = TCI.crossinterpolate2(Float64, f, localdims; batchedf!)

        tci_ref, ranks_ref, errors_ref = TCI.crossinterpolate2(Float64, f, localdims)

        @test TCI.fulltensor(TCI.TensorTrain(tci)) ≈ TCI.fulltensor(TCI.TensorTrain(tci_ref))
    end
end
