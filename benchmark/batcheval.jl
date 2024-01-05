import TensorCrossInterpolation as TCI
using BenchmarkTools
using Random

function print_benchmark_result(benchmark_result)
    println("Minimum time: ", minimum(benchmark_result).time, " ns")
    println("Median time: ", median(benchmark_result).time, " ns")
    println("Mean time: ", mean(benchmark_result).time, " ns")
    println("Maximum time: ", maximum(benchmark_result).time, " ns")
    println("Allocations: ", benchmark_result.allocs)
    println("Memory usage: ", benchmark_result.memory, " bytes")
end

function benchmark1()
    localdims = [2, 2, 2, 2, 2]
    localset = [collect(1:d) for d in localdims]
    leftindexset = [[1, 1] for _ in 1:100]
    rightindexset = [[1, 1] for _ in 1:100]

    benchmark_result = @benchmark TCI._batchevaluate_dispatch(Float64, x -> 1.0, $localset, $leftindexset, $rightindexset, Val(1))

    print_benchmark_result(benchmark_result)

    leftindexset = [[1] for _ in 1:100]
    rightindexset = [[1, 1] for _ in 1:100]

    benchmark_result = @benchmark TCI._batchevaluate_dispatch(Float64, x -> 1.0, $localset, $leftindexset, $rightindexset, Val(2))

    print_benchmark_result(benchmark_result)
end

function benchmark2()
    Random.seed!(1234)
    L = 30
    localdims = fill(2, L)
    localset = [collect(1:d) for d in localdims]
    f = x -> 1.0
    cf = TCI.CachedFunction{Float64}(f, localdims)

    nfilldata = 1000000
    filldata = Set([rand(1:2, L) for _ in 1:nfilldata])
    _ = sum(cf.(filldata))

    testdata = rand(1:2, L)
    while testdata ∈ filldata
        testdata = rand(1:2, L)
    end

    benchmark_result = @benchmark $testdata ∈ $cf

    println("")
    println("# in cache: ", length(keys(cf.cache)))
    print_benchmark_result(benchmark_result)

    #==
    hashset = Set([hash(x) for x in filldata])
    benchmark_result = @benchmark hash($testdata) ∈ $hashset

    println("")
    println("# in cache: ", length(keys(cf.cache)))
    print_benchmark_result(benchmark_result)
    ==#
end

benchmark1()
benchmark2()