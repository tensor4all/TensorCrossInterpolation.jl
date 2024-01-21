import TensorCrossInterpolation as TCI
using Random
using BenchmarkTools

function print_benchmark_result(benchmark_result)
    println("Minimum time: ", minimum(benchmark_result).time, " ns")
    println("Median time: ", median(benchmark_result).time, " ns")
    println("Mean time: ", mean(benchmark_result).time, " ns")
    println("Maximum time: ", maximum(benchmark_result).time, " ns")
    println("Allocations: ", benchmark_result.allocs)
    println("Memory usage: ", benchmark_result.memory, " bytes")
end

function readcache(cf, indexsets)
    for i  in indexsets
        cf(i)
    end
end

function targetf(::Type{T}, L) where {T}
    localdims = fill(2, L)
    f(x) = 1.0
    cf = TCI.CachedFunction{T}(f, localdims)
    N = Int(1e+4)
    indexsets = [[rand(1:d) for d in localdims] for _ in 1:N]
    for i in indexsets
        cf(i)
    end

    #return @benchmark $cf($(indexsets[1]))
    return @benchmark readcache($cf, $indexsets)
end


print_benchmark_result(targetf(Float64, 40))
print_benchmark_result(targetf(ComplexF64, 40))
