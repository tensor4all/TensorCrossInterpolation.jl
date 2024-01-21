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

function computekey(L, N)
    localdims = fill(2, L)
    indexsets = [[rand(1:d) for d in localdims] for _ in 1:N]
    cf = TCI.CachedFunction{ComplexF64}(x->1.0, localdims)
    return @benchmark for i in $indexsets TCI._key($cf, i) end
end


function cachequery(::Type{T}, L, N) where {T}
    localdims = fill(2, L)
    f(x) = 1.0
    cf = TCI.CachedFunction{T}(f, localdims)
    indexsets = [[rand(1:d) for d in localdims] for _ in 1:N]
    for i in indexsets
        cf(i)
    end
    return @benchmark readcache($cf, $indexsets)
end

N = 1000

println()
println("compute key")
print_benchmark_result(computekey(40, N))

println()
println("Float64 (cache-query)")
print_benchmark_result(cachequery(Float64, 40, N))

println()
println("ComplexF64 (cache query)")
print_benchmark_result(cachequery(ComplexF64, 40, N))
