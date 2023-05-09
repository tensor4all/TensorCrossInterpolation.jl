import TensorCrossInterpolation as TCI
using Plots
using BenchmarkTools
using CUDA

function crossinterpolate(
    a::CuMatrix{T};
    tolerance=1e-6,
    maxiter=200,
    firstpivot=argmax(abs.(a))
) where {T}
    ci = TCI.MatrixCI(a, firstpivot)
    pivoterrors = Float64[]
    for iter in 1:maxiter
        pivoterror, newpivot = TCI.findmax(TCI.localerror(ci, a))
        if pivoterror < tolerance
            return ci
        end
        push!(pivoterrors, pivoterror)
        TCI.addpivot!(ci, a, newpivot)
    end
    return ci, pivoterrors
end

matrixsizes = 2 .^ (6:10)
walltimes = Float64[]
for n in matrixsizes
    println(n)
    A = CUDA.rand(2 ^ 12, 2 ^ 12)
    walltime = @elapsed ci, pivoterrors = crossinterpolate(A, maxiter=n//2)
    push!(walltimes, walltime)
end

plot(matrixsizes, walltimes, xscale=:log10, yscale=:log10, marker=:circle)
plot!(matrixsizes, walltimes[end] * (matrixsizes / matrixsizes[end]) .^ 2)
