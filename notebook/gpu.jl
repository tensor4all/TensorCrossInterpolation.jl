using LinearAlgebra
import TensorCrossInterpolation as TCI
using Plots
using CUDA
using BenchmarkTools

CUDA.versioninfo()
println("CUDA is functional: $(CUDA.functional())")

lorentz(x) = 1 / (1 + sum(x .^ 2))

xvals = -3:0.3:3
lorentzi(index) = lorentz([xvals[i] for i in index])

d = length(xvals)
c = div(d, 2) + 1
localdims = [d for i in 1:5]
z = [lorentz((x, y)) for x in xvals, y in xvals]

tci, ranks, errors = TCI.crossinterpolate(Float64, lorentzi, localdims; verbosity=1, tolerance=1e-12)

tt = TCI.TensorTrain(tci)
@benchmark ztt = [tt(x, y, c, c, c) for x in eachindex(xvals), y in eachindex(xvals)]
ttcuda = TCI.TensorTrainCUDA(tci)
@benchmark zttcuda = [ttcuda(x, y, c, c, c) for x in eachindex(xvals), y in eachindex(xvals)]

c1 = contour(xvals, xvals, z; fill=true)

c2 = contour(xvals, xvals, ztt; fill=true)
c3 = contour(xvals, xvals, log.(abs.(z .- ztt)); fill=true)

c2 = contour(xvals, xvals, zttcuda; fill=true)
c3 = contour(xvals, xvals, log.(abs.(z .- zttcuda)); fill=true)

p4 = plot(
    ranks, errors, yscale=:log10, marker=:circle,
    ylim=[0.5 * minimum(errors[errors .> 0]), 2 * maximum(errors)]
)

println(maximum(abs.(z .- ztt)))
plot(c1, 4, c2, c3, c4, c5, layout=(3, 2), size=(600,750))

savefig("gputest.pdf")
