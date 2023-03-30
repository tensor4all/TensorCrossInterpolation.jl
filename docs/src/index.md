```@meta
CurrentModule = TensorCrossInterpolation
```

# TensorCrossInterpolation

This is the documentation for [TensorCrossInterpolation](https://gitlab.com/marc.ritter/TensorCrossInterpolation.jl).

With the user manual and usage examples below, users should be able to use this library as a "black box" in most cases. Detailed documentation of (almost) all methods can be found in the [Documentation](@ref) section, and [Implementation](@ref) contains a detailed explanation of this implementation of TCI.

## Interpolating functions

The most convenient way to create a TCI is [`crossinterpolate`](@ref). For example, consider the lorentzian in 8 dimensions, i.e. $f(\mathbf v) = 1/(1 + \mathbf v^2)$, where $\mathbf{v} \in \{1, 2, ..., 10\}^8$.
We can interpolate it on a mesh with $10$ points in each dimension as follows:
```@example simple
import TensorCrossInterpolation as TCI
f(v) = 1/(1 + v' * v)
localdims = fill(10, 8)    # There are 8 tensor indices, each with values 1...10
tolerance = 1e-8
constructtime = @elapsed tci, ranks, errors = TCI.crossinterpolate(Float64, f, localdims; tolerance=tolerance)
println("$tci constructed in $constructtime s.")
```
(Note that the return type of `f` has to be stated explicitly in the call to [crossinterpolate](@ref).)

Most users will want to evaluate the tensor cross interpolation of their function. Though there is a method to evaluate [`TensorCI`](@ref) objects, this leads to repeated effort if more than a single evaluation is performed. The proper way to do this is to convert the [`TensorCI`](@ref) into a [`TensorTrain`](@ref) first, and then evaluate the tensor train. For example, to evaluate it at $(1, 2, 3, 4, 5, 6, 7, 8)^T$, use:
```@example simple
tt = TCI.TensorTrain(tci)
println("Original function: $(f([1, 2, 3, 4, 5, 6, 7, 8]))")
println("TCI approximation: $(tt([1, 2, 3, 4, 5, 6, 7, 8]))")
```
For easy integration into tensor network algorithms, the tensor train can be converted to ITensors MPS format using [TCIITensorConversion.jl](https://gitlab.com/quanticstci/tciitensorconversion.jl).

## Sums

Tensor trains are a way to efficiently obtain sums over all lattice sites, since this sum can be factorized:
```@example simple
allindices = [getindex.(Ref(i), 1:8) for i in CartesianIndices(Tuple(localdims))]
timeoriginal = @elapsed sumorig = sum(f.(allindices))
timett = @elapsed sumtt = sum(tt)
println("Sum of original function: $sumorig; took $timeoriginal s.")
println("Sum of tensor train: $sumtt; took $timett s.")
```
For further information, see [sum](@ref).

## Properties of the TCI object

After running the code above, `tci` is a [`TensorCI`](@ref) object that can be interrogated for various properties. The most important ones are the rank (i.e. maximum bond dimension) and the link dimensions:
```@example simple
println("Maximum bond dimension / rank of tci: $(TCI.rank(tci))")
println("Bond dimensions along the links of tci: $(TCI.linkdims(tci))")
```
The latter can be plotted conveniently:
```@example simple
using Plots, LaTeXStrings
bondindices = 1:length(tci)-1
plot(
    bondindices,
    min.(10 .^ bondindices, 10 .^ (length(tci) .- bondindices)),
    yscale=:log10,
    label="full rank")
plot!(bondindices, TCI.linkdims(tci), label="TCI compression")
xlabel!(L"\ell")
ylabel!(L"D_\ell")
```
Other methods are documented in the [Tensor cross interpolation (TCI)](@ref) section of the documentation.

## Checking convergence

The vectors `ranks` and `errors` contain the pivot errors reached for different maximum bond dimensions. They are intended for convergence checks. The last element of `errors` can be used to check whether the tolerance was met within the maximum number of iterations:
```@example simple
println("Error in last iteration was: $(last(errors))")
println("Is this below tolerance? $(last(errors) < tolerance)")
```
Plotting `errors` against `ranks` shows convergence behavior. In most cases, convergence will become exponential after some initial iterations.
```@example simple
plot(ranks, errors, yscale=:log10)
xlabel!(L"D_\max")
ylabel!(L"\varepsilon")
```

## Optimizing the first pivot

Sometimes, the performance of TCI is lacking due to an unfortunate choice of first pivot. In most cases, it is sufficient to choose a first pivot close to structure, e.g. on a maximum of the function. Simply pass the first pivot as a parameter to [`crossinterpolate`](@ref):
```julia
firstpivot = [1, 2, 3, 4, 5, 6, 7, 8]
tci, ranks, errors = TCI.crossinterpolate(
    Float64, f, localdims, firstpivot; tolerance=tolerance
)
```

If this is not known (or you still have difficulties with bad convergence), the method [`optfirstpivot`](@ref) provides a simple algorithm to find a good first pivot.
```julia
firstpivot = optfirstpivot(f, localdims, [1, 2, 3, 4, 5, 6, 7, 8])
tci, ranks, errors = TCI.crossinterpolate(
    Float64, f, localdims, firstpivot; tolerance=tolerance
)
```
This algorithm optimizes the given index set (in this case `[1, 2, 3, 4, 5, 6, 7, 8]`) by searching for a maximum absolute value, alternating through the dimensions. If no starting point is given, `[1, 1, ...]` is used.

## Caching
TODO.
