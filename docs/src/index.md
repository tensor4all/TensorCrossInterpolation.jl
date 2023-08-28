```@meta
CurrentModule = TensorCrossInterpolation
```

# TensorCrossInterpolation

This is the documentation for [TensorCrossInterpolation](https://gitlab.com/tensors4fields/TensorCrossInterpolation.jl).

With the user manual and usage examples below, users should be able to use this library as a "black box" in most cases. Detailed documentation of (almost) all methods can be found in the [Documentation](@ref) section, and [Implementation](@ref) contains a detailed explanation of this implementation of TCI.

```@contents
Pages = ["index.md", "documentation.md", "implementation.md"]
```

## Interpolating functions

The most convenient way to create a TCI is [`crossinterpolate2`](@ref). For example, consider the lorentzian in 5 dimensions, i.e. $f(\mathbf v) = 1/(1 + \mathbf v^2)$ on a mesh $\mathbf{v} \in \{1, 2, ..., 10\}^5$.
$f$ can be interpolated as follows:
```@example simple
import TensorCrossInterpolation as TCI
f(v) = 1/(1 + v' * v)
localdims = fill(10, 5)    # There are 5 tensor indices, each with values 1...10
tolerance = 1e-8
tci, ranks, errors = TCI.crossinterpolate2(Float64, f, localdims; tolerance=tolerance)
println(tci)
```
Note that the return type of `f` has to be stated explicitly; inferring it automatically is too error-prone.

To evaluate the TCI approximation, simply call it the same way as the original function. For example, to evaluate it at $(1, 2, 3, 4, 5)^T$, use:
```@example simple
println("Original function: $(f([1, 2, 3, 4, 5]))")
println("TCI approximation: $(tci([1, 2, 3, 4, 5]))")
```
For easy integration into tensor network algorithms, the tensor train can be converted to ITensors MPS format. If you're using julia version 1.9 or later, an extension is automatically loaded if both `TensorCrossInterpolation.jl` and `ITensors.jl` are present.
For older versions of julia, use the package using [TCIITensorConversion.jl](https://gitlab.com/tensors4fields/tciitensorconversion.jl).

## Sums

Tensor trains are a way to efficiently obtain sums over all lattice sites, since this sum can be factorized:
```@example simple
allindices = [getindex.(Ref(i), 1:5) for i in CartesianIndices(Tuple(localdims))]
sumorig = sum(f.(allindices))
println("Sum of original function: $sumorig")

sumtt = sum(tci)
println("Sum of tensor train: $sumtt")
```
For further information, see [`sum`](@ref).

## Properties of the TCI object

After running the code above, `tci` is a [`TensorCI2`](@ref) object that can be interrogated for various properties. The most important ones are the rank (i.e. maximum bond dimension) and the link dimensions:
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
Plotting `errors` against `ranks` shows convergence behavior. In most cases, convergence will become exponential after some initial iterations. Furthermore, `tci.pivoterrors[D]` contains the error that a truncation to bond dimension `D` would incur. Plotting both, we see tha
```@example simple
plot(ranks, errors, yscale=:log10, seriestype=:scatter)
plot!(1:TCI.rank(tci), tci.pivoterrors / tci.maxsamplevalue, yscale=:log10)
ylims!(1e-10, 2)
xlabel!(L"D_\max")
ylabel!(L"\varepsilon")
```
Note that the errors are normalized with `tci.maxsamplevalue`, i.e. the maximum function value that was encountered during construction of the tci. This behaviour can be disabled by passing `normalizeerror=false` to [`crossinterpolate2`](@ref).

## Optimizing the first pivot

Sometimes, the performance of TCI is suboptimal due to an unfortunate choice of first pivot. In most cases, it is sufficient to choose a first pivot close to structure, e.g. on a maximum of the function. Simply pass the first pivot as a parameter to [`crossinterpolate2`](@ref):
```julia
firstpivot = [1, 2, 3, 4, 5]
tci, ranks, errors = TCI.crossinterpolate2(
    Float64, f, localdims, [firstpivot]; tolerance=tolerance
)
```

If this is not known (or you still have difficulties with bad convergence), the method [`optfirstpivot`](@ref) provides a simple algorithm to find a good first pivot.
```julia
firstpivot = optfirstpivot(f, localdims, [1, 2, 3, 4, 5])
tci, ranks, errors = TCI.crossinterpolate2(
    Float64, f, localdims, [firstpivot]; tolerance=tolerance
)
```
This algorithm optimizes the given index set (in this case `[1, 2, 3, 4, 5]`) by searching for a maximum absolute value, alternating through the dimensions. If no starting point is given, `[1, 1, ...]` is used.

## Caching
TODO.
