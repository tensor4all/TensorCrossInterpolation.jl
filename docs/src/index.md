```@meta
CurrentModule = TensorCrossInterpolation
```

# TensorCrossInterpolation

This is the documentation for [TensorCrossInterpolation](https://gitlab.com/marc.ritter/TensorCrossInterpolation.jl).

With the user manual and usage examples below, users should be able to use this library as a "black box" in most cases. Detailed documentation of (almost) all methods can be found in the [Documentation](@ref) section, and [Implementation](@ref) contains a detailed explanation of this implementation of TCI.

## Basic Usage

The most convenient way to create a TCI is [`crossinterpolate`](@ref). For example, an approximation to the 4d lorentzian $f(\mathbf v) = 1/(1 + \mathbf v^2)$ on a mesh with $10^4$ points is obtained as follows:
```@example simple
import TensorCrossInterpolation as TCI
f(v) = 1/(1 + v' * v)
localdims = [10, 10, 10, 10]    # There are 4 tensor indices, each with values 1...10
tolerance = 1e-8
# In the next line, the return type of f (in this case Float64) has to be stated explicitly.
tci, ranks, errors = TCI.crossinterpolate(Float64, f, localdims; tolerance=tolerance)
println(tci)
```

Now, `tci` is a [`TensorCI`](@ref) object that can be interrogated for various properties, such as the link dimensions:
```@example simple
println("Bond dimensions of the links of tci: $(TCI.linkdims(tci))")
```
Other methods are documented in the [Tensor cross interpolation (TCI)](@ref) section of the documentation.

The vectors `ranks` and `errors` contain the pivot errors reached for different maximum bond dimensions. They are intended for convergence checks; plotting `errors` against `ranks` gives an idea of the convergence. The last element of `errors` can be used to check for convergence:
```@example simple
println("Error in last iteration was: $(last(errors))")
println("Is this below tolerance? $(last(errors) < tolerance)")
```

Most users will want to evaluate the tensor cross interpolation of their function. Though there is a method to evaluate [`TensorCI`](@ref) objects, this leads to repeated effort if more than a single evaluation is performed. The proper way to do this is to convert the [`TensorCI`](@ref) into a [`TensorTrain`](@ref) first, and then evaluate the tensor train like this:
```@example simple
tt = TCI.TensorTrain(tci)
println("Original function: $(f([1, 2, 3, 4]))")
println("TCI approximation: $(TCI.evaluate(tt, [1, 2, 3, 4]))")
```

This tensor train can be converted to ITensors MPS format using [TCIITensorConversion.jl](https://gitlab.com/quanticstci/tciitensorconversion.jl).

## Optimizing the first pivot

Sometimes, the performance of TCI is lacking due to an unfortunate choice of first pivot. In most cases, it is sufficient to choose a first pivot close to structure, e.g. on a maximum of the function. Simply pass the first pivot as a parameter to [`crossinterpolate`](@ref):
```julia
firstpivot = [1, 2, 3, 4]
tci, ranks, errors = TCI.crossinterpolate(
    Float64, f, localdims, firstpivot; tolerance=tolerance
)
```

If this is not known (or you still have difficulties with bad convergence), the method [`optfirstpivot`](@ref) provides a simple algorithm to find a good first pivot.
```julia
firstpivot = optfirstpivot(f, localdims, [1, 2, 3, 4])
tci, ranks, errors = TCI.crossinterpolate(
    Float64, f, localdims, firstpivot; tolerance=tolerance
)
```
This algorithm optimizes the given index set (in this case `[1, 2, 3, 4]`) by searching for a maximum absolute value, alternating through the dimensions. If no starting point is given, `[1, 1, ...]` is used.

## Caching
TODO.
