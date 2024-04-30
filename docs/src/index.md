```@meta
CurrentModule = TensorCrossInterpolation
```

# TensorCrossInterpolation

This is the documentation for [TensorCrossInterpolation](https://gitlab.com/tensors4fields/TensorCrossInterpolation.jl).

With the user manual and usage examples below, users should be able to use this library as a "black box" in most cases. Detailed documentation of (almost) all methods can be found in the [Documentation](@ref) section, and [Implementation](@ref) contains a detailed explanation of this implementation of TCI.

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

## Sums and Integrals

Tensor trains are a way to efficiently obtain sums over all lattice sites, since this sum can be factorized:
```@example simple
allindices = [getindex.(Ref(i), 1:5) for i in CartesianIndices(Tuple(localdims))]
sumorig = sum(f.(allindices))
println("Sum of original function: $sumorig")

sumtt = sum(tci)
println("Sum of tensor train: $sumtt")
```
For further information, see [`sum`](@ref).

This factorized sum can be used for efficient evaluation of high-dimensional integrals. This is implemented with Gauss-Kronrod quadrature rules in [`integrate`](@ref). For example, the integral
```math
I = 10^3 \int\limits_{[-1, +1]^{10}} d^{10} \vec{x} \,
     \cos\!\left(10 \textstyle\sum_{n=1}^{10} x_n^2 \right)
     \exp\!\left[-10^{-3}\left(\textstyle\sum_{n=1}^{10} x_n\right)^4\right]
```
is evaluated by the following code:
```@example simple
function f(x)
    return 1e3 * cos(10 * sum(x .^ 2)) * exp(-sum(x)^4 / 1e3)
end
I = TCI.integrate(Float64, f, fill(-1.0, 10), fill(+1.0, 10); GKorder=15, tolerance=1e-8)
println("GK15 integral value: $I")
```
The argument `GKorder` controls the Gauss-Kronrod quadrature rule used for the integration, and `tolerance` controls the tolerance in the TCI approximation, which is distinct from the tolerance in the integral. For complicated functions, it is recommended to integrate using two different GK rules and to compare the results to get a good estimate of the discretization error.

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
plot!(tci.pivoterrors / tci.maxsamplevalue, yscale=:log10)
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

## Estiamte true interpolation error by random global search
Since the TCI update algorithms are local, the true interpolation error is not known. However, the error can be estimated by global searches. This is implemented in the function `estimatetrueerror`:

```julia
pivoterrors = TCI.estimatetrueerror(TCI.TensorTrain(tci), f)
```

This function approximately estimates the error that would be reached by repeating a greedy search from a random initial point.
The result is a vector of a found indexset and the corresponding error, sorted by error. The error is the maximum absolute difference between the function and the TT approximation.

## Caching
During constructing a TCI, the function to be interpolated can be evaluated for the same index set multiple times.
If an evaluation of the function to be interpolated is costly, i.e., takes more than 100 ns,
it may be beneficial to cache the results of function evaluations.
`CachedFunction{T}` provides this functionality.

We can wrap your function as follows:

```Julia
import TensorCrossInterpolation as TCI

# Local dimensions of TCI
localdims = [2, 2, 2, 2]

# Function to be interpolated. Evaluation take 2 seconds.
f(x) = (sleep(2); sum(x))

# Cached Function. `T` is the return type of the function.
cf = TCI.CachedFunction{Float64}(f, localdims)

# The first evaluation takes two seconds. The result will be cached.
x = [1, 1, 1, 1]
@time cf(x)

# Cached value is returned (Really quick!).
@time cf(x)
```

## Batch Evalaution
By default, in TCI2, the function to be interpolated is evaluated for a single index at a time. However, there may be a need to parallelize the code by evaluating the function across multiple index sets concurrently using several CPU cores. This type of custom optimization can be achieved through batch evaluation.

To utilize this feature, your function must inherit from  `TCI.BatchEvaluator{T}` and supports two additional types of function calls for evaluating $\mathrm{T}$ (one local index) and $\Pi$ tensors (two local indices):

```julia
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: MultiIndex

struct TestFunction{T} <: TCI.BatchEvaluator{T}
    localdims::Vector{Int}
    function TestFunction{T}(localdims) where {T}
        new{T}(localdims)
    end
end

# Evaluation for a single index set
function (obj::TestFunction{T})(indexset::MultiIndex)::T where {T}
    return sum(indexset)
end


# Evaluaiton of a T tensor with one local index
function (obj::TestFunction{T})(leftindexset::Vector{MultiIndex}, rightindexset::Vector{MultiIndex}, ::Val{1})::Array{T,3} where {T}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,3}(undef, 0, 0, 0, 0)
    end

    nl = length(leftindexset[1])
    # This can be parallelized if you want
    result = [obj(vcat(l, s1, r)) for l in leftindexset, s1 in 1:obj.localdims[nl+1], r in rightindexset]
    return reshape(result, length(leftindexset), obj.localdims[nl+1], length(rightindexset))
end


# Evaluaiton of a Pi tensor with two local indices
function (obj::TestFunction{T})(leftindexset::Vector{MultiIndex}, rightindexset::Vector{MultiIndex}, ::Val{2})::Array{T,4} where {T}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,4}(undef, 0, 0, 0, 0)
    end

    nl = length(leftindexset[1])
    # This can be parallelized if you want
    result = [obj(vcat(l, s1, s2, r)) for l in leftindexset, s1 in 1:obj.localdims[nl+1], s2 in 1:obj.localdims[nl+2], r in rightindexset]
    return reshape(result, length(leftindexset), obj.localdims[nl+1:nl+2]..., length(rightindexset))
end

localdims = [2, 2, 2, 2, 2]
f = TestFunction{Float64}(localdims)

# Compute T tensor
let
    leftindexset = [[1, 1, 1], [2, 2, 1], [2, 1, 1]]
    rightindexset = [[1, 1], [2, 2], [1, 2]]

    # The returned object has shape of (3, 2, 3)
    @assert f(leftindexset, rightindexset, Val(1)) ≈ [sum(vcat(l, s1, r)) for l in leftindexset, s1 in 1:localdims[4], r in rightindexset]
end

# Compute Pi tensor
let
    leftindexset = [[1, 1], [2, 2], [2, 1]]
    rightindexset = [[1, 1], [2, 2], [1, 2]]

    # The returned object has shape of (3, 2, 2, 3)
    @assert f(leftindexset, rightindexset, Val(2)) ≈ [sum(vcat(l, s1, s2, r)) for l in leftindexset, s1 in 1:localdims[3], s2 in 1:localdims[4], r in rightindexset]
end
```

`CachedFunction{T}`  can wrap a function inheriting from `BatchEvaluator{T}`. In such cases, `CachedFunction{T}`  caches the results of batch evaluation.

# Batch evaluation + parallelization
The batch evalution can be combined with parallelization using threads, MPI, etc.
The following sample code use `Threads` to parallelize function evaluations.
Note that the function evaluation for a single index set must be thread-safe.

We can run the code as (with 6 threads):

```Bash
julia --project=@. -t 6 samplecode.jl
```

```Julia
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: MultiIndex

struct TestFunction{T} <: TCI.BatchEvaluator{T}
    localdims::Vector{Int}
    function TestFunction{T}(localdims) where {T}
        new{T}(localdims)
    end
end

# Evaluation for a single index set (takes 1 millisec)
function (obj::TestFunction{T})(indexset::MultiIndex)::T where {T}
    sleep(1e-3)
    return sum(indexset)
end


# Batch evaluation (loop over all index sets)
function (obj::TestFunction{T})(leftindexset::Vector{Vector{Int}}, rightindexset::Vector{Vector{Int}}, ::Val{M})::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M+2}(undef, ntuple(i->0, M+2)...)
    end

    nl = length(first(leftindexset))

    t = time_ns()
    cindexset = vec(collect(Iterators.product(ntuple(i->1:obj.localdims[nl+i], M)...)))
    elements = collect(Iterators.product(1:length(leftindexset), 1:length(cindexset), 1:length(rightindexset)))
    result = Array{T,3}(undef, length(leftindexset), length(cindexset), length(rightindexset))
    t2 = time_ns()

    Threads.@threads for indices in elements
        l, c, r = leftindexset[indices[1]], cindexset[indices[2]], rightindexset[indices[3]]
        result[indices...] = obj(vcat(l, c..., r))
    end
    t3 = time_ns()
    println("Time: ", (t2-t)/1e9, " ", (t3-t2)/1e9)
    return reshape(result, length(leftindexset), obj.localdims[nl+1:nl+M]..., length(rightindexset))
end


L = 20
localdims = fill(2, L)
f = TestFunction{Float64}(localdims)

println("Number of threads: ", Threads.nthreads())

# Compute Pi tensor
nl = 10
nr = L - nl - 2

# 20 left index sets, 20 right index sets
leftindexset = [[rand(1:d) for d in localdims[1:nl]] for _ in 1:20]
rightindexset = [[rand(1:d) for d in localdims[nl+3:end]] for _ in 1:20]

f(leftindexset, rightindexset, Val(2))

for i in 1:4
    @time f(leftindexset, rightindexset, Val(2))
end
```

If your function is thread-safe, you can parallelize your function readily using `ThreadedBatchEvaluator` as follows (the internal implementation is identical to the sample code shown above):

```Julia
import TensorCrossInterpolation as TCI

# Evaluation takes 1 millisecond, make sure the function is thread-safe.
function f(x)
    sleep(1e-3)
    return sum(x)
end


L = 20
localdims = fill(2, L)
parf = TCI.ThreadedBatchEvaluator{Float64}(f, localdims)

println("Number of threads: ", Threads.nthreads())

# Compute Pi tensor
nl = 10
nr = L - nl - 2

# 20 left index sets, 20 right index sets
leftindexset = [[rand(1:d) for d in localdims[1:nl]] for _ in 1:20]
rightindexset = [[rand(1:d) for d in localdims[nl+3:end]] for _ in 1:20]

parf(leftindexset, rightindexset, Val(2))

for i in 1:4
    @time parf(leftindexset, rightindexset, Val(2))
end
```

You can simply pass the wrapped function `parf` to `crossinterpolate2`.
