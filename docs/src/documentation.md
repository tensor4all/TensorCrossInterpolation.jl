# Documentation

Documentation of all types and methods in module [TensorCrossInterpolation](https://gitlab.com/marc.ritter/TensorCrossInterpolation.jl).

```@index
```

## Matrix approximation

### Shared infrastructure

```@autodocs
Modules = [TensorCrossInterpolation]
Pages = ["abstractmatrixci.jl"]
```

### Matrix cross interpolation (MCI)

```@autodocs
Modules = [TensorCrossInterpolation]
Pages = ["abstractmatrixci.jl", "matrixci.jl"]
```

### Adaptive cross approximation (ACA)

```@autodocs
Modules = [TensorCrossInterpolation]
Pages = ["matrixaca.jl"]
```

### Rank-revealing LU decomposition (rrLU)
```@autodocs
Modules = [TensorCrossInterpolation]
Pages = ["matrixlu.jl", "matrixluci.jl"]
```

## Tensor cross interpolation (TCI)

```@autodocs
Modules = [TensorCrossInterpolation, TensorCrossInterpolation.SweepStrategies]
Pages = ["tensorci.jl", "indexset.jl", "sweepstrategies.jl"]
```

## Tensor cross interpolation 2 (TCI2)

```@autodocs
Modules = [TensorCrossInterpolation, TensorCrossInterpolation.SweepStrategies]
Pages = ["tensorci2.jl"]
```

## Tensor train (TT)

```@autodocs
Modules = [TensorCrossInterpolation]
Pages = ["abstracttensortrain.jl", "tensortrain.jl"]
```

## Helpers and utility methods

```@autodocs
Modules = [TensorCrossInterpolation]
Pages = ["cachedfunction.jl", "util.jl"]
```

