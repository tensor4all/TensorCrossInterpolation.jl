# Documentation

Documentation of all types and methods in module [TensorCrossInterpolation](https://gitlab.com/tensors4fields/TensorCrossInterpolation.jl).

```@index
```

## Matrix approximation


### Matrix cross interpolation (MCI)
```@autodocs
Modules = [TensorCrossInterpolation]
Pages = ["matrixci.jl"]
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

## Tensor trains and tensor cross Interpolation

### Tensor train (TT)
```@autodocs
Modules = [TensorCrossInterpolation]
Pages = ["abstracttensortrain.jl", "tensortrain.jl"]
```

### Tensor cross interpolation (TCI)
Note: In most cases, it is advantageous to use [`TensorCrossInterpolation.TensorCI2`](@ref) instead.
```@autodocs
Modules = [TensorCrossInterpolation, TensorCrossInterpolation.SweepStrategies]
Pages = ["TensorCI1.jl", "indexset.jl", "sweepstrategies.jl"]
```

### Tensor cross interpolation 2 (TCI2)
```@autodocs
Modules = [TensorCrossInterpolation, TensorCrossInterpolation.SweepStrategies]
Pages = ["TensorCI2.jl"]
```

## Helpers and utility methods
```@autodocs
Modules = [TensorCrossInterpolation]
Pages = ["cachedfunction.jl", "batcheval.jl", "util.jl"]
```

