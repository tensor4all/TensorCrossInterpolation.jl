# TensorCrossInterpolation

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensors4fields.gitlab.io/tensorcrossinterpolation.jl/dev/index.html)
[![Build Status](https://gitlab.com/tensors4fields/tensorcrossinterpolation.jl/badges/main/pipeline.svg)](https://gitlab.com/tensors4fields/tensorcrossinterpolation.jl/-/pipelines)
[![Coverage](https://gitlab.com/tensors4fields/tensorcrossinterpolation.jl/badges/main/coverage.svg)](https://gitlab.com/tensors4fields/tensorcrossinterpolation.jl/-/commits/main)

The [TensorCrossInterpolation module](https://gitlab.com/tensors4fields/tensorcrossinterpolation.jl) implements the *tensor cross interpolation* algorithm for efficient interpolation of multi-index tensors and multivariate functions.

This algorithm is used in the *quantics tensor cross interpolation* (QTCI) method for exponentially efficient interpolation of functions with scale separation. QTCI is implemented in the [QuanticsTCI.jl](https://gitlab.com/tensors4fields/quanticstci.jl) module.

## Installation

`TensorCrossInterpolation.jl` has been registered to the General registry. Installing the package is as simple as
```julia
]
add TensorCrossInterpolation
```

## Usage

*This section only contains the bare minimum to get you started. An example with more explanation can be found in the [user manual](https://tensors4fields.gitlab.io/tensorcrossinterpolation.jl/dev/index.html).*

Given a multivariate function `f`, the function `crossinterpolate2` will generate a tensor cross interpolation for `f`. For example, to interpolate the 8d lorentzian $f(\mathbf v) = 1/(1 + \mathbf v^2)$ on an 8-dimensional lattice of integers, $\mathbf{v} \in \{1, 2, ..., 10\}^8$:
```julia
import TensorCrossInterpolation as TCI
f(v) = 1/(1 + v' * v)
# There are 8 tensor indices, each with values 1...10
localdims = fill(10, 8)
tolerance = 1e-8
tci, ranks, errors = TCI.crossinterpolate2(Float64, f, localdims; tolerance=tolerance)
```
Note:
- `f` is defined as a function that takes a single `Vector` of integers.
- The return type of `f` (`Float64` in this case) must be stated explicitly in the call to `crossinterpolate2`.

The resulting `TensorCI2` object can be further manipulated, see [user manual](https://tensors4fields.gitlab.io/tensorcrossinterpolation.jl/dev/index.html).
To evaluate the TCI interpolation, simply call your `TensorCI1` object like you would call the original function:
```julia
originalvalue = f([1, 2, 3, 4, 5, 6, 7, 8])
interpolatedvalue = tci([1, 2, 3, 4, 5, 6, 7, 8])
```
The sum of all function values on the lattice can be obtained very efficiently from a tensor train:
```julia
sumvalue = sum(tci)
```

## Online user manual
An example with more explanation can be found in the [user manual](https://tensors4fields.gitlab.io/tensorcrossinterpolation.jl/dev/index.html).

## Related modules

### [TCIITensorConversion.jl](https://gitlab.com/tensors4fields/tciitensorconversion.jl)
A small helper module for easy conversion of `TensorCI1`, `TensorCI2` and `TensorTrain` objects into ITensors `MPS` objects. This should be helpful for those integrating TCI into a larger tensor network algorithm.
For this conversion, simply call the `MPS` constructor on the object:
```julia
mps = MPS(tci)
```

### [QuanticsTCI.jl](https://gitlab.com/tensors4fields/QuanticsTCI.jl)
A module that implements the *quantics representation* and combines it with TCI for exponentially efficient interpolation of functions with scale separation.

## Contributions

- If you are having have technical trouble, feel free to contact me directly.

- Feature requests and bug reports are always welcome; please open an issue for those.

---

## References

- Y. Núñez Fernández, M. Jeannin, P. T. Dumitrescu, T. Kloss, J. Kaye, O. Parcollet, and X. Waintal, *Learning Feynman Diagrams with Tensor Trains*, [Phys. Rev. X 12, 041018 (2022)](https://link.aps.org/doi/10.1103/PhysRevX.12.041018).
(arxiv link: [arXiv:2207.06135](http://arxiv.org/abs/2207.06135))
- I. V. Oseledets, Tensor-Train Decomposition, [SIAM J. Sci. Comput. 33, 2295 (2011)](https://epubs.siam.org/doi/10.1137/090752286).
