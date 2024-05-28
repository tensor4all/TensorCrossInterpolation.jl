# TensorCrossInterpolation

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensor4all.github.io/TensorCrossInterpolation.jl/dev)
[![CI](https://github.com/tensor4all/TensorCrossInterpolation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/tensor4all/TensorCrossInterpolation.jl/actions/workflows/CI.yml)

The [TensorCrossInterpolation module](https://github.com/tensor4all/TensorCrossInterpolation.jl) implements the *tensor cross interpolation* algorithm for efficient interpolation of multi-index tensors and multivariate functions.

This algorithm is used in the *quantics tensor cross interpolation* (QTCI) method for exponentially efficient interpolation of functions with scale separation. QTCI is implemented in the [QuanticsTCI.jl](https://github.com/tensor4all/quanticstci.jl) module.

## Installation

This module has been registered in the General registry. It can be installed by typing the following in a Julia REPL:
```julia
using Pkg; Pkg.add("TensorCrossInterpolation.jl")
```

## Usage

*This section only contains the bare minimum to get you started. An example with more explanation can be found in the [user manual](https://tensor4all.github.io/TensorCrossInterpolation.jl/dev).*

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

The resulting `TensorCI2` object can be further manipulated, see [user manual](https://tensor4all.github.io/TensorCrossInterpolation.jl/dev).
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
An example with more explanation can be found in the [user manual](https://tensor4all.github.io/TensorCrossInterpolation.jl/dev).

## Related modules

### [TCIITensorConversion.jl](https://github.com/tensor4all/tciitensorconversion.jl)
A small helper module for easy conversion of `TensorCI1`, `TensorCI2` and `TensorTrain` objects into ITensors `MPS` objects. This should be helpful for those integrating TCI into a larger tensor network algorithm.
For this conversion, simply call the `MPS` constructor on the object:
```julia
mps = MPS(tci)
```

### [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl)
A module that implements the *quantics representation* and combines it with TCI for exponentially efficient interpolation of functions with scale separation.

## Contributions

- If you are having have technical trouble, feel free to contact me directly.
- Feature requests and bug reports are always welcome, feel free to open an [issue](https://github.com/tensor4all/TensorCrossInterpolation.jl/-/issues) for those.
- If you have implemented something that might be useful for others, we'd appreciate a [merge request](https://github.com/tensor4all/TensorCrossInterpolation.jl/-/merge_requests)!

## Authors

This project is maintained by
- Marc K. Ritter @marc_ritter
- Hiroshi Shinaoka @h.shinaoka

For their contributions to this library's code, we thank
- Satoshi Terasaki @terasakisatoshi

---

## References

- Y. Núñez Fernández, M. Jeannin, P. T. Dumitrescu, T. Kloss, J. Kaye, O. Parcollet, and X. Waintal, *Learning Feynman Diagrams with Tensor Trains*, [Phys. Rev. X 12, 041018 (2022)](https://link.aps.org/doi/10.1103/PhysRevX.12.041018).
(arxiv link: [arXiv:2207.06135](http://arxiv.org/abs/2207.06135))
- I. V. Oseledets, Tensor-Train Decomposition, [SIAM J. Sci. Comput. 33, 2295 (2011)](https://epubs.siam.org/doi/10.1137/090752286).
