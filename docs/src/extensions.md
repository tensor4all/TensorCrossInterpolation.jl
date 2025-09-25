# Extensions

This page documents extensions to [TensorCrossInterpolation](https://github.com/tensor4all/TensorCrossInterpolation.jl).

Currently, there is only one extension, TCIITensorConversion.

## TCIITensorConversion

Dependencies:
- [ITensors.jl](https://github.com/ITensor/ITensors.jl)
- [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl)

This module is automatically loaded when ITensors and ITensorMPS are present in addition to TensorCrossInterpolation. It offers conversion constructors between ITensorMPS types and TensorCrossInterpolation types, and an analogue of the [`evaluate`](@ref) function for ITensor types.

```@autodocs
Modules = [Base.get_extension(TensorCrossInterpolation, :TCIITensorConversion)]
```
