module TensorCrossInterpolation

using LinearAlgebra

# To add a method for rank(tci)
import LinearAlgebra: rank
# To define equality of IndexSet
import Base: ==

export MatrixCI, nrows, ncols, size
export pivotmatrix, leftmatrix, rightmatrix, rank, Matrix
export localerror, addpivot!, crossinterpolate
export TensorCI, linkdims, tensortrain, evaluate, crossinterpolate
export MultiIndex, CachedFunction
export SweepStrategies

include("matrixci.jl")
include("indexset.jl")
include("cachedfunction.jl")
include("tensorci.jl")

end
