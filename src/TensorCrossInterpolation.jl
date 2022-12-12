module TensorCrossInterpolation

using LinearAlgebra

# To add a method for rank(tci)
import LinearAlgebra: rank
# To define equality of IndexSet
import Base: ==

export MatrixCI, n_rows, n_cols, size,
    pivot_matrix, left_matrix, right_matrix, rank, Matrix,
    local_error, add_pivot!, cross_interpolate
export TensorCI, tensortrain, evaluateTCI, cross_interpolate
export MultiIndex, CachedFunction
export SweepStrategies

include("matrixci.jl")
include("indexset.jl")
include("cachedfunction.jl")
include("tensorci.jl")

end
