module TensorCrossInterpolation

using LinearAlgebra

# To add methods to rank
import LinearAlgebra: rank

export MatrixCrossInterpolation, n_rows, n_cols, size,
    pivot_matrix, left_matrix, right_matrix, rank, Matrix,
    local_error, add_pivot!, cross_interpolate

include("MatrixCI.jl")

end
