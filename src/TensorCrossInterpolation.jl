module TensorCrossInterpolation

using LinearAlgebra

# To add a method for rank(tci)
import LinearAlgebra: rank
# To define equality of IndexSet
import Base: ==, isempty

export crossinterpolate
export optfirstpivot, tensortrain
export SweepStrategies

include("abstractmatrixci.jl")
include("matrixci.jl")
include("matrixaca.jl")
include("indexset.jl")
include("cachedfunction.jl")
include("tensorci.jl")

end
