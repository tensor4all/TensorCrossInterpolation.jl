module TensorCrossInterpolation

using LinearAlgebra

# To add a method for rank(tci)
import LinearAlgebra: rank
# To define equality of IndexSet
import Base: ==
# To define iterators and element access for MCI, TCI and TT objects
import Base: isempty, iterate, length, getindex, lastindex, broadcastable, sum

export crossinterpolate, optfirstpivot, SweepStrategies
export tensortrain

include("abstractmatrixci.jl")
include("matrixci.jl")
include("matrixaca.jl")
include("indexset.jl")
include("cachedfunction.jl")
include("tensorci.jl")
include("tensortrain.jl")

end
