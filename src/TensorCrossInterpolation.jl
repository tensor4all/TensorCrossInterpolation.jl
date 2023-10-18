module TensorCrossInterpolation

using LinearAlgebra

# To add a method for rank(tci)
import LinearAlgebra: rank
# To define equality of IndexSet
import Base: ==
# To define iterators and element access for MCI, TCI and TT objects
import Base: isempty, iterate, getindex, lastindex, broadcastable
import Base: length, size, sum

export crossinterpolate, crossinterpolate2, optfirstpivot, SweepStrategies
export tensortrain

include("util.jl")
include("sweepstrategies.jl")
include("abstractmatrixci.jl")
include("matrixci.jl")
include("matrixaca.jl")
include("matrixlu.jl")
include("matrixluci.jl")
include("indexset.jl")
include("cachedfunction.jl")
include("abstracttensortrain.jl")
include("cachedtensortrain.jl")
include("tensorci.jl")
include("tensorci2.jl")
include("tensortrain.jl")

end
