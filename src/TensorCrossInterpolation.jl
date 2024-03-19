module TensorCrossInterpolation

using LinearAlgebra
using EllipsisNotation

# To add a method for rank(tci)
import LinearAlgebra: rank, diag
import LinearAlgebra as LA
# To define equality of IndexSet, and TT addition
import Base: ==, +
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
include("abstracttensortrain.jl")
include("cachedtensortrain.jl")
include("batcheval.jl")
include("cachedfunction.jl")
include("tensorci1.jl")
include("tensorci2.jl")
include("tensortrain.jl")
include("conversion.jl")

end
