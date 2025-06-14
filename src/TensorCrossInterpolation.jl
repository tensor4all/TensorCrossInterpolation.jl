module TensorCrossInterpolation

using LinearAlgebra
using EllipsisNotation
using BitIntegers
import QuadGK

# To add a method for rank(tci)
import LinearAlgebra: rank, diag
import LinearAlgebra as LA
# To define equality of IndexSet, and TT addition
import Base: ==, +
# To define iterators and element access for MCI, TCI and TT objects
import Base: isempty, iterate, getindex, lastindex, broadcastable
import Base: length, size, sum
import Random

export crossinterpolate1, crossinterpolate2, optfirstpivot
export tensortrain, TensorTrain, sitedims, evaluate

export contract

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
include("tensortrain.jl")
include("tensorci1.jl")
include("globalpivotfinder.jl")
include("tensorci2.jl")
include("conversion.jl")
include("integration.jl")
include("contraction.jl")
include("globalsearch.jl")

end
