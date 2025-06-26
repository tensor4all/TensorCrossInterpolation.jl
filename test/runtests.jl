import TensorCrossInterpolation as TCI
using Test
using LinearAlgebra

include("test_with_aqua.jl")
include("test_with_jet.jl")
include("test_mpi.jl")
include("test_util.jl")
include("test_sweepstrategies.jl")
include("test_indexset.jl")
include("test_cachedfunction.jl")
include("test_matrixci.jl")
include("test_matrixaca.jl")
include("test_matrixlu.jl")
include("test_matrixluci.jl")
include("test_batcheval.jl")
include("test_cachedtensortrain.jl")
include("test_tensorci1.jl")
include("test_tensorci2.jl")
include("test_tensortrain.jl")
include("test_conversion.jl")
include("test_contraction.jl")
include("test_integration.jl")
include("test_globalsearch.jl")
