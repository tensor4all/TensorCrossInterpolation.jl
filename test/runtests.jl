import TensorCrossInterpolation as TCI
using Test
using LinearAlgebra

include("test_with_aqua.jl")
include("test_with_jet.jl")
include("test_util.jl")
include("test_sweepstrategies.jl")
include("test_indexset.jl")
include("test_cachedfunction.jl")
include("test_matrixci.jl")
include("test_matrixaca.jl")
include("test_matrixlu.jl")
include("test_matrixluci.jl")
include("test_batcheval.jl")
include("test_tensorci1.jl")
include("test_tensorci2.jl")
include("test_tensortrain.jl")
include("test_integration.jl")

#==
if VERSION.major >= 2 || (VERSION.major == 1 && VERSION.minor >= 9)
    @testset "ITensor conversion interface" begin
        include("test_ttmpsconversion.jl")
        include("test_mpsutil.jl")
    end
end
==#
