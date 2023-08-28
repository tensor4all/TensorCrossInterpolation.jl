module TCIITensorConversion

import TensorCrossInterpolation as TCI
using ITensors

export MPS
export evaluate_mps

include("ttmpsconversion.jl")
include("mpsutil.jl")

end
