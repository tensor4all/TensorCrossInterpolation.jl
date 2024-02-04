using JET
import TensorCrossInterpolation as TCI

@testset "JET" begin
    JET.test_package(TCI; target_defined_modules=true)
end