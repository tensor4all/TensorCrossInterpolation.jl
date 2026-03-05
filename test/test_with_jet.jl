using JET
import TensorCrossInterpolation as TCI

@testset "JET" begin
    if VERSION ≥ v"1.10"
        JET.test_package(TCI; target_modules=[TCI])
    end
end
