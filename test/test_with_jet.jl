using JET
import TensorCrossInterpolation as TCI

@testset "JET" begin
    if VERSION ≥ v"1.9"
        @test_broken JET.test_package(TCI; target_defined_modules=true)
    end
end