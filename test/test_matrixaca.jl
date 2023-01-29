using Test

using TensorCrossInterpolation
using LinearAlgebra

import TensorCrossInterpolation as TCI

@testset "MatrixACA" begin
    @testset "2x2 matrix" begin
        A = [
            1.0 0.1
           -0.1 2.0
        ]
        aca = TCI.MatrixACA{Float64}(2, 2)
        TCI.addpivot!(A, aca, (1,1))
        TCI.addpivot!(A, aca, (2,2))
        @test TCI.evaluate(aca) ≈ A
        @test [TCI.evaluate(aca, i, j) for i in 1:2, j in 1:2] ≈ A
    end
end

nothing