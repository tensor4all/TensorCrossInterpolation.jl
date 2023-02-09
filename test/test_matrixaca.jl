using Test

using TensorCrossInterpolation
using LinearAlgebra

import TensorCrossInterpolation as TCI

@testset "MatrixACA" begin
    @testset "2x2 matrix" begin
        A = [
            1.0 0.1 -1.0
           -0.1 2.0 -1.0
            0.5 0.2 0.3
        ]

        # aca = TCI.MatrixACA{Float64}(2, 2)
        # TCI.addpivot!(A, aca, (1,1))

        aca = TCI.MatrixACA(A, (1, 1))

        @test TCI.ncols(aca) == 3
        @test TCI.nrows(aca) == 3

        @test TCI.npivots(aca) == 1
        @test aca.rowindices == [1]
        @test aca.colindices == [1]

        TCI.addpivot!(A, aca, (2, 3))

        @test TCI.npivots(aca) == 2
        @test aca.rowindices == [1, 2]
        @test aca.colindices == [1, 3]

        TCI.addpivot!(A, aca)

        @test TCI.npivots(aca) == 3
        @test aca.rowindices == [1, 2, 3]
        @test aca.colindices == [1, 3, 2]

        @test TCI.evaluate(aca) ≈ A
        @test aca[:, :] ≈ A
    end
end

nothing
