using Test

using LinearAlgebra
using CUDA
import TensorCrossInterpolation as TCI

@testset "MatrixACA" begin
    @testset "3x3 matrix, real" begin
        A = CuMatrix([
            1.0 0.1 -1.0
            -0.1 2.0 -1.0
            0.5 0.2 0.3
        ])

        aca = TCI.MatrixACA(A, (1, 1))

        @test TCI.ncols(aca) == 3
        @test TCI.nrows(aca) == 3

        @test TCI.npivots(aca) == 1
        @test aca.rowindices == [1]
        @test aca.colindices == [1]

        @test CUDA.@allowscalar TCI.evaluate(aca, 1, 1) ≈ A[1, 1]
        @test CUDA.@allowscalar aca[1, 1] ≈ A[1, 1]
        @test TCI.row(aca, 1) ≈ A[1, :]
        @test aca[1, :] ≈ A[1, :]
        @test TCI.col(aca, 1) ≈ A[:, 1]
        @test aca[:, 1] ≈ A[:, 1]

        TCI.addpivot!(aca, A, (2, 3))

        @test TCI.npivots(aca) == 2
        @test aca.rowindices == [1, 2]
        @test aca.colindices == [1, 3]

        @test CUDA.@allowscalar aca[2, 3] ≈ A[2, 3]
        @test CUDA.@allowscalar TCI.evaluate(aca, 2, 3) ≈ A[2, 3]
        @test aca[[1, 2], [1, 3]] ≈ A[[1, 2], [1, 3]]
        @test TCI.submatrix(aca, [1, 2], [1, 3]) ≈ A[[1, 2], [1, 3]]

        TCI.addpivot!(aca, A)

        @test TCI.npivots(aca) == 3
        @test aca.rowindices == [1, 2, 3]
        @test aca.colindices == [1, 3, 2]

        @test TCI.evaluate(aca) ≈ A
        @test aca[:, :] ≈ A
    end

    @testset "3x3 matrix, complex" begin
        A = CuMatrix([
            0.641325+0.331139im 0.63414+0.902753im 0.385012+0.359676im
            0.89194+0.783782im 0.236955+0.0828438im 0.98353+0.729723im
            0.219505+0.429946im 0.544289+0.378888im 0.14397+0.701327im
        ])

        aca = TCI.MatrixACA(A, (1, 1))

        @test TCI.ncols(aca) == 3
        @test TCI.nrows(aca) == 3

        @test TCI.npivots(aca) == 1
        @test aca.rowindices == [1]
        @test aca.colindices == [1]

        TCI.addpivot!(aca, A)
        TCI.addpivot!(aca, A)

        @test TCI.evaluate(aca) ≈ A
        @test aca[:, :] ≈ A
    end
end

nothing
