using Test
import TensorCrossInterpolation as TCI
using LinearAlgebra

@testset "LU decomposition" begin
    @testset "Argmax finder" begin
        A = [
            0.0698159 0.334367 -0.589437 0.145762 0.812079 -0.756145 0.295355 0.474037
            0.700284 0.53583 -0.879161 0.0259543 -0.17721 0.872417 -0.130773 0.806836
            -0.27785 0.75619 -0.6596 0.697439 0.751422 -0.694813 0.5158 -0.812036
            -0.621557 0.183863 -0.163899 -0.0200506 0.418512 0.456449 0.779305 0.771141
            -0.71849 -0.343808 0.360291 0.311619 -0.609726 0.309062 -0.214459 -0.830421
            -0.320604 -0.998123 0.45783 0.990825 -0.790207 -0.227163 -0.535666 -0.950299
            -0.136987 -0.0648093 -0.960298 0.454315 -0.722124 0.782378 0.356427 0.987233
            -0.209571 -0.0171136 0.189971 0.578491 -0.663334 -0.482773 -0.0205025 0.570071
            -0.942577 0.306031 0.696775 -0.853113 0.554776 -0.25695 0.229594 -0.0306027
            -0.490229 -0.0501003 0.163198 -0.253586 0.941586 0.0345018 0.737874 -0.963045
        ]

        @test TCI.submatrixargmax(A, [3], [5]) == (3, 5)

        @test TCI.submatrixargmax(A, :, :) == Tuple(argmax(A))
        @test TCI.submatrixargmax(A, [1], :) == (1, argmax(A[1, :]))
        @test TCI.submatrixargmax(A, :, [1]) == (argmax(A[:, 1]), 1)

        @test TCI.submatrixargmax(A, 1) == Tuple(argmax(A))
        @test TCI.submatrixargmax(A, minimum(size(A))) == (minimum(size(A)), minimum(size(A)))
    end

    @testset "Implementation of rank-revealing LU" begin
        A = [
            0.711002 0.724557 0.789335 0.382373
            0.910429 0.726781 0.719957 0.486302
            0.632716 0.39967 0.571809 0.0803125
            0.885709 0.531645 0.569399 0.481214
        ]

        LU = TCI.rrlu(A)

        @test size(LU) == size(A)

        L = TCI.left(LU; permute=false)
        @test all(L .== UnitLowerTriangular(L))

        U = TCI.right(LU; permute=false)
        @test all(U .== UpperTriangular(U))

        @test TCI.left(LU) * TCI.right(LU) ≈ A
    end
end
