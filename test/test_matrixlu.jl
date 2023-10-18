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

    @testset "Argmax finder test_throws" begin
        @test_throws ArgumentError("rows must not be empty") TCI.submatrixargmax(rand(10, 10), 100)
        @test_throws ArgumentError("cols must not be empty") TCI.submatrixargmax(rand(10, 10), 3, [])

        @test_throws ArgumentError("rows ⊆ axes(A, 1) must be satified") TCI.submatrixargmax(rand(10, 10), [1, 100, 1000], [1])
        @test_throws ArgumentError("cols ⊆ axes(A, 2) must be satified") TCI.submatrixargmax(rand(10, 10), [1], [1, 100, 1000])
    end

    @testset "Argmax finder A::Matrix{ComplexF64}" begin
        A = ComplexF64[
            0 1 2 3 4 5
            1 +im 2+im 3+im 4+im 5+im
            1 +2im 2+2im 3+2im 4+2im 5+2im
        ]
        @test TCI.submatrixargmax(abs2, A, [3], [5]) == (3, 5)

        @test TCI.submatrixargmax(abs2, A, :, :) == Tuple(argmax(abs2.(A)))
        @test TCI.submatrixargmax(abs2, A, [1], :) == (1, argmax(abs2.(A[1, :])))
        @test TCI.submatrixargmax(abs2, A, :, [1]) == (argmax(abs2.(A[:, 1])), 1)

        @test TCI.submatrixargmax(abs2, A, 1) == Tuple(argmax(abs2.(A)))
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

    @testset "Implementation of approximate rank-revealing LU" begin
        A = [
            0.711002 0.724557 0.789335 0.382373
            0.910429 0.726781 0.719957 0.486302
            0.632716 0.39967 0.571809 0.0803125
            0.885709 0.531645 0.569399 0.481214
        ]

        LU = TCI.arrlu(Float64, (i, j) -> A[i, j], size(A), [1], [1])
        @test size(LU) == size(A)
        L = TCI.left(LU; permute=false)
        @test all(L .== UnitLowerTriangular(L))
        U = TCI.right(LU; permute=false)
        @test all(U .== UpperTriangular(U))
        @test TCI.left(LU) * TCI.right(LU) ≈ A
    end

    @testset "Approximation in rrLU" begin
        A = [
            0.684025 0.784249 0.826742 0.054321 0.0234695 0.467096
            0.73928 0.295516 0.877126 0.111711 0.103509 0.653785
            0.394016 0.753239 0.889128 0.291669 0.873509 0.0965536
            0.378539 0.0123737 0.20112 0.758088 0.973042 0.308372
            0.235156 0.51939 0.788184 0.363171 0.230001 0.984971
            0.893223 0.220834 0.18001 0.258537 0.396583 0.142105
            0.0417881 0.890706 0.328631 0.279332 0.963188 0.706944
            0.914298 0.792345 0.311083 0.129653 0.350062 0.683966
        ]

        LU = TCI.rrlu(A, maxrank=4)
        @test size(LU) == size(A)
        @test length(TCI.rowindices(LU)) == 4
        @test length(TCI.colindices(LU)) == 4

        L = TCI.left(LU; permute=false)
        @test size(L) == (size(A, 1), 4)
        @test all(L .== tril(L))
        U = TCI.right(LU; permute=false)
        @test size(U) == (4, size(A, 2))
        @test all(U .== triu(U))

        A = hcat(A, A .+ 1e-3 * rand(8, 6))

        LU = TCI.rrlu(A, reltol=1e-2)
        @test size(LU) == size(A)
        @test length(TCI.rowindices(LU)) < size(A, 1)
        @test length(TCI.colindices(LU)) < size(A, 2)

        L = TCI.left(LU; permute=false)
        @test size(L, 1) == size(A, 1)
        @test all(L .== tril(L))
        U = TCI.right(LU; permute=false)
        @test size(U, 2) == size(A, 2)
        @test all(U .== triu(U))
        @test size(L, 2) == size(U, 1)

        @test maximum(abs.(TCI.left(LU) * TCI.right(LU) .- A)) < 1e-2
    end

    @testset "lastpivoterror for full-rank matrix" begin
        A = [
            1.0 0.0;
            0.0 1.0
        ]
        LU1 = TCI.rrlu(A)
        @test TCI.pivoterrors(LU1) == [1.0, 1.0]
        @test TCI.lastpivoterror(LU1) == 0.0
    end
end
