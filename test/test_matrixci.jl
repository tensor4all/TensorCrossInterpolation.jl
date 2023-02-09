using Test

using TensorCrossInterpolation
using LinearAlgebra

import TensorCrossInterpolation: nrows, ncols, addpivot!, MatrixCI, evaluate

@testset "MatrixCI" begin
    @testset "Matrix util" begin
        A = [
            0.262819 0.740968 0.505743
            0.422301 0.831443 0.32687
            0.439065 0.426132 0.453675
            0.128233 0.0490983 0.902257
            0.371653 0.810275 0.75838
        ]
        eye(n) = Matrix(LinearAlgebra.I, n, n)
        @test A ≈ TensorCrossInterpolation.AtimesBinv(A, eye(3))
        @test A ≈ TensorCrossInterpolation.AinvtimesB(eye(5), A)

        B = [
            0.852891 0.945401 0.585575
            0.800289 0.478038 0.661408
            0.685688 0.619311 0.309872
        ]
        C = [
            0.304463 0.399473 0.767147 0.337228 0.86603
            0.147815 0.508933 0.794015 0.326105 0.8079
            0.665499 0.0571589 0.766872 0.167927 0.028576
            0.411886 0.397681 0.473644 0.527007 0.4264
            0.244107 0.0669144 0.347337 0.947754 0.76624
        ]

        @test eye(3) ≈ TensorCrossInterpolation.AtimesBinv(B, B)
        @test eye(3) ≈ TensorCrossInterpolation.AinvtimesB(B, B)
        @test eye(5) ≈ TensorCrossInterpolation.AtimesBinv(C, C)
        @test eye(5) ≈ TensorCrossInterpolation.AinvtimesB(C, C)

        @test A * inv(B) ≈ TensorCrossInterpolation.AtimesBinv(A, B)
        @test inv(C) * A ≈ TensorCrossInterpolation.AinvtimesB(C, A)
    end

    @testset "Empty constructor" begin
        ci = MatrixCI(Float64, 10, 25)

        @test ci.rowindices == []
        @test ci.colindices == []
        @test ci.pivotcols == zeros(10, 0)
        @test ci.pivotrows == zeros(0, 25)
        @test nrows(ci) == 10
        @test ncols(ci) == 25
        @test size(ci) == (10, 25)
        @test rank(ci) == 0
        @test ci[:, :] ≈ zeros(size(ci))
        for i in 1:10
            @test TensorCrossInterpolation.row(ci, i) ≈ zeros(25)
            @test ci[i, :] ≈ zeros(25)
        end
        for j in 1:25
            @test TensorCrossInterpolation.col(ci, j) ≈ zeros(10)
            @test ci[:, j] ≈ zeros(10)
        end
    end

    @testset "Full constructor" begin
        A = [
            0.735188 0.718229 0.206528 0.89223 0.23432
            0.58692 0.383284 0.906576 0.3389 0.24915
            0.0866507 0.812134 0.683979 0.798798 0.63418
            0.694491 0.585013 0.623725 0.25272 0.72730
            0.100076 0.248325 0.770408 0.342828 0.080717
            0.748823 0.653965 0.47961 0.909719 0.037413
            0.902325 0.743668 0.193464 0.380086 0.91558
            0.0614368 0.0709293 0.343843 0.197515 0.45067
        ]

        rowindices = [8, 2, 3]
        colindices = [1, 5, 4]

        ci = MatrixCI(
            rowindices, colindices,
            A[:, colindices], A[rowindices, :]
        )

        @test ci.rowindices == rowindices
        @test ci.colindices == colindices
        @test ci.pivotcols == A[:, colindices]
        @test ci.pivotrows == A[rowindices, :]
        @test nrows(ci) == 8
        @test ncols(ci) == 5
        @test size(ci) == size(A)
        @test rank(ci) == 3

        Apivot = A[rowindices, colindices]
        @test TensorCrossInterpolation.pivotmatrix(ci) == Apivot
        @test TensorCrossInterpolation.leftmatrix(ci) ≈ A[:, colindices] * inv(Apivot)
        @test TensorCrossInterpolation.rightmatrix(ci) ≈ inv(Apivot) * A[rowindices, :]

        @test TensorCrossInterpolation.availablerows(ci) == [1, 4, 5, 6, 7]
        @test TensorCrossInterpolation.availablecols(ci) == [2, 3]

        for i in rowindices, j in colindices
            @test TensorCrossInterpolation.evaluate(ci, i, j) ≈ A[i, j]
            ci[i, j] ≈ A[i, j]
        end

        for i in rowindices
            @test TensorCrossInterpolation.row(ci, i)[colindices] ≈ A[i, colindices]
            @test ci[i, colindices] ≈ A[i, colindices]
        end

        for j in colindices
            @test TensorCrossInterpolation.col(ci, j)[rowindices] ≈ A[rowindices, j]
            @test ci[rowindices, j] ≈ A[rowindices, j]
        end

        @test TensorCrossInterpolation.submatrix(ci, rowindices, colindices) ≈
              A[rowindices, colindices]
        @test ci[rowindices, colindices] ≈ A[rowindices, colindices]
        @test Matrix(ci)[rowindices, colindices] ≈ A[rowindices, colindices]
    end

    @testset "Finding pivots, trivial case" begin
        A = fill(1.0, 5, 3)

        ci = MatrixCI(Float64, size(A)...)

        @test_throws DimensionMismatch addpivot!(zeros(6, 6), ci)
        @test_throws BoundsError addpivot!(A, ci, (6, 3))
        @test_throws BoundsError addpivot!(A, ci, (5, 4))
        @test_throws BoundsError addpivot!(A, ci, (-1, 2))
        @test_throws BoundsError addpivot!(A, ci, (3, -1))
        @test_throws ArgumentError TensorCrossInterpolation.findnewpivot(A, ci, zeros(Int64, 0), [2, 3])
        @test_throws ArgumentError TensorCrossInterpolation.findnewpivot(A, ci, [1, 2], zeros(Int64, 0))

        @test rank(ci) == 0

        addpivot!(A, ci, (2, 3))
        @test ci.rowindices == [2]
        @test ci.colindices == [3]
        @test ci.pivotrows == fill(1.0, 1, 3)
        @test ci.pivotcols == fill(1.0, 5, 1)
        @test rank(ci) == 1
        for i in 1:5, j in 1:3
            @test evaluate(ci, i, j) ≈ 1.0
            @test ci[i, j] ≈ 1.0
        end
        addpivot!(A, ci)
        @test ci.pivotrows == fill(1.0, 2, 3)
        @test ci.pivotcols == fill(1.0, 5, 2)
        @test length(ci.rowindices) == 2
        @test length(ci.colindices) == 2
        @test rank(ci) == 2
        addpivot!(A, ci, (TensorCrossInterpolation.availablerows(ci)[1], TensorCrossInterpolation.availablecols(ci)[1]))
        @test ci.pivotrows == fill(1.0, 3, 3)
        @test ci.pivotcols == fill(1.0, 5, 3)
        @test length(ci.rowindices) == 3
        @test length(ci.colindices) == 3
        @test rank(ci) == 3
    end


    @testset "Finding pivots, rank-1 case" begin
        A = [1.0; 2.0; 3.0] * [2.0 4.0 8.0 16.0]
        ci = MatrixCI(Float64, 3, 4)

        @test TensorCrossInterpolation.localerror(A, ci) ≈ A
        @test TensorCrossInterpolation.findnewpivot(A, ci) == ((3, 4), 48.0)
        addpivot!(A, ci)

        @test ci ≈ MatrixCI(A, (3, 4))

        @test ci.rowindices == [3]
        @test ci.colindices == [4]
        @test ci.pivotrows ≈ 3.0 .* [2.0 4.0 8.0 16.0]
        @test ci.pivotcols ≈ 16.0 .* [1.0; 2.0; 3.0]
        @test ci[:, :] ≈ A
        @test TensorCrossInterpolation.availablerows(ci) == [1, 2]
        @test TensorCrossInterpolation.availablecols(ci) == [1, 2, 3]

        addpivot!(A, ci)
        @test length(ci.rowindices) == 2
        @test length(ci.colindices) == 2
        @test unique(ci.rowindices) == ci.rowindices
        @test unique(ci.colindices) == ci.colindices
        @test size(ci.pivotrows) == (2, 4)
        @test size(ci.pivotcols) == (3, 2)
        @test ci[:, :] ≈ A

        addpivot!(A, ci)
        # From here, the ci will become singular; checking values is useless.
        @test length(ci.rowindices) == 3
        @test length(ci.colindices) == 3
        @test unique(ci.rowindices) == ci.rowindices
        @test unique(ci.colindices) == ci.colindices
        @test size(ci.pivotrows) == (3, 4)
        @test size(ci.pivotcols) == (3, 3)

        @test_throws ArgumentError TensorCrossInterpolation.findnewpivot(A, ci)
        @test_throws ArgumentError addpivot!(A, ci)
    end

    @testset "Cross interpolate smooth functions" begin
        grid = range(0, 1, length=21)

        gauss = exp.(-grid .^ 2 .- grid' .^ 2)
        cigauss = crossinterpolate(gauss)
        @test rank(cigauss) == 1
        @test nrows(cigauss) == 21
        @test ncols(cigauss) == 21
        @test cigauss.rowindices == [1]
        @test cigauss.colindices == [1]

        lorentz = 1 ./ (1 .+ grid .^ 2 .+ grid' .^ 2)
        cilorentz = crossinterpolate(lorentz, tolerance=1e-6, maxiter=10)
        @test rank(cilorentz) == 5
        @test nrows(cilorentz) == 21
        @test ncols(cilorentz) == 21
        @test Set(cilorentz.rowindices) == Set([21, 7, 12, 17, 1])
        @test Set(cilorentz.colindices) == Set([21, 7, 12, 17, 1])
    end
end
