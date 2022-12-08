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
        ci = TensorCrossInterpolation.MatrixCI{Float64}(10, 25)

        @test ci.row_indices == []
        @test ci.col_indices == []
        @test ci.pivot_cols == zeros(10, 0)
        @test ci.pivot_rows == zeros(0, 25)
        @test n_rows(ci) == 10
        @test n_cols(ci) == 25
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

        row_indices = [8, 2, 3]
        col_indices = [1, 5, 4]

        ci = MatrixCI(
            row_indices, col_indices,
            A[:, col_indices], A[row_indices, :]
        )

        @test ci.row_indices == row_indices
        @test ci.col_indices == col_indices
        @test ci.pivot_cols == A[:, col_indices]
        @test ci.pivot_rows == A[row_indices, :]
        @test n_rows(ci) == 8
        @test n_cols(ci) == 5
        @test size(ci) == size(A)
        @test rank(ci) == 3

        Apivot = A[row_indices, col_indices]
        @test pivot_matrix(ci) == Apivot
        @test left_matrix(ci) ≈ A[:, col_indices] * inv(Apivot)
        @test right_matrix(ci) ≈ inv(Apivot) * A[row_indices, :]

        @test TensorCrossInterpolation.avail_rows(ci) == [1, 4, 5, 6, 7]
        @test TensorCrossInterpolation.avail_cols(ci) == [2, 3]

        for i in row_indices, j in col_indices
            @test TensorCrossInterpolation.eval(ci, i, j) ≈ A[i, j]
            ci[i, j] ≈ A[i, j]
        end

        for i in row_indices
            @test TensorCrossInterpolation.row(ci, i)[col_indices] ≈ A[i, col_indices]
            @test ci[i, col_indices] ≈ A[i, col_indices]
        end

        for j in col_indices
            @test TensorCrossInterpolation.col(ci, j)[row_indices] ≈ A[row_indices, j]
            @test ci[row_indices, j] ≈ A[row_indices, j]
        end

        @test TensorCrossInterpolation.submatrix(ci, row_indices, col_indices) ≈
              A[row_indices, col_indices]
        @test ci[row_indices, col_indices] ≈ A[row_indices, col_indices]
        @test Matrix(ci)[row_indices, col_indices] ≈ A[row_indices, col_indices]
    end

    @testset "Finding pivots, trivial case" begin
        A = fill(1.0, 5, 3)

        ci = MatrixCI{Float64}(size(A)...)

        @test_throws DimensionMismatch add_pivot!(zeros(6, 6), ci)
        @test_throws BoundsError add_pivot!(A, ci, (6, 3))
        @test_throws BoundsError add_pivot!(A, ci, (5, 4))
        @test_throws BoundsError add_pivot!(A, ci, (-1, 2))
        @test_throws BoundsError add_pivot!(A, ci, (3, -1))
        @test_throws ArgumentError TensorCrossInterpolation.find_new_pivot(A, ci, zeros(Int64, 0), [2, 3])
        @test_throws ArgumentError TensorCrossInterpolation.find_new_pivot(A, ci, [1, 2], zeros(Int64, 0))

        @test rank(ci) == 0

        add_pivot!(A, ci, (2, 3))
        @test ci.row_indices == [2]
        @test ci.col_indices == [3]
        @test ci.pivot_rows == fill(1.0, 1, 3)
        @test ci.pivot_cols == fill(1.0, 5, 1)
        @test rank(ci) == 1
        for i in 1:5, j in 1:3
            @test TensorCrossInterpolation.eval(ci, i, j) ≈ 1.0
            @test ci[i, j] ≈ 1.0
        end
        add_pivot!(A, ci)
        @test ci.pivot_rows == fill(1.0, 2, 3)
        @test ci.pivot_cols == fill(1.0, 5, 2)
        @test length(ci.row_indices) == 2
        @test length(ci.col_indices) == 2
        @test rank(ci) == 2
        add_pivot!(A, ci, (TensorCrossInterpolation.avail_rows(ci)[1], TensorCrossInterpolation.avail_cols(ci)[1]))
        @test ci.pivot_rows == fill(1.0, 3, 3)
        @test ci.pivot_cols == fill(1.0, 5, 3)
        @test length(ci.row_indices) == 3
        @test length(ci.col_indices) == 3
        @test rank(ci) == 3
    end


    @testset "Finding pivots, rank-1 case" begin
        A = [1.0; 2.0; 3.0] * [2.0 4.0 8.0 16.0]
        ci = MatrixCI{Float64}(3, 4)

        @test local_error(A, ci) ≈ A
        @test TensorCrossInterpolation.find_new_pivot(A, ci) == ((3, 4), 48.0)
        add_pivot!(A, ci)

        @test ci ≈ MatrixCI(A, (3, 4))

        @test ci.row_indices == [3]
        @test ci.col_indices == [4]
        @test ci.pivot_rows ≈ 3.0 .* [2.0 4.0 8.0 16.0]
        @test ci.pivot_cols ≈ 16.0 .* [1.0; 2.0; 3.0]
        @test ci[:, :] ≈ A
        @test TensorCrossInterpolation.avail_rows(ci) == [1, 2]
        @test TensorCrossInterpolation.avail_cols(ci) == [1, 2, 3]

        add_pivot!(A, ci)
        @test length(ci.row_indices) == 2
        @test length(ci.col_indices) == 2
        @test unique(ci.row_indices) == ci.row_indices
        @test unique(ci.col_indices) == ci.col_indices
        @test size(ci.pivot_rows) == (2, 4)
        @test size(ci.pivot_cols) == (3, 2)
        @test ci[:, :] ≈ A

        add_pivot!(A, ci)
        # From here, the ci will become singular; checking values is useless.
        @test length(ci.row_indices) == 3
        @test length(ci.col_indices) == 3
        @test unique(ci.row_indices) == ci.row_indices
        @test unique(ci.col_indices) == ci.col_indices
        @test size(ci.pivot_rows) == (3, 4)
        @test size(ci.pivot_cols) == (3, 3)

        @test_throws ArgumentError TensorCrossInterpolation.find_new_pivot(A, ci)
        @test_throws ArgumentError add_pivot!(A, ci)
    end

    @testset "Cross interpolate smooth functions" begin
        grid = range(0, 1, length=21)

        gauss = exp.(-grid .^ 2 .- grid' .^ 2)
        cigauss = cross_interpolate(gauss)
        @test rank(cigauss) == 1
        @test n_rows(cigauss) == 21
        @test n_cols(cigauss) == 21
        @test cigauss.row_indices == [1]
        @test cigauss.col_indices == [1]

        lorentz = 1 ./ (1 .+ grid .^ 2 .+ grid' .^ 2)
        cilorentz = cross_interpolate(lorentz, tolerance=1e-6, maxiter=10)
        @test rank(cilorentz) == 5
        @test n_rows(cilorentz) == 21
        @test n_cols(cilorentz) == 21
        @test Set(cilorentz.row_indices) == Set([21, 7, 12, 17, 1])
        @test Set(cilorentz.col_indices) == Set([21, 7, 12, 17, 1])
    end
end