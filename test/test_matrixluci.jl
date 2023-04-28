using Test
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: MatrixLUCI

@testset "Test MatrixLUCI" begin
    @testset "Approximation in LUCI" begin
        A = [
            0.684025 0.784249 0.826742 0.054321 0.0234695 0.46709
            0.73928 0.295516 0.877126 0.111711 0.103509 0.653785
            0.394016 0.753239 0.889128 0.291669 0.873509 0.0965536
            0.378539 0.0123737 0.20112 0.758088 0.973042 0.308372
            0.235156 0.51939 0.788184 0.363171 0.230001 0.984971
            0.893223 0.220834 0.18001 0.258537 0.396583 0.142105
            0.0417881 0.890706 0.328631 0.279332 0.963188 0.706944
            0.914298 0.792345 0.311083 0.129653 0.350062 0.683966
        ]

        luci = TCI.MatrixLUCI(A, maxrank=4)
        @test size(luci) == size(A)
        @test length(TCI.rowindices(luci)) == 4
        @test length(TCI.colindices(luci)) == 4

        ci = TCI.MatrixCI(
            TCI.rowindices(luci),
            TCI.colindices(luci),
            A[:, TCI.colindices(luci)],
            A[TCI.rowindices(luci), :]
        )
        @test TCI.colstimespivotinv(luci) ≈ TCI.leftmatrix(ci)
        @test TCI.pivotinvtimesrows(luci) ≈ TCI.rightmatrix(ci)

        L = TCI.left(luci)
        @test size(L) == (size(A, 1), 4)
        U = TCI.right(luci)
        @test size(U) == (4, size(A, 2))
        @test size(L, 2) == size(U, 1)
        @test L * U ≈ ci[:, :]

        # Artificially create a low-rank matrix
        A = hcat(A, A .+ 1e-3 * rand(8, 6))
        luci = TCI.MatrixLUCI(A, reltol=1e-2)
        @test size(luci) == size(A)
        @test length(TCI.rowindices(luci)) < size(A, 1)
        @test length(TCI.colindices(luci)) < size(A, 2)
        @test maximum(abs.(TCI.left(luci) * TCI.right(luci) .- A)) < 1e-2
    end
end
