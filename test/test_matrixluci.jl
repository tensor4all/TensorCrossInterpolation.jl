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

    @testset "luci for exact low-rank matrix" begin
        p = [
            0.284975 0.505168 0.570921
            0.302884 0.475901 0.645776
            0.622955 0.361755 0.99539
            0.748447 0.354849 0.431366
            0.28338 0.0378148 0.994162
            0.643177 0.74173 0.802733
            0.58113 0.526715 0.879048
            0.238002 0.557812 0.251512
            0.458861 0.141355 0.0306212
            0.490269 0.810266 0.7946
        ]
        q = [
            0.239552   0.306094  0.299063  0.0382492  0.185462  0.0334971  0.697561   0.389596  0.105665  0.0912763
            0.0570609  0.56623   0.97183   0.994184   0.371695  0.284437   0.993251   0.902347  0.572944  0.0531369
            0.45002    0.461168  0.6086    0.613702   0.543997  0.759954   0.0959818  0.638499  0.407382  0.482592
        ]

        A = p * q
        luci = TCI.luci(A)

        @test TCI.npivots(luci) == 3
        @test TCI.left(luci) * TCI.right(luci) ≈ A
        pivotmatrix = TCI.colmatrix(luci)[1:npivtos(luci), :]
        @test cond(pivotmatrix) < 1e12
    end
end
