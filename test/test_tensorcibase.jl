using Test
import TensorCrossInterpolation as TCI

@testset "TensorCIBase" begin
    @testset "Rank-1 matrix" begin
        # Rank-1 matrix
        x = [1.0, 2.0]
        y = transpose([1.0, 3.0])
        A = x * y

        t = TCI.TensorCIBase{Float64}(idx->A[idx[1], idx[2]], [2, 2], [1, 1])
        @test [TCI.evaluate(t, [i,j]) for i in 1:2, j in 1:2] ≈ A
    end
end
