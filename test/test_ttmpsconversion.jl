@testset "TT to MPS conversion" begin
    tt = TCI.TensorTrain([rand(1, 4, 4), rand(4, 4, 2), rand(2, 4, 7), rand(7, 4, 1)])
    mps = ITensors.MPS(tt)
    @test ITensors.linkdims(mps) == [4, 2, 7]
end

@testset "TT to MPO conversion and back" begin
    tt = TCI.TensorTrain([rand(1, 4, 3, 4), rand(4, 2, 4, 2), rand(2, 5, 1, 7), rand(7, 9, 4, 1)])
    mpo = ITensors.MPO(tt)
    @test ITensors.linkdims(mpo) == [4, 2, 7]
    @test ITensors.dim.(ITensors.siteinds(mpo)[1]) == [4, 3]
    @test ITensors.dim.(ITensors.siteinds(mpo)[2]) == [2, 4]
    @test ITensors.dim.(ITensors.siteinds(mpo)[3]) == [5, 1]
    @test ITensors.dim.(ITensors.siteinds(mpo)[4]) == [9, 4]

    tt2 = TCI.TensorTrain{Float64,4}(mpo)
    @test all(tt2.T .== tt.T)

    sites = reverse.(ITensors.siteinds(mpo))
    ttreverse = TCI.TensorTrain{Float64,4}(mpo, sites=sites)
    @test all(permutedims.(ttreverse.T, Ref([1, 3, 2, 4])) .== tt.T)
end
