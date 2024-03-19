using Test
import TensorCrossInterpolation: forwardsweep

@testset "SweepStrategies" begin
    iters = 1:5
    @test forwardsweep.(:forward, iters) == fill(true, 5)
    @test forwardsweep.(:backward, iters) == fill(false, 5)
    @test forwardsweep.(:backandforth, iters) == isodd.(iters)
end
