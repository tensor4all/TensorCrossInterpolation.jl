using Test
import TensorCrossInterpolation: SweepStrategies, forwardsweep

@testset "SweepStrategies" begin
    iters = 1:5
    @test forwardsweep.(SweepStrategies.forward, iters) == fill(true, 5)
    @test forwardsweep.(SweepStrategies.backward, iters) == fill(false, 5)
    @test forwardsweep.(SweepStrategies.backandforth, iters) == isodd.(iters)
end
