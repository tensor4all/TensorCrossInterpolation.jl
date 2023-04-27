"""
    module SweepStrategies

Wraps the enum `SweepStrategy` that represents different ways to perform sweeps during TCI
optimization. The enum has the following entries:
- `SweepStrategy.back_and_forth`
- `SweepStrategy.forward`
- `SweepStrategy.backward`
"""
module SweepStrategies
@enum SweepStrategy begin
    backandforth
    forward
    backward
end
end

function forwardsweep(sweepstrategy::SweepStrategies.SweepStrategy, iteration::Int)
    return (
        (sweepstrategy == SweepStrategies.forward) ||
        (sweepstrategy == SweepStrategies.backandforth && isodd(iteration))
    )
end
