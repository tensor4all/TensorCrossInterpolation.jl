function forwardsweep(sweepstrategy::Symbol, iteration::Int)
    return (
        (sweepstrategy == :forward) ||
        (sweepstrategy == :backandforth && isodd(iteration))
    )
end