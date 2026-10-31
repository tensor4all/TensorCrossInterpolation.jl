using Test
import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: MultiIndex, DefaultGlobalPivotFinder, GlobalPivotSearchInput, CachedFunction
import Random
import Random: MersenneTwister

# Counts the batch calls so that the pointwise fallback cannot silently come back.
mutable struct CountingEvaluator <: TCI.BatchEvaluator{Float64}
    f
    localdims::Vector{Int}
    batches::Int
end

(ce::CountingEvaluator)(v::MultiIndex) = ce.f(v)

function (ce::CountingEvaluator)(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
) where {M}
    ce.batches += 1
    nl = length(first(leftindexset))
    nr = length(first(rightindexset))
    dims = ce.localdims[nl + 1:length(ce.localdims) - nr]
    result = Array{Float64}(undef, length(leftindexset), dims..., length(rightindexset))
    for (j, rindex) in enumerate(rightindexset)
        for (k, cindex) in enumerate(Iterators.product(ntuple(x -> 1:dims[x], M)...))
            for (i, lindex) in enumerate(leftindexset)
                result[i, k, j] = ce.f(vcat(lindex, collect(Iterators.flatten(cindex)), rindex))
            end
        end
    end
    return result
end

@testset "global pivot finder: batched local search" begin
    R = 6
    localdims = fill(4, R)
    f(v) = exp(-0.3 * sum(v)) + 0.05 * sin(3.0 * sum(i * v[i] for i in 1:length(v)))
    firstpivots = [ones(Int, R), vcat(1, fill(4, R - 1))]
    tci, _, _ = TCI.crossinterpolate2(
        Float64, f, localdims, firstpivots;
        tolerance=1e-12, maxbonddim=2, verbosity=0, normalizeerror=false
    )
    input = GlobalPivotSearchInput(tci)
    finder = DefaultGlobalPivotFinder(nsearch=4, maxnglobalpivot=5, tolmarginglobalsearch=10.0)
    cached = CachedFunction{Float64}(f, localdims)

    # A BatchEvaluator must reach the batched path ...
    counting = CountingEvaluator(f, localdims, 0)
    pivots_batched = finder(input, counting, 1e-6; rng=MersenneTwister(7))
    @test !isempty(pivots_batched)
    @test counting.batches == finder.nsearch * R

    # ... and a plain callable the pointwise path.
    pointwise_only = CountingEvaluator(f, localdims, 0)
    pivots_pointwise = finder(input, v -> pointwise_only(v), 1e-6; rng=MersenneTwister(7))
    @test pointwise_only.batches == 0

    # Same RNG state, same start points, same pivots in the same order.
    @test pivots_batched == pivots_pointwise
    @test pivots_batched == finder(input, cached, 1e-6; rng=MersenneTwister(7))

    # The selected points are the local maxima of the error along each coordinate.
    for point in pivots_batched
        @test abs(f(point) - input.current_tt(point)) > 1e-6 * finder.tolmarginglobalsearch
    end
end
