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

@testset "global pivot finder: maxnglobalpivot = 0 skips the search" begin
    R = 6
    localdims = fill(4, R)
    f(v) = exp(-0.3 * sum(v)) + 0.05 * sin(3.0 * sum(i * v[i] for i in 1:length(v)))
    firstpivots = [ones(Int, R), vcat(1, fill(4, R - 1))]
    tci, _, _ = TCI.crossinterpolate2(
        Float64, f, localdims, firstpivots;
        tolerance=1e-12, maxbonddim=2, verbosity=0, normalizeerror=false
    )
    input = GlobalPivotSearchInput(tci)

    ncalls = Ref(0)
    counted = v -> (ncalls[] += 1; f(v))
    pivots = DefaultGlobalPivotFinder(nsearch=4, maxnglobalpivot=0)(input, counted, 1e-6; rng=MersenneTwister(7))
    @test isempty(pivots)
    @test ncalls[] == 0

    # nsearch = 0 has always meant the same thing and must keep working.
    ncalls0 = Ref(0)
    counted0 = v -> (ncalls0[] += 1; f(v))
    pivots0 = DefaultGlobalPivotFinder(nsearch=0, maxnglobalpivot=5)(input, counted0, 1e-6; rng=MersenneTwister(7))
    @test isempty(pivots0)
    @test ncalls0[] == 0
end

@testset "global pivots: maxnglobalpivot = 0 costs the same as nsearchglobalpivot = 0" begin
    L = 4
    d = fill(6, L)
    g(v) = exp(-0.5 * sum(abs2, (v .- 3) ./ 3)) + 0.1 * prod(cos.(v))
    function count_evals(kw)
        n = Ref(0)
        h = v -> (n[] += 1; g(v))
        TCI.crossinterpolate2(Float64, h, d, [ones(Int, L)];
            tolerance=1e-8, maxiter=4, verbosity=0, normalizeerror=false, kw...)
        return n[]
    end
    @test count_evals((; maxnglobalpivot=0)) ==
        count_evals((; maxnglobalpivot=0, nsearchglobalpivot=0))
end

@testset "global pivots: the rejection line counts true rejections" begin
    h(v) = (x = (v .- 1) ./ 64; exp(-sum(abs2, x .- 0.2) / 0.002) + exp(-sum(abs2, x .- 0.8) / 0.002))
    path, io = mktemp()
    try
        redirect_stdout(io) do
            TCI.crossinterpolate2(Float64, h, fill(64, 3), [ones(Int, 3)];
                tolerance=1e-8, maxiter=3, verbosity=1, loginterval=1)
        end
    finally
        close(io)
    end
    log = read(path, String)
    rm(path; force=true)
    # A global pivot is only rejected when its error at the next iteration is still above abstol.
    @test !occursin("Rejected", log)
    @test occursin("nglobalpivot=5", log)
end
