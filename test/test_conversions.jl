using Test
import TensorCrossInterpolation: TensorCI, TensorCI2, sitedims, linkdims, rank,
    addglobalpivot!, SweepStrategies, crossinterpolate, optimize!

@testset "Conversion between TCI1 and TCI2" begin
    d = 5
    n = 5
    tci1 = TensorCI{ComplexF64}(fill(d, n))
    tci2 = TensorCI2{ComplexF64}(tci1)

    @test length(tci2) == length(tci1)
    @test sitedims(tci2) == sitedims(tci1)
    @test rank(tci2) == 0
    @test all(isempty.(tci2.Iset))
    @test all(isempty.(tci2.Jset))
    @test all(isempty.(tci2.T))

    globalpivot = [2, 4, 3, 5, 5]
    tci1 = TensorCI{ComplexF64}(rand, fill(d, n), globalpivot)
    tci2 = TensorCI2{ComplexF64}(tci1)
    @test rank(tci2) == 1
    @test linkdims(tci2) == linkdims(tci1)

    f(v) = (1.0 + 2.0im) ./ (sum(v .^ 2) + 1)
    tci1, ranks, errors = crossinterpolate(
        ComplexF64,
        f,
        fill(d, n),
        ones(Int, n);
        tolerance=1e-6,
        pivottolerance=1e-8,
        maxiter=8,
        sweepstrategy=SweepStrategies.forward
    )
    tci2 = TensorCI2{ComplexF64}(tci1)
    # tci1_backconverted = TensorCI(tci2)
    # tci2_backconverted = TensorCI2(tci1_backconverted)
    @test rank(tci2) == rank(tci1)
    # @test rank(tci1_backconverted) == rank(tci1)
    # @test rank(tci2_backconverted) == rank(tci2)
    @test linkdims(tci2) == linkdims(tci1)
    # @test linkdims(tci1_backconverted) == linkdims(tci1)
    # @test linkdims(tci2_backconverted) == linkdims(tci2)
    for v in Iterators.product([1:d for p in 1:n]...)
        @test tci1(v) ≈ tci2(v)
        # @test tci1(v) ≈ tci1_backconverted(v)
    end

    print(optimize!(tci2, f; tolerance=1e-12))
    @test last(tci2.pivoterrors) <= 1e-12
    @test rank(tci2) > rank(tci1)
    for v in Iterators.product([1:d for p in 1:n]...)
        @test tci2(v) ≈ f(v)
    end
end
