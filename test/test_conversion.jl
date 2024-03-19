using Test
import TensorCrossInterpolation: TensorCI1, TensorCI2, sitedims, linkdims, rank,
    addglobalpivot!, crossinterpolate, optimize!, MatrixACA, rrlu,
    nrows, ncols, evaluate, left, right

@testset "Conversion between rrLU and ACA" begin
    A = [
        0.412779  0.423091   0.166912  0.953768    0.207438   0.600653
        0.273203  0.622319   0.715224  0.646002    0.0508133  0.482628
        0.562037  0.0616797  0.455742  0.00227183  0.411564   0.345012
        0.537797  0.955916   0.656385  0.463868    0.449098   0.146251
        0.245995  0.77942    0.389488  0.714201    0.416509   0.00404971
        0.604805  0.0745451  0.228923  0.881908    0.0640686  0.514265
    ]

    lu = rrlu(A; maxrank=4)
    aca = MatrixACA(lu)
    @test nrows(aca) == 6
    @test ncols(aca) == 6
    @test evaluate(aca) ≈ left(lu) * right(lu)
end

@testset "Conversion between TCI1 and TCI2" begin
    d = 3
    n = 4
    tci1 = TensorCI1{ComplexF64}(fill(d, n))
    tci2 = TensorCI2{ComplexF64}(tci1)

    @test length(tci2) == length(tci1)
    @test sitedims(tci2) == sitedims(tci1)
    @test rank(tci2) == 0
    @test all(isempty.(tci2.Iset))
    @test all(isempty.(tci2.Jset))
    @test all(isempty.(tci2.T))

    globalpivot = [2, 2, 3, 1]
    tci1 = TensorCI1{ComplexF64}(rand, fill(d, n), globalpivot)
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
        maxiter=4,
        sweepstrategy=:forward
    )
    tci2 = TensorCI2{ComplexF64}(tci1)
    tci1_backconverted = TensorCI1{ComplexF64}(tci2, f)
    tci2_backconverted = TensorCI2{ComplexF64}(tci1_backconverted)
    @test rank(tci2) == rank(tci1)
    @test rank(tci1_backconverted) == rank(tci1)
    @test rank(tci2_backconverted) == rank(tci2)
    @test linkdims(tci2) == linkdims(tci1)
    @test linkdims(tci1_backconverted) == linkdims(tci1)
    @test linkdims(tci2_backconverted) == linkdims(tci2)
    for v in Iterators.product([1:d for p in 1:n]...)
        @test tci1(v) ≈ tci2(v)
        @test tci1(v) ≈ tci1_backconverted(v)
    end

    optimize!(tci2, f; tolerance=1e-12)
    @test pivoterror(tci2) <= 1e-12
    @test rank(tci2) > rank(tci1)
    for v in Iterators.product([1:d for p in 1:n]...)
        @test tci2(v) ≈ f(v)
    end
end
