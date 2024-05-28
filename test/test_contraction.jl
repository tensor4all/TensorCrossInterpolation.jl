import TensorCrossInterpolation as TCI
using TensorCrossInterpolation
using Random

function _tomat(tto::TensorTrain{T,4}) where {T}
    sitedims = TCI.sitedims(tto)
    localdims1 = [s[1] for s in sitedims]
    localdims2 = [s[2] for s in sitedims]
    mat = Matrix{T}(undef, prod(localdims1), prod(localdims2))
    for (i, inds1) in enumerate(CartesianIndices(Tuple(localdims1)))
        for (j, inds2) in enumerate(CartesianIndices(Tuple(localdims2)))
            mat[i, j] = evaluate(tto, collect(zip(Tuple(inds1), Tuple(inds2))))
        end
    end
    return mat
end

function _tovec(tt::TensorTrain{T,3}) where {T}
    sitedims = TCI.sitedims(tt)
    localdims1 = [s[1] for s in sitedims]
    return evaluate.(Ref(tt), CartesianIndices(Tuple(localdims1))[:])
end

@testset "_contract" begin
    a = rand(2, 3, 4)
    b = rand(2, 5, 4)
    ab = TCI._contract(a, b, (1, 3), (1, 3))
    @test vec(reshape(permutedims(a, (2, 1, 3)), 3, :) * reshape(permutedims(b, (1, 3, 2)), :, 5)) ≈ vec(ab)
end

function _gen_testdata_TTO_TTO()
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    a = TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1])
        for n = 1:N
    ])
    return N, a, b, localdims1, localdims2, localdims3
end

function _gen_testdata_TTO_TTS()
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [3, 3, 3, 3]
    localdims2 = [3, 3, 3, 3]

    a = TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TensorTrain{ComplexF64,3}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], bonddims_b[n+1])
        for n = 1:N
    ])
    return N, a, b, localdims1, localdims2
end

@testset "MPO-MPO contraction" for f in [nothing, x -> 2 * x], algorithm in [:TCI, :naive]
    #==
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    a = TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1])
        for n = 1:N
    ])
    ==#
    N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()

    if f !== nothing && algorithm === :naive
        @test_throws ErrorException contract(a, b; f=f, algorithm=algorithm)
    else
        ab = contract(a, b; f=f, algorithm=algorithm)
        @test sitedims(ab) == [[localdims1[i], localdims3[i]] for i = 1:N]
        if f === nothing
            @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
        else
            @test _tomat(ab) ≈ f.(_tomat(a) * _tomat(b))
        end
    end
end

@testset "MPO-MPS contraction" for f in [nothing, x -> 2 * x], algorithm in [:TCI, :naive]
    #==
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [3, 3, 3, 3]
    localdims2 = [3, 3, 3, 3]

    a = TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TensorTrain{ComplexF64,3}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], bonddims_b[n+1])
        for n = 1:N
    ])
    ==#
    N, a, b, localdims1, localdims2 = _gen_testdata_TTO_TTS()

    if f !== nothing && algorithm === :naive
        @test_throws ErrorException contract(a, b; f=f, algorithm=algorithm)
        @test_throws ErrorException contract(b, a; f=f, algorithm=algorithm)
    else
        ab = contract(a, b; f=f, algorithm=algorithm)
        ba = contract(b, a; f=f, algorithm=algorithm)
        @test sitedims(ab) == [[localdims1[i]] for i = 1:N]
        if f === nothing
            @test _tovec(ab) ≈ _tomat(a) * _tovec(b)
            @test transpose(_tovec(ba)) ≈ transpose(_tovec(b)) * _tomat(a)
        else
            @test _tovec(ab) ≈ f.(_tomat(a) * _tovec(b))
            @test transpose(_tovec(ba)) ≈ f.(transpose(_tovec(b)) * _tomat(a))
        end
    end
end


@testset "MPO-MPO contraction (zipup)" for method in [:SVD, :LU]
    N, a, b, localdims1, localdims2, localdims3 = _gen_testdata_TTO_TTO()
    ab = contract(a, b; algorithm=:zipup, method=method)
    @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
end

@testset "MPO-MPS contraction (zipup)" for method in [:SVD, :LU]
    N, a, b, localdims1, localdims2 = _gen_testdata_TTO_TTS()
    ab = contract(a, b; algorithm=:zipup, method=method)
    @test _tovec(ab) ≈ _tomat(a) * _tovec(b)
end