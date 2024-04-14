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

@testset "_contract" begin
    a = rand(2, 3, 4)
    b = rand(2, 5, 4)
    ab = TCI._contract(a, b, (1, 3), (1, 3))
    @test vec(reshape(permutedims(a, (2, 1, 3)), 3, :) * reshape(permutedims(b, (1, 3, 2)), :, 5)) ≈ vec(ab)
end

@testset "MPO-MPO contraction" for f in [nothing, x -> 2 * x], algorithm in ["TCI", "naive"]
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

    if f === nothing || algorithm != "naive"
        ab = contract(a, b; f=f, algorithm=algorithm)
        @test sitedims(ab) == [[localdims1[i], localdims3[i]] for i = 1:N]
        if f === nothing
            @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
        else
            @test _tomat(ab) ≈ f.(_tomat(a) * _tomat(b))
        end
    end
end