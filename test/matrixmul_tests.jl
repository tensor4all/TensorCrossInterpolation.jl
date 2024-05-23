using Test
using LinearAlgebra
import TensorCrossInterpolation as TCI
import TCIAlgorithms: MatrixProduct
import TCIAlgorithms as TCIA
using TCIITensorConversion

using ITensors

#==
==#

function _tomat(tto::TCI.TensorTrain{T,4}) where {T}
    sitedims = TCI.sitedims(tto)
    localdims1 = [s[1] for s in sitedims]
    localdims2 = [s[2] for s in sitedims]
    mat = Matrix{T}(undef, prod(localdims1), prod(localdims2))
    for (i, inds1) in enumerate(CartesianIndices(Tuple(localdims1)))
        for (j, inds2) in enumerate(CartesianIndices(Tuple(localdims2)))
            mat[i, j] = TCI.evaluate(tto, collect(zip(Tuple(inds1), Tuple(inds2))))
        end
    end
    return mat
end

@testset "MPO-MPO naive contraction" begin
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    a = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1])
        for n = 1:N
    ])

    ab = TCIA.naivecontract(a, b)

    sites1 = Index.(localdims1, "1")
    sites2 = Index.(localdims2, "2")
    sites3 = Index.(localdims3, "3")

    #amps = MPO(a, sites = collect(zip(sites1, sites2)))
    #bmps = MPO(b, sites = collect(zip(sites2, sites3)))
    #abmps = amps * bmps

    @test _tomat(ab) ≈ _tomat(a) * _tomat(b)

    #for inds1 in CartesianIndices(Tuple(localdims1))
        #for inds3 in CartesianIndices(Tuple(localdims3))
            #refvalue = evaluate_mps(
                    #abmps,
                    #collect(zip(sites1, Tuple(inds1))),
                    #collect(zip(sites3, Tuple(inds3))),
                #)
            #inds = collect(zip(Tuple(inds1), Tuple(inds3)))
            #@test ab(inds) ≈ refvalue
        #end
    #end
end

@testset "MPO-MPO contraction" for f in [x -> x, x -> 2 * x]
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    a = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
        for n = 1:N
    ])
    b = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1])
        for n = 1:N
    ])

    ab = TCIA.contract(a, b; f = f)
    @test TCI.sitedims(ab) == [[localdims1[i], localdims3[i]] for i = 1:N]
    @test _tomat(ab) ≈ f.(_tomat(a) * _tomat(b))
end