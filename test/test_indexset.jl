using Test
import TensorCrossInterpolation as TCI

@testset "IndexSet tests" begin
    is = TCI.IndexSet{Vector{Int}}()
    @test is.toint == Dict{Vector{Int64},Int64}()
    @test is.fromint == []
    @test length(is) == 0
    @test isempty(is)
    @test is == TCI.IndexSet{Vector{Int}}()

    L = [
        [6, 0, 9, 1, 0],
        [8, 7, 4, 7, 6],
        [1, 8, 4, 3, 0],
        [3, 7, 1, 6, 8],
        [7, 7, 0, 6, 0],
        [8, 3, 6, 0, 9],
        [1, 1, 4, 7, 0],
        [9, 6, 9, 9, 5],
        [1, 8, 5, 9, 9],
        [6, 3, 6, 4, 6],
    ]

    for i in eachindex(L)
        push!(is, L[i])
        @test is[i] == L[i]
        @test is.toint[L[i]] == i
        @test is.fromint[i] == L[i]
    end

    @test length(is) == length(L)
    @test !isempty(is)
    @test is == TCI.IndexSet(L)
end

@testset "IndexSet nested" begin
    is1 = [[1,], [2,]]

    is2 = [[1, 4], [2, 3]]

    @test TCI.isnested(is1, is2)

    is3 = [[4, 1], [3, 2]]

    @test TCI.isnested(is1, is3, :col)
end
