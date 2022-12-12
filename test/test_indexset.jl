@testset "IndexSet tests" begin
    is = TensorCrossInterpolation.IndexSet{Vector{Int}}()
    @test is.to_int == Dict{Vector{Int64},Int64}()
    @test is.from_int == []
    @test length(is) == 0
    @test isempty(is)
    @test is == TensorCrossInterpolation.IndexSet{Vector{Int}}()

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
        @test is.to_int[L[i]] == i
        @test is.from_int[i] == L[i]
    end

    @test length(is) == length(L)
    @test !isempty(is)
    @test is == TensorCrossInterpolation.IndexSet(L)
end