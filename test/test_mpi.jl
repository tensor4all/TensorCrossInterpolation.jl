using MPI
using Test
import TensorCrossInterpolation as TCI

@testset "MPI" begin
    @testset "MPI init" begin
        @test !MPI.Initialized()
        TCI.initializempi(false)
        @test MPI.Initialized()
    end

    @testset "MPI communicator" begin
        comm = MPI.COMM_WORLD
        nprocs = MPI.Comm_size(comm)
        mpirank = MPI.Comm_rank(comm)
        @test nprocs ≥ 1
        @test 0 ≤ mpirank < nprocs
    end

    @testset "synchronize_tt!" begin
        comm = MPI.COMM_WORLD
        nprocs = MPI.Comm_size(comm)
        mpirank = MPI.Comm_rank(comm)
        juliarank = mpirank + 1
        
        tolerance = 1e-8
        maxbonddim = 50

        function f0(vec)
            cos(prod(vec) * pi / 2^6)
        end
        function f(vec)
            cos(prod(vec.^juliarank) * pi / 2^6)
        end

        tt0, _, _ = TCI.crossinterpolate2(Float64, f0, fill(2, 6); tolerance, maxbonddim)
        tt, _, _ = TCI.crossinterpolate2(Float64, f, fill(2, 6); tolerance, maxbonddim)

        maxerr = 0.
        for x1 in 1:2, x2 in 1:2, x3 in 1:2, x4 in 1:2, x5 in 1:2, x6 in 1:2
            maxerr = max(maxerr, abs(TCI.evaluate(tt0, [x1,x2,x3,x4,x5,x6]) - TCI.evaluate(tt, [x1,x2,x3,x4,x5,x6])))
        end
        if mpirank == 0
            @test maxerr < 1e-8
        else
            @test maxerr > 1e-8
        end

        juliasource = 1
        TCI.synchronize_tt!(tt; juliasource=juliasource)

        maxerr = 0.
        for x1 in 1:2, x2 in 1:2, x3 in 1:2, x4 in 1:2, x5 in 1:2, x6 in 1:2
            maxerr = max(maxerr, abs(TCI.evaluate(tt0, [x1,x2,x3,x4,x5,x6]) - TCI.evaluate(tt, [x1,x2,x3,x4,x5,x6])))
        end
        @test maxerr < 1e-8
    end
end