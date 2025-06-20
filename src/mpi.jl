"""
    function initializempi()

Initialize the MPI environment if it has not been initialized yet. If mute=true, then all the processes with rank>0 (i.e. not the root node) won't output anything to stdout.
"""
function initializempi(mute::Bool=true)
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(comm)
    if mute # Mute all processors which have mpirank != 0.
        if mpirank != 0
            open("/dev/null", "w") do devnull
                redirect_stdout(devnull)
                redirect_stderr(devnull)
            end
        end
    end
end

"""
    function finalizempi()

Finalize the MPI environment if it has not been finalized yet.
"""
function finalizempi()
    MPI.Barrier(MPI.COMM_WORLD)
    if MPI.Comm_rank == 0
        if !MPI.Finalized()
            MPI.Finalize()
        end
    end
end

function synchronize_tt!(tt::Union{TensorTrain{ValueType,N},TensorCI2{ValueType}}; subcomm = nothing, juliasource::Int = 1) where {ValueType, N}
    if !MPI.Initialized()
        println("Warning! synchronize_tt has been called, but MPI is not initialized, please use TCI.initializempi() before contract() and use TCI.finalizempi() afterwards")
    else
        if subcomm != nothing
            comm = subcomm
        else
            comm = MPI.COMM_WORLD
        end
        mpirank = MPI.Comm_rank(comm)
        juliarank = mpirank + 1
        nprocs = MPI.Comm_size(comm)

        MPI.Barrier(comm)
        reqs = MPI.Request[]
        if juliarank == juliasource
            for j in 1:nprocs
                if j != juliarank
                    push!(reqs, MPI.isend(tt.sitetensors[1:end], comm; dest=j-1, tag=juliarank))
                end
            end
        end
        if juliarank != juliasource
            tt.sitetensors[1:end] = MPI.recv(comm; source=juliasource-1, tag=juliasource)
        end
        MPI.Waitall(reqs)
    end
end