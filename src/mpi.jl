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
    if mute # Mute all processors which have rank != 0.
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
