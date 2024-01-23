using Aqua
import TensorCrossInterpolation as TCI

@testset "Aqua" begin
    Aqua.test_stale_deps(TCI)
end
