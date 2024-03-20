using Test
import TensorCrossInterpolation as TCI
import Random

@testset "Integration" begin
    Random.seed!(1234)

    coefficients = [
        0.23637074801483304, 0.20661524945577847, 0.1850826417895819, 0.8433788714289417,
        0.5801482873508491, 0.20339438932656262, 0.21593267492457668, 0.8052490409622802,
        0.7189346124875339, 0.9400806688257749, 0.355210845205325, 0.5251561513473092,
        0.6819965273401778, 0.9221987248861162, 0.04166444723413998
    ]

    polynomial(x::Float64) = sum(c * x^(i-1) for (i, c) in enumerate(coefficients))
    polynomialintegral(x::Float64) = sum(c * x^i / i for (i, c) in enumerate(coefficients))
    f(x::Vector{Float64}) = prod(polynomial.(x))

    N = 5
    exactval = polynomialintegral(1.0)^N
    @test TCI.integrate(Float64, f, fill(0.0, N), fill(1.0, N)) ≈ exactval

    b = rand(N)
    a = rand(N)
    exactval = prod(polynomialintegral.(b) .- polynomialintegral.(a))
    @test TCI.integrate(Float64, f, a, b) ≈ exactval
end
