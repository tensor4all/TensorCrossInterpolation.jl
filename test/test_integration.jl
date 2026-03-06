using Test
import TensorCrossInterpolation as TCI
import Random

@testset "Integration" begin
    @testset "Integrate real polynomials" begin
        Random.seed!(1234)

        coefficients = [
            0.23637074801483304, 0.20661524945577847, 0.1850826417895819, 0.8433788714289417,
            0.5801482873508491, 0.20339438932656262, 0.21593267492457668, 0.8052490409622802,
            0.7189346124875339, 0.9400806688257749, 0.355210845205325, 0.5251561513473092,
            0.6819965273401778, 0.9221987248861162, 0.04166444723413998
        ]

        polynomial(x::Float64) = sum(c * x^(i - 1) for (i, c) in enumerate(coefficients))
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

    @testset "Integrate complex polynomials" begin
        Random.seed!(1234)

        coefficients = [
            0.3593310036882851 + 0.5576449076512986im,
            0.21621263269455804 + 0.08800235200842366im,
            0.46808012645915487 + 0.1415909017229775im,
            0.24698289079616775 + 0.9658935330334624im,
            0.4554497486419905 + 0.7984680137635275im,
            0.6957883252881866 + 0.9570499781505035im,
            0.289419938556415 + 0.9984881496050377im,
            0.9577173946390758 + 0.5442255738586897im,
            0.7034120251166891 + 0.3168670299990256im,
            0.7305752395989986 + 0.14383153865593656im
        ]

        polynomial(x::Float64) = sum(c * x^(i - 1) for (i, c) in enumerate(coefficients))
        polynomialintegral(x::Float64) = sum(c * x^i / i for (i, c) in enumerate(coefficients))
        f(x::Vector{Float64}) = prod(polynomial.(x))

        N = 5
        exactval = polynomialintegral(1.0)^N
        @test TCI.integrate(ComplexF64, f, fill(0.0, N), fill(1.0, N)) ≈ exactval

        b = rand(N)
        a = rand(N)
        exactval = prod(polynomialintegral.(b) .- polynomialintegral.(a))
        @test TCI.integrate(ComplexF64, f, a, b) ≈ exactval
    end


    @testset "Integrate 10d function" begin
        Random.seed!(1234)

        function f(x)
            return 1000 * cos(10 * sum(x .^ 2)) * exp(-sum(x)^4 / 1000)
        end
        I15 = TCI.integrate(Float64, f, fill(-1.0, 10), fill(+1.0, 10); GKorder=15, tolerance=1e-8)
        Iref = -5.4960415218049
        @test abs(I15 - Iref) < 1e-3
    end

end
