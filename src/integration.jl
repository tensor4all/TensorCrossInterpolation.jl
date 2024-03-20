function integrate(
    ::Type{ValueType},
    f,
    a::Vector{ValueType},
    b::Vector{ValueType};
    tolerance=1e-8,
    order::Int=15
) where {ValueType}
    @assert isodd(order)
    @assert length(a) == length(b)
    nodes1d, weights1d, _ = QuadGK.kronrod(order ÷ 2, -1, +1)
    nodes = @. (b - a) * (nodes1d' + 1) / 2 + a
    weights = @. (b - a) * weights1d' / 2

    localdims = fill(length(nodes1d), length(a))

    function F(indices)
        x = [nodes[n, i] for (n, i) in enumerate(indices)]
        w = prod(weights[n, i] for (n, i) in enumerate(indices))
        return w * f(x)
    end

    tci2, ranks, errors = crossinterpolate2(
        ValueType,
        F,
        localdims;
        tolerance
    )

    return sum(tci2)
end
