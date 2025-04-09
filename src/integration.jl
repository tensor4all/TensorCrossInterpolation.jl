"""
    function integrate(
        ::Type{ValueType},
        f,
        a::Vector{ValueType},
        b::Vector{ValueType};
        tolerance=1e-8,
        GKorder::Int=15
    ) where {ValueType}

Integrate the function `f` using TCI and Gauss--Kronrod quadrature rules.

Arguments:
- `ValueType``: return type of `f`.
- `a``: Vector of lower bounds in each dimension. Effectively, the lower corner of the hypercube that is being integrated over.
- `b``: Vector of upper bounds in each dimension.
- `tolerance`: tolerance of the TCI approximation for the values of f.
- `GKorder`: Order of the Gauss--Kronrod rule, e.g. 15.
"""
function integrate(
    ::Type{ValueType},
    f,
    a::Vector{ValueType},
    b::Vector{ValueType};
    GKorder::Int=15,
    kwargs...
) where {ValueType}
    if iseven(GKorder)
        error("Gauss--Kronrod order must be odd, e.g. 15 or 61.")
    end

    if length(a) != length(b)
        error("Integral bounds must have the same dimensionality, but got $(length(a)) lower bounds and $(length(b)) upper bounds.")
    end

    nodes1d, weights1d, _ = QuadGK.kronrod(GKorder ÷ 2, -1, +1)
    nodes = @. (b - a) * (nodes1d' + 1) / 2 + a
    weights = @. (b - a) * weights1d' / 2
    normalization = GKorder^length(a)

    localdims = fill(length(nodes1d), length(a))

    function F(indices)
        x = [nodes[n, i] for (n, i) in enumerate(indices)]
        w = prod(weights[n, i] for (n, i) in enumerate(indices))
        return w * f(x) * normalization
    end

    tci2, ranks, errors = crossinterpolate2(
        ValueType,
        F,
        localdims;
        nsearchglobalpivot=10,
        kwargs...
    )

    return sum(tci2) / normalization
end