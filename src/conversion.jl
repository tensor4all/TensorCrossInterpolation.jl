function MatrixACA(lu::rrLU{T}) where {T}
    aca = MatrixACA(T, size(lu)...)
    aca.rowindices = rowindices(lu)
    aca.colindices = colindices(lu)
    aca.u = left(lu)
    aca.v = right(lu)
    aca.alpha = 1 ./ diag(lu)

    if lu.leftorthogonal
        # Set the permuted diagonal of aca.u to the diagonal elements of lu.
        for j in axes(aca.u, 2)
            aca.u[:, j] *= diag(lu)[j]
        end
    else
        # Same with aca.v
        for i in axes(aca.v, 1)
            aca.v[i, :] *= diag(lu)[i]
        end
    end
    return aca
end

function TensorCI{ValueType}(
    tci2::TensorCI2{ValueType},
    f;
    kwargs...
) where {ValueType}
    L = length(tci2)
    tci1 = TensorCI{ValueType}(length.(tci2.localset))
    tci1.Iset = IndexSet.(tci2.Iset)
    tci1.Jset = IndexSet.(tci2.Jset)
    tci1.PiIset = getPiIset.(Ref(tci1), 1:L)
    tci1.PiJset = getPiJset.(Ref(tci1), 1:L)
    tci1.Pi = getPi.(Ref(tci1), 1:L-1, f)

    for ell in 1:L-1
        # Get indices along axes of Pi to reconstruct T, P
        iset = pos(tci1.PiIset[ell], tci1.Iset[ell+1].fromint)
        jset = pos(tci1.PiJset[ell+1], tci1.Jset[ell].fromint)
        updateT!(tci1, ell, tci1.Pi[ell][:, jset])
        if ell == L - 1
            updateT!(tci1, L, tci1.Pi[ell][iset, :])
        end
        tci1.P[ell] = tci1.Pi[ell][iset, jset]

        tci1.aca[ell] = MatrixACA(tci1.Pi[ell], (iset[1], jset[1]))
        for (rowindex, colindex) in zip(iset[2:end], jset[2:end])
            addpivotcol!(tci1.aca[ell], tci1.Pi[ell], colindex)
            addpivotrow!(tci1.aca[ell], tci1.Pi[ell], rowindex)
        end
    end
    tci1.P[end] = ones(ValueType, 1, 1)

    tci1.pivoterrors = max.(tci2.bonderrorsforward, tci2.bonderrorsbackward)
    tci1.maxsamplevalue = tci2.maxsamplevalue
    return tci1
end

function TensorCI2{ValueType}(tci1::TensorCI{ValueType}) where {ValueType}
    tci2 = TensorCI2{ValueType}(vcat(sitedims(tci1)...)::Vector{Int})
    tci2.Iset = [i.fromint for i in tci1.Iset]
    tci2.Jset = [j.fromint for j in tci1.Jset]
    tci2.localset = tci1.localset
    L = length(tci1)
    tci2.sitetensors[1:L-1] = TtimesPinv.(tci1, 1:L-1)
    tci2.sitetensors[end] = tci1.T[end]
    tci2.pivoterrors = Float64[]
    tci2.bonderrorsforward = tci1.pivoterrors
    tci2.bonderrorsbackward = tci1.pivoterrors
    tci2.maxsamplevalue = tci1.maxsamplevalue
    return tci2
end
