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

function TensorCI1{ValueType}(
    tci2::TensorCI2{ValueType},
    f;
    kwargs...
) where {ValueType}
    L = length(tci2)
    tci1 = TensorCI1{ValueType}(tci2.localdims)
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

    tci1.pivoterrors = tci2.bonderrors
    tci1.maxsamplevalue = tci2.maxsamplevalue
    return tci1
end

function TensorCI2{ValueType}(tci1::TensorCI1{ValueType}) where {ValueType}
    tci2 = TensorCI2{ValueType}(vcat(collect.(sitedims(tci1))...)::Vector{Int})
    tci2.Iset = [i.fromint for i in tci1.Iset]
    tci2.Jset = [j.fromint for j in tci1.Jset]
    tci2.localdims = tci1.localdims
    tci2.pivoterrors = Float64[]
    tci2.bonderrors = tci1.pivoterrors
    tci2.maxsamplevalue = tci1.maxsamplevalue
    return tci2
end

function sweep1sitegetindices!(
    tt::TensorTrain{ValueType,3}, forwardsweep::Bool,
    spectatorindices::Vector{Vector{MultiIndex}}=Vector{MultiIndex}[];
    maxbonddim=typemax(Int), tolerance=0.0
) where {ValueType}
    indexset = Vector{MultiIndex}[MultiIndex[[]]]
    pivoterrorsarray = zeros(rank(tt) + 1)

    function groupindices(T::AbstractArray, next::Bool)
        shape = size(T)
        if forwardsweep != next
            reshape(T, prod(shape[1:end-1]), shape[end])
        else
            reshape(T, shape[1], prod(shape[2:end]))
        end
    end

    function splitindices(T::AbstractArray, shape, newbonddim, next::Bool)
        if forwardsweep != next
            newshape = (shape[1:end-1]..., newbonddim)
        else
            newshape = (newbonddim, shape[2:end]...)
        end
        reshape(T, newshape)
    end

    L = length(tt)
    for i in 1:L-1
        ell = forwardsweep ? i : L - i + 1
        ellnext = forwardsweep ? i + 1 : L - i
        shape = size(tt.sitetensors[ell])
        shapenext = size(tt.sitetensors[ellnext])

        luci = MatrixLUCI(
            groupindices(tt.sitetensors[ell], false), leftorthogonal=forwardsweep,
            abstol=tolerance, maxrank=maxbonddim
        )

        if forwardsweep
            push!(indexset, kronecker(last(indexset), shape[2])[rowindices(luci)])
            if !isempty(spectatorindices)
                spectatorindices[ell] = spectatorindices[ell][colindices(luci)]
            end
        else
            push!(indexset, kronecker(shape[2], last(indexset))[colindices(luci)])
            if !isempty(spectatorindices)
                spectatorindices[ell] = spectatorindices[ell][rowindices(luci)]
            end
        end


        tt.sitetensors[ell] = splitindices(
            forwardsweep ? left(luci) : right(luci),
            shape, npivots(luci), false
        )

        nexttensor = (
            forwardsweep
            ? right(luci) * groupindices(tt.sitetensors[ellnext], true)
            : groupindices(tt.sitetensors[ellnext], true) * left(luci)
        )

        tt.sitetensors[ellnext] = splitindices(nexttensor, shapenext, npivots(luci), true)
        pivoterrorsarray[1:npivots(luci) + 1] = max.(pivoterrorsarray[1:npivots(luci) + 1], pivoterrors(luci))
    end

    if forwardsweep
        return indexset, pivoterrorsarray
    else
        return reverse(indexset), pivoterrorsarray
    end
end

function TensorCI2{ValueType}(
    tt::TensorTrain{ValueType,3}; tolerance=1e-12, maxbonddim=typemax(Int), maxiter=3
) where {ValueType}
    local pivoterrors::Vector{Float64}

    Iset, = sweep1sitegetindices!(tt, true; maxbonddim, tolerance)
    Jset, pivoterrors = sweep1sitegetindices!(tt, false; maxbonddim, tolerance)

    for iter in 3:maxiter
        if isodd(iter)
            Isetnew, pivoterrors = sweep1sitegetindices!(tt, true, Jset)
            if Isetnew == Iset
                break
            end
        else
            Jsetnew, pivoterrors = sweep1sitegetindices!(tt, false, Iset)
            if Jsetnew == Jset
                break
            end
        end
    end

    tci2 = TensorCI2{ValueType}(first.(sitedims(tt)))
    tci2.Iset = Iset
    tci2.Jset = Jset
    tci2.pivoterrors = pivoterrors
    tci2.maxsamplevalue = maximum(maximum.(abs, tt.sitetensors))

    return tci2
end
